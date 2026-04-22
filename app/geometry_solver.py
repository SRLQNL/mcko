from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

from app.logger import logger

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = (20, 180)
JSON_MAX_TOKENS = 2200

DEFAULT_KIMI_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_QWEN_MODEL = "qwen/qwen2.5-vl-72b-instruct"
DEFAULT_LLAMA_MODEL = "meta-llama/llama-4-maverick"
DEFAULT_MODE = "cheap"

_log = logging.getLogger("mcko.geometry_solver")

KIMI_SYSTEM_PROMPT = (
    "You solve geometry and stereometry problems from photos. Output JSON only. "
    "Prioritize correct diagram interpretation, extract givens and target, solve carefully, "
    "and do not invent objects or relations. If the diagram is ambiguous, report it explicitly."
)

QWEN_SYSTEM_PROMPT = (
    "You are a visual parser for geometry photos. Output JSON only. "
    "Extract OCR text, diagram entities, diagram relations, givens, target, and ambiguities. "
    "Do not optimize for solving. Lower confidence instead of guessing."
)

LLAMA_SYSTEM_PROMPT = (
    "You are a verifier for geometry photo problems. Output JSON only. "
    "Independently interpret the diagram, verify givens and target, and check the proposed solution. "
    "Do not invent missing relations. If the image is ambiguous, report it explicitly."
)


class GeometryPhotoSolver:
    def __init__(
        self,
        api_key: str,
        kimi_model: str = DEFAULT_KIMI_MODEL,
        qwen_model: str = DEFAULT_QWEN_MODEL,
        llama_model: str = DEFAULT_LLAMA_MODEL,
        mode: str = DEFAULT_MODE,
    ):
        self.api_key = api_key
        self.kimi_model = kimi_model
        self.qwen_model = qwen_model
        self.llama_model = llama_model
        self.mode = mode if mode in ("cheap", "accurate") else DEFAULT_MODE
        _log.info(
            "GeometryPhotoSolver initialized: mode=%s kimi=%s qwen=%s llama=%s",
            self.mode,
            self.kimi_model,
            self.qwen_model,
            self.llama_model,
        )

    def solve_content_blocks(self, content_blocks: List[Dict]) -> str:
        image_urls, user_text = self._extract_image_payload(content_blocks)
        if not image_urls:
            _log.info("Running text-only solver path with Kimi only")
            kimi_result = self._call_kimi_text_only(user_text)
            return self._format_text_only_result(kimi_result)

        preprocessed = self._prepare_variants(image_urls[0]) if image_urls else {
            "full_image": None,
            "text_crop": None,
            "diagram_crop": None,
        }
        _log.info(
            "Prepared request variants: has_image=%s text_crop=%s diagram_crop=%s",
            bool(image_urls),
            preprocessed.get("text_crop") is not None,
            preprocessed.get("diagram_crop") is not None,
        )

        qwen_result = self._call_qwen(preprocessed, user_text)
        kimi_result = self._call_kimi(preprocessed, user_text, qwen_result, None)

        if self.mode == "cheap" and self._can_accept_cheap(qwen_result, kimi_result):
            consensus = self._build_cheap_consensus(qwen_result, kimi_result)
            _log.info("Cheap mode accepted without llama: score=%.3f", consensus["score"])
            return self._format_user_result(consensus, kimi_result, qwen_result, None)

        llama_result = self._call_llama(preprocessed, user_text, qwen_result, kimi_result, None)
        consensus = self._compare_results(kimi_result, qwen_result, llama_result)
        _log.info("Consensus after llama: status=%s score=%.3f", consensus["status"], consensus["score"])

        if consensus["status"] == "self_check":
            mismatch_summary = self._build_mismatch_summary(consensus, kimi_result, qwen_result, llama_result)
            _log.info("Running self-check round: %s", mismatch_summary)
            kimi_check = self._call_kimi(preprocessed, user_text, qwen_result, mismatch_summary)
            llama_check = self._call_llama(preprocessed, user_text, qwen_result, kimi_check, mismatch_summary)
            consensus = self._compare_results(kimi_check, qwen_result, llama_check)
            kimi_result = kimi_check
            llama_result = llama_check
            _log.info("Consensus after self-check: status=%s score=%.3f", consensus["status"], consensus["score"])

        return self._format_user_result(consensus, kimi_result, qwen_result, llama_result)

    def _extract_image_payload(self, content_blocks: List[Dict]) -> Tuple[List[str], str]:
        image_urls = []
        text_parts = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image_url":
                image_url = (block.get("image_url") or {}).get("url")
                if image_url:
                    image_urls.append(image_url)
            elif block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    text_parts.append(text)
        return image_urls, "\n".join(text_parts)

    def _prepare_variants(self, image_url: str) -> Dict[str, Optional[str]]:
        variants = {
            "full_image": image_url,
            "text_crop": None,
            "diagram_crop": None,
        }
        image_bytes = self._data_url_to_bytes(image_url)
        if image_bytes is None:
            _log.warning("Could not decode image data URL, using full image only")
            return variants

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                image = img.convert("RGB")
                width, height = image.size
                _log.info("Preparing image variants: width=%d height=%d", width, height)
                if height >= int(width * 1.15) and height >= 300:
                    split_y = int(height * 0.45)
                    text_crop = image.crop((0, 0, width, split_y))
                    diagram_crop = image.crop((0, split_y, width, height))
                    variants["text_crop"] = self._image_to_data_url(text_crop)
                    variants["diagram_crop"] = self._image_to_data_url(diagram_crop)
        except Exception as exc:
            _log.error("Image preprocessing failed: %s", exc)
        return variants

    def _call_qwen(self, variants: Dict[str, Optional[str]], user_text: str) -> Dict:
        user_content = self._build_qwen_content(variants, user_text)
        result = self._request_json(self.qwen_model, QWEN_SYSTEM_PROMPT, user_content)
        return self._normalize_result(result)

    def _call_kimi_text_only(self, user_text: str) -> Dict:
        prompt = (
            "Solve the geometry or stereometry problem from text and return strict JSON.\n"
            "Extract givens, target, concise reasoning, final answer, confidence, and ambiguities.\n"
            "If the problem statement is incomplete or ambiguous, report it explicitly.\n"
        )
        if user_text:
            prompt += "Problem text:\n%s" % user_text
        result = self._request_json(
            self.kimi_model,
            KIMI_SYSTEM_PROMPT,
            [{"type": "text", "text": prompt}],
        )
        return self._normalize_result(result)

    def _call_kimi(
        self,
        variants: Dict[str, Optional[str]],
        user_text: str,
        qwen_result: Dict,
        mismatch_summary: Optional[str],
    ) -> Dict:
        user_content = self._build_kimi_content(variants, user_text, qwen_result, mismatch_summary)
        result = self._request_json(self.kimi_model, KIMI_SYSTEM_PROMPT, user_content)
        return self._normalize_result(result)

    def _call_llama(
        self,
        variants: Dict[str, Optional[str]],
        user_text: str,
        qwen_result: Dict,
        kimi_result: Dict,
        mismatch_summary: Optional[str],
    ) -> Dict:
        user_content = self._build_llama_content(variants, user_text, qwen_result, kimi_result, mismatch_summary)
        result = self._request_json(self.llama_model, LLAMA_SYSTEM_PROMPT, user_content)
        return self._normalize_result(result)

    def _build_qwen_content(self, variants: Dict[str, Optional[str]], user_text: str) -> List[Dict]:
        has_image = bool(variants.get("full_image"))
        if has_image:
            text_prompt = (
                "Parse this geometry photo into strict JSON.\n"
                "Extract OCR text, entities, relations, givens, target, ambiguities, and confidence.\n"
                "Do not solve unless needed for normalization."
            )
        else:
            text_prompt = (
                "Parse this geometry or stereometry problem text into strict JSON.\n"
                "Extract givens, target, inferred entities, relations, ambiguities, and confidence.\n"
                "Do not optimize for solving."
            )
        if user_text:
            text_prompt += "\nUser hint:\n%s" % user_text

        content = [{"type": "text", "text": text_prompt}]
        if has_image:
            content.append({"type": "image_url", "image_url": {"url": variants["full_image"]}})
        if variants.get("text_crop"):
            content.append({"type": "text", "text": "Top crop likely contains problem text."})
            content.append({"type": "image_url", "image_url": {"url": variants["text_crop"]}})
        if variants.get("diagram_crop"):
            content.append({"type": "text", "text": "Bottom crop likely contains the diagram."})
            content.append({"type": "image_url", "image_url": {"url": variants["diagram_crop"]}})
        return content

    def _build_kimi_content(
        self,
        variants: Dict[str, Optional[str]],
        user_text: str,
        qwen_result: Dict,
        mismatch_summary: Optional[str],
    ) -> List[Dict]:
        has_image = bool(variants.get("full_image"))
        if has_image:
            prompt = (
                "Solve the geometry problem from the image and return strict JSON.\n"
                "Use the Qwen visual parse as a helper, but correct it if the image clearly disagrees.\n"
                "Prioritize correct diagram interpretation over aggressive solving.\n"
            )
        else:
            prompt = (
                "Solve the geometry or stereometry problem from text and return strict JSON.\n"
                "Use the Qwen parse as a helper, but reason independently.\n"
                "Prioritize faithful interpretation of givens and target.\n"
            )
        if user_text:
            prompt += "User hint:\n%s\n" % user_text
        prompt += "Qwen parse:\n%s\n" % json.dumps(qwen_result, ensure_ascii=False)
        if mismatch_summary:
            prompt += "Self-check mismatch summary:\n%s\n" % mismatch_summary

        content = [{"type": "text", "text": prompt}]
        if has_image:
            content.append({"type": "image_url", "image_url": {"url": variants["full_image"]}})
        return content

    def _build_llama_content(
        self,
        variants: Dict[str, Optional[str]],
        user_text: str,
        qwen_result: Dict,
        kimi_result: Dict,
        mismatch_summary: Optional[str],
    ) -> List[Dict]:
        has_image = bool(variants.get("full_image"))
        if has_image:
            prompt = (
                "Verify the geometry problem from the image and return strict JSON.\n"
                "Check the diagram interpretation, givens, target, and final answer.\n"
                "Do not blindly copy Kimi. If unsure, lower confidence or mark ambiguity.\n"
            )
        else:
            prompt = (
                "Verify the geometry or stereometry problem from text and return strict JSON.\n"
                "Check givens, target, reasoning, and final answer.\n"
                "Do not blindly copy Kimi. If unsure, lower confidence or mark ambiguity.\n"
            )
        if user_text:
            prompt += "User hint:\n%s\n" % user_text
        prompt += "Qwen parse:\n%s\n" % json.dumps(qwen_result, ensure_ascii=False)
        prompt += "Kimi result:\n%s\n" % json.dumps(kimi_result, ensure_ascii=False)
        if mismatch_summary:
            prompt += "Self-check mismatch summary:\n%s\n" % mismatch_summary

        content = [{"type": "text", "text": prompt}]
        if has_image:
            content.append({"type": "image_url", "image_url": {"url": variants["full_image"]}})
        return content

    def _request_json(self, model: str, system_prompt: str, user_content: List[Dict]) -> Dict:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "max_tokens": JSON_MAX_TOKENS,
            "provider": {"allow_fallbacks": True},
        }
        _log.info("Requesting geometry JSON: model=%s blocks=%d", model, len(user_content))
        response = requests.post(
            ENDPOINT,
            headers={
                "Authorization": "Bearer %s" % self.api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if not response.ok:
            body = response.text[:500]
            request_id = response.headers.get("x-request-id") or response.headers.get("cf-ray") or "-"
            _log.error("Geometry solver API error: model=%s status=%d request_id=%s body=%s", model, response.status_code, request_id, body)
            raise RuntimeError("[Ошибка API %d: %s]" % (response.status_code, body))

        data = response.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        raw_text = message.get("content")
        if raw_text is None:
            raw_text = message.get("reasoning") or ""
        parsed = self._extract_json_object(raw_text)
        _log.info("Geometry JSON parsed: model=%s chars=%d", model, len(raw_text))
        return parsed

    def _normalize_result(self, raw: Dict) -> Dict:
        result = {
            "task_type": raw.get("task_type") or "geometry_photo",
            "ocr_text": raw.get("ocr_text") or "",
            "normalized_problem_text": raw.get("normalized_problem_text") or "",
            "diagram_entities": raw.get("diagram_entities") or raw.get("objects") or [],
            "diagram_relations": raw.get("diagram_relations") or raw.get("relations") or [],
            "givens": raw.get("givens") or [],
            "target": raw.get("target") or {"statement": ""},
            "visual_interpretation": raw.get("visual_interpretation") or {
                "summary": raw.get("visual_summary") or "",
                "confidence": self._coerce_confidence(raw.get("visual_interpretation_confidence")),
                "possible_ambiguities": raw.get("possible_ambiguities") or [],
            },
            "reasoning_summary": raw.get("reasoning_summary") or [],
            "solution_steps": raw.get("solution_steps") or [],
            "final_answer": raw.get("final_answer") or {"value": "", "format": "text"},
            "answer_confidence": self._coerce_confidence(raw.get("answer_confidence")),
            "consistency_checks": raw.get("consistency_checks") or [],
            "needs_clarification": bool(raw.get("needs_clarification", False)),
        }
        if isinstance(result["target"], str):
            result["target"] = {"statement": result["target"]}
        if isinstance(result["final_answer"], str):
            result["final_answer"] = {"value": result["final_answer"], "format": "text"}
        visual = result["visual_interpretation"]
        if isinstance(visual, str):
            result["visual_interpretation"] = {
                "summary": visual,
                "confidence": 0.5,
                "possible_ambiguities": [],
            }
        else:
            visual["summary"] = visual.get("summary") or ""
            visual["confidence"] = self._coerce_confidence(visual.get("confidence"))
            visual["possible_ambiguities"] = visual.get("possible_ambiguities") or []
        final_answer = result["final_answer"]
        final_answer["value"] = "" if final_answer.get("value") is None else str(final_answer.get("value"))
        final_answer["format"] = str(final_answer.get("format") or "text")
        return result

    def _compare_results(self, kimi: Dict, qwen: Dict, llama: Dict) -> Dict:
        qwen_diagram = self._jaccard(self._relation_keys(kimi), self._relation_keys(qwen))
        llama_diagram = self._jaccard(self._relation_keys(kimi), self._relation_keys(llama))
        qwen_givens = self._jaccard(self._given_keys(kimi), self._given_keys(qwen))
        llama_givens = self._jaccard(self._given_keys(kimi), self._given_keys(llama))
        target_agreement = 1.0 if self._normalize_text(kimi["target"].get("statement", "")) == self._normalize_text(qwen["target"].get("statement", "")) else 0.0
        answer_agreement = 1.0 if self._normalize_text(kimi["final_answer"].get("value", "")) == self._normalize_text(llama["final_answer"].get("value", "")) else 0.0

        diagram_agreement = (qwen_diagram + llama_diagram) / 2.0
        givens_agreement = (qwen_givens + llama_givens) / 2.0
        confidence_alignment = (
            kimi["visual_interpretation"]["confidence"] +
            qwen["visual_interpretation"]["confidence"] +
            llama["visual_interpretation"]["confidence"]
        ) / 3.0

        score = (
            0.35 * diagram_agreement +
            0.20 * givens_agreement +
            0.15 * target_agreement +
            0.20 * answer_agreement +
            0.10 * confidence_alignment
        )
        reasons = []
        if diagram_agreement < 0.5:
            reasons.append("low diagram agreement")
        if givens_agreement < 0.6:
            reasons.append("low givens agreement")
        if answer_agreement < 1.0:
            reasons.append("answer mismatch")
        if qwen["visual_interpretation"]["possible_ambiguities"]:
            reasons.append("diagram ambiguity detected")

        status = "ambiguous"
        accepted = False
        if score >= 0.85 and diagram_agreement >= 0.5:
            status = "accepted"
            accepted = True
        elif score >= 0.65:
            status = "self_check"

        return {
            "accepted": accepted,
            "score": score,
            "status": status,
            "final_answer": kimi["final_answer"].get("value", ""),
            "diagram_agreement": diagram_agreement,
            "givens_agreement": givens_agreement,
            "answer_agreement": answer_agreement,
            "reasons": reasons,
        }

    def _build_cheap_consensus(self, qwen: Dict, kimi: Dict) -> Dict:
        score = (
            0.55 * qwen["visual_interpretation"]["confidence"] +
            0.45 * kimi["answer_confidence"]
        )
        return {
            "accepted": True,
            "score": score,
            "status": "accepted",
            "final_answer": kimi["final_answer"].get("value", ""),
            "diagram_agreement": qwen["visual_interpretation"]["confidence"],
            "givens_agreement": 1.0,
            "answer_agreement": 1.0,
            "reasons": [],
        }

    def _can_accept_cheap(self, qwen: Dict, kimi: Dict) -> bool:
        if qwen["visual_interpretation"]["possible_ambiguities"]:
            return False
        if qwen["visual_interpretation"]["confidence"] < 0.78:
            return False
        if kimi["visual_interpretation"]["confidence"] < 0.80:
            return False
        if kimi["answer_confidence"] < 0.82:
            return False
        if kimi["needs_clarification"]:
            return False
        return True

    def _build_mismatch_summary(self, consensus: Dict, kimi: Dict, qwen: Dict, llama: Dict) -> str:
        parts = [
            "consensus_score=%.3f" % consensus["score"],
            "reasons=%s" % ", ".join(consensus["reasons"]),
            "kimi_answer=%s" % kimi["final_answer"].get("value", ""),
            "llama_answer=%s" % llama["final_answer"].get("value", ""),
            "qwen_ambiguities=%s" % "; ".join(qwen["visual_interpretation"]["possible_ambiguities"]),
        ]
        return "\n".join([part for part in parts if part.strip()])

    def _format_user_result(self, consensus: Dict, kimi: Dict, qwen: Dict, llama: Optional[Dict]) -> str:
        final_answer = kimi["final_answer"].get("value", "").strip()
        if consensus["status"] == "accepted" and final_answer:
            return "1) %s" % final_answer

        ambiguity_parts = list(qwen["visual_interpretation"]["possible_ambiguities"])
        if llama is not None:
            ambiguity_parts.extend(llama["visual_interpretation"]["possible_ambiguities"])
        ambiguity_parts = [part for part in ambiguity_parts if part]
        ambiguity_text = "; ".join(dict.fromkeys(ambiguity_parts)) if ambiguity_parts else "интерпретация рисунка ненадёжна"

        if final_answer:
            return "1) Низкая уверенность: %s. Предварительный ответ: %s" % (ambiguity_text, final_answer)
        return "1) Низкая уверенность: %s" % ambiguity_text

    def _extract_json_object(self, raw_text: str) -> Dict:
        text = raw_text.strip()
        if not text:
            raise ValueError("Empty JSON response")

        try:
            return json.loads(text)
        except ValueError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response: %s" % text[:200])
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except ValueError:
            repaired = self._repair_json(candidate)
            return json.loads(repaired)

    def _repair_json(self, text: str) -> str:
        repaired = text
        # Remove trailing commas before closing braces/brackets.
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        # Close obviously unterminated braces/brackets.
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        if close_braces < open_braces:
            repaired += "}" * (open_braces - close_braces)
        open_brackets = repaired.count("[")
        close_brackets = repaired.count("]")
        if close_brackets < open_brackets:
            repaired += "]" * (open_brackets - close_brackets)
        # Remove markdown fences if the model wrapped JSON in them.
        repaired = repaired.replace("```json", "").replace("```", "").strip()
        _log.warning("Applied JSON repair before parsing model output")
        return repaired

    def _coerce_confidence(self, value) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.5
        if number < 0:
            return 0.0
        if number > 1:
            return 1.0
        return number

    def _relation_keys(self, result: Dict) -> List[str]:
        keys = []
        for relation in result.get("diagram_relations", []):
            if not isinstance(relation, dict):
                continue
            rel_type = relation.get("type", "")
            subject = relation.get("subject", "")
            obj = relation.get("object", "")
            key = "%s:%s:%s" % (rel_type, subject, obj)
            if key not in keys:
                keys.append(key)
        return keys

    def _given_keys(self, result: Dict) -> List[str]:
        keys = []
        for given in result.get("givens", []):
            if isinstance(given, dict):
                statement = given.get("statement", "")
            else:
                statement = str(given)
            statement = self._normalize_text(statement)
            if statement and statement not in keys:
                keys.append(statement)
        return keys

    def _jaccard(self, left: List[str], right: List[str]) -> float:
        left_set = set(left)
        right_set = set(right)
        union = left_set | right_set
        if not union:
            return 1.0
        return float(len(left_set & right_set)) / float(len(union))

    def _normalize_text(self, text: str) -> str:
        if text is None:
            return ""
        return " ".join(str(text).strip().lower().split())

    def _format_text_only_result(self, kimi_result: Dict) -> str:
        final_answer = kimi_result["final_answer"].get("value", "").strip()
        if final_answer and not kimi_result.get("needs_clarification"):
            return "1) %s" % final_answer

        ambiguities = kimi_result["visual_interpretation"].get("possible_ambiguities") or []
        ambiguity_text = "; ".join(ambiguities) if ambiguities else "условие понято неоднозначно"
        if final_answer:
            return "1) Низкая уверенность: %s. Предварительный ответ: %s" % (ambiguity_text, final_answer)
        return "1) Низкая уверенность: %s" % ambiguity_text

    def _data_url_to_bytes(self, data_url: str) -> Optional[bytes]:
        if not data_url.startswith("data:image/"):
            return None
        try:
            encoded = data_url.split(",", 1)[1]
            return base64.b64decode(encoded)
        except Exception as exc:
            _log.error("Failed to decode data URL: %s", exc)
            return None

    def _image_to_data_url(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return "data:image/png;base64,%s" % encoded
