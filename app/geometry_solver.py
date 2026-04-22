from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = (20, 90)
JSON_MAX_TOKENS = 2200

DEFAULT_KIMI_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_QWEN_MODEL = "qwen/qwen2.5-vl-72b-instruct"
DEFAULT_LLAMA_MODEL = "meta-llama/llama-4-maverick"
DEFAULT_MODE = "cheap"
RETRYABLE_STATUSES = (408, 429, 502, 503, 504)

_log = logging.getLogger("mcko.geometry_solver")

JSON_SCHEMA_NOTE = (
    'Return one valid JSON object with keys: '
    '"task_type","ocr_text","normalized_problem_text","diagram_entities","diagram_relations",'
    '"givens","target","visual_interpretation","reasoning_summary","solution_steps",'
    '"final_answer","answer_confidence","consistency_checks","needs_clarification". '
    'Use double quotes. No markdown. No prose outside JSON.'
)

KIMI_SYSTEM_PROMPT = (
    "You are the primary geometry and stereometry solver. "
    "Reason as fully as needed internally, but output only strict JSON. "
    "Prioritize correct interpretation of the problem and image over speed. "
    "Do not invent objects or relations. "
    "If the problem is solvable, final_answer.value must be non-empty and contain the answer only. "
    + JSON_SCHEMA_NOTE
)

QWEN_SYSTEM_PROMPT = (
    "You are the parser and visual verifier for geometry and stereometry tasks. "
    "Extract OCR text, entities, relations, givens, target, and ambiguities. "
    "Do not optimize for solving. Lower confidence instead of guessing. "
    "Leave final_answer empty unless the answer is explicitly printed in the source itself. "
    + JSON_SCHEMA_NOTE
)

LLAMA_SYSTEM_PROMPT = (
    "You are the independent verifier for geometry and stereometry tasks. "
    "Re-check interpretation, givens, target, reasoning, and final answer. "
    "Do not blindly copy the proposed answer. "
    + JSON_SCHEMA_NOTE
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
        if not image_urls and not user_text.strip():
            return "1) Не удалось определить ответ"

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
        try:
            result = self._request_json(self.qwen_model, QWEN_SYSTEM_PROMPT, user_content)
            return self._normalize_result(result, role="parser")
        except RuntimeError as exc:
            _log.warning("Qwen parser unavailable, using degraded parser fallback: %s", exc)
            return self._fallback_parser_result(user_text, variants)

    def _call_kimi(
        self,
        variants: Dict[str, Optional[str]],
        user_text: str,
        qwen_result: Dict,
        mismatch_summary: Optional[str],
    ) -> Dict:
        user_content = self._build_kimi_content(variants, user_text, qwen_result, mismatch_summary)
        result = self._request_json(self.kimi_model, KIMI_SYSTEM_PROMPT, user_content)
        return self._normalize_result(result, role="solver")

    def _call_llama(
        self,
        variants: Dict[str, Optional[str]],
        user_text: str,
        qwen_result: Dict,
        kimi_result: Dict,
        mismatch_summary: Optional[str],
    ) -> Dict:
        user_content = self._build_llama_content(variants, user_text, qwen_result, kimi_result, mismatch_summary)
        try:
            result = self._request_json(self.llama_model, LLAMA_SYSTEM_PROMPT, user_content)
            return self._normalize_result(result, role="verifier")
        except RuntimeError as exc:
            _log.warning("Llama verifier unavailable, using degraded verifier fallback: %s", exc)
            return self._fallback_verifier_result(kimi_result)

    def _build_qwen_content(self, variants: Dict[str, Optional[str]], user_text: str) -> List[Dict]:
        has_image = bool(variants.get("full_image"))
        if has_image:
            text_prompt = (
                "Parse this geometry photo into strict JSON.\n"
                "Extract OCR text, entities, relations, givens, target, ambiguities, and confidence.\n"
                "Normalize labels and spatial relations. Do not solve unless needed for normalization.\n"
                "%s" % JSON_SCHEMA_NOTE
            )
        else:
            text_prompt = (
                "Parse this geometry or stereometry problem text into strict JSON.\n"
                "Extract givens, target, inferred entities, relations, ambiguities, and confidence.\n"
                "Do not optimize for solving.\n"
                "%s" % JSON_SCHEMA_NOTE
            )
        if user_text:
            text_prompt += "\nUser hint:\n%s" % user_text

        content = [{"type": "text", "text": text_prompt}]
        content.extend(self._build_image_blocks(variants))
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
                "Keep full reasoning inside solution_steps and reasoning_summary.\n"
                "%s\n" % JSON_SCHEMA_NOTE
            )
        else:
            prompt = (
                "Solve the geometry or stereometry problem from text and return strict JSON.\n"
                "Use the Qwen parse as a helper, but reason independently.\n"
                "Prioritize faithful interpretation of givens and target.\n"
                "Keep full reasoning inside solution_steps and reasoning_summary.\n"
                "%s\n" % JSON_SCHEMA_NOTE
            )
        if user_text:
            prompt += "User hint:\n%s\n" % user_text
        prompt += "Qwen parse:\n%s\n" % json.dumps(qwen_result, ensure_ascii=False)
        if mismatch_summary:
            prompt += "Self-check mismatch summary:\n%s\n" % mismatch_summary

        content = [{"type": "text", "text": prompt}]
        content.extend(self._build_image_blocks(variants))
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
                "Keep full reasoning inside solution_steps and reasoning_summary.\n"
                "%s\n" % JSON_SCHEMA_NOTE
            )
        else:
            prompt = (
                "Verify the geometry or stereometry problem from text and return strict JSON.\n"
                "Check givens, target, reasoning, and final answer.\n"
                "Do not blindly copy Kimi. If unsure, lower confidence or mark ambiguity.\n"
                "Keep full reasoning inside solution_steps and reasoning_summary.\n"
                "%s\n" % JSON_SCHEMA_NOTE
            )
        if user_text:
            prompt += "User hint:\n%s\n" % user_text
        prompt += "Qwen parse:\n%s\n" % json.dumps(qwen_result, ensure_ascii=False)
        prompt += "Kimi result:\n%s\n" % json.dumps(kimi_result, ensure_ascii=False)
        if mismatch_summary:
            prompt += "Self-check mismatch summary:\n%s\n" % mismatch_summary

        content = [{"type": "text", "text": prompt}]
        content.extend(self._build_image_blocks(variants))
        return content

    def _build_image_blocks(self, variants: Dict[str, Optional[str]]) -> List[Dict]:
        content = []
        if variants.get("full_image"):
            content.append({"type": "image_url", "image_url": {"url": variants["full_image"]}})
        if variants.get("text_crop"):
            content.append({"type": "text", "text": "Auxiliary crop: likely problem text region."})
            content.append({"type": "image_url", "image_url": {"url": variants["text_crop"]}})
        if variants.get("diagram_crop"):
            content.append({"type": "text", "text": "Auxiliary crop: likely diagram region."})
            content.append({"type": "image_url", "image_url": {"url": variants["diagram_crop"]}})
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
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "provider": {"allow_fallbacks": True},
        }
        _log.info("Requesting geometry JSON: model=%s blocks=%d", model, len(user_content))
        response = None
        last_error = None
        for attempt in range(1, 3):
            started_at = time.monotonic()
            try:
                response = requests.post(
                    ENDPOINT,
                    headers={
                        "Authorization": "Bearer %s" % self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.Timeout as exc:
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                last_error = RuntimeError("[Таймаут OpenRouter: модель %s не ответила вовремя]" % model)
                _log.error("Geometry solver timeout: model=%s attempt=%d elapsed_ms=%d timeout=%s error=%s", model, attempt, elapsed_ms, REQUEST_TIMEOUT, exc)
                if attempt < 2:
                    self._sleep_before_retry(attempt)
                    continue
                raise last_error
            except requests.RequestException as exc:
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                last_error = RuntimeError("[Ошибка сети OpenRouter: %s]" % exc)
                _log.error("Geometry solver request failed: model=%s attempt=%d elapsed_ms=%d error=%s", model, attempt, elapsed_ms, exc)
                if attempt < 2:
                    self._sleep_before_retry(attempt)
                    continue
                raise last_error

            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            _log.info("Geometry solver HTTP response: model=%s attempt=%d status=%d elapsed_ms=%d", model, attempt, response.status_code, elapsed_ms)
            if response.ok:
                break

            body = response.text[:500]
            request_id = response.headers.get("x-request-id") or response.headers.get("cf-ray") or "-"
            _log.error("Geometry solver API error: model=%s attempt=%d status=%d request_id=%s body=%s", model, attempt, response.status_code, request_id, body)
            last_error = RuntimeError("[Ошибка API %d: %s]" % (response.status_code, body))
            if self._is_retryable_status(response.status_code) and attempt < 2:
                self._sleep_before_retry(attempt)
                continue
            raise last_error

        if response is None:
            raise last_error or RuntimeError("[Ошибка API: пустой ответ]")

        data = response.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        raw_text = message.get("content")
        if raw_text is None:
            raw_text = message.get("reasoning") or ""
        try:
            parsed = self._extract_json_object(raw_text)
        except ValueError as exc:
            _log.warning("Primary JSON parse failed for model=%s: %s", model, exc)
            repaired_text = self._repair_non_json_response(model, raw_text)
            parsed = self._extract_json_object(repaired_text)
            _log.info("JSON repair pass validated successfully: model=%s chars=%d", model, len(repaired_text))
        _log.info("Geometry JSON parsed: model=%s chars=%d", model, len(raw_text))
        return parsed

    def _is_retryable_status(self, status_code: int) -> bool:
        return status_code in RETRYABLE_STATUSES

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = 1.0 if attempt <= 1 else 2.0
        time.sleep(delay)

    def _repair_non_json_response(self, model: str, raw_text: str) -> str:
        repair_prompt = (
            "Convert the following model output into one strict JSON object without losing meaning.\n"
            "%s\n"
            "Original output:\n%s" % (JSON_SCHEMA_NOTE, raw_text)
        )
        _log.info("Requesting JSON repair pass: model=%s", model)
        response = requests.post(
            ENDPOINT,
            headers={
                "Authorization": "Bearer %s" % self.api_key,
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You repair malformed outputs into strict JSON only."},
                    {"role": "user", "content": [{"type": "text", "text": repair_prompt}]},
                ],
                "stream": False,
                "max_tokens": 1800,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "provider": {"allow_fallbacks": True},
            },
            timeout=REQUEST_TIMEOUT,
        )
        if not response.ok:
            body = response.text[:300]
            raise ValueError("JSON repair failed: %s" % body)
        data = response.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        repaired_text = message.get("content") or ""
        return repaired_text

    def _normalize_result(self, raw: Dict, role: str = "generic") -> Dict:
        result = {
            "task_type": raw.get("task_type") or "geometry_photo",
            "ocr_text": raw.get("ocr_text") or "",
            "normalized_problem_text": raw.get("normalized_problem_text") or "",
            "diagram_entities": self._normalize_entities(raw.get("diagram_entities") or raw.get("objects") or []),
            "diagram_relations": self._normalize_relations(raw.get("diagram_relations") or raw.get("relations") or []),
            "givens": self._normalize_givens(raw.get("givens") or []),
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

        if role == "parser":
            result["final_answer"] = {"value": "", "format": "text"}
            result["answer_confidence"] = 0.0
            result["reasoning_summary"] = []
            result["solution_steps"] = []

        return result

    def _fallback_parser_result(self, user_text: str, variants: Dict[str, Optional[str]]) -> Dict:
        summary = "Parser fallback used because Qwen was temporarily unavailable."
        if variants.get("full_image"):
            summary += " The image should still be checked directly by Kimi and Llama."
        return {
            "task_type": "geometry_photo",
            "ocr_text": user_text or "",
            "normalized_problem_text": user_text or "",
            "diagram_entities": [],
            "diagram_relations": [],
            "givens": self._normalize_givens([]),
            "target": {"statement": ""},
            "visual_interpretation": {
                "summary": summary,
                "confidence": 0.0,
                "possible_ambiguities": ["parser unavailable"],
            },
            "reasoning_summary": [],
            "solution_steps": [],
            "final_answer": {"value": "", "format": "text"},
            "answer_confidence": 0.0,
            "consistency_checks": [],
            "needs_clarification": True,
        }

    def _fallback_verifier_result(self, kimi_result: Dict) -> Dict:
        final_answer = kimi_result.get("final_answer") or {"value": "", "format": "text"}
        return {
            "task_type": kimi_result.get("task_type") or "geometry_photo",
            "ocr_text": kimi_result.get("ocr_text") or "",
            "normalized_problem_text": kimi_result.get("normalized_problem_text") or "",
            "diagram_entities": kimi_result.get("diagram_entities") or [],
            "diagram_relations": kimi_result.get("diagram_relations") or [],
            "givens": kimi_result.get("givens") or [],
            "target": kimi_result.get("target") or {"statement": ""},
            "visual_interpretation": {
                "summary": "Verifier fallback used because Llama was temporarily unavailable.",
                "confidence": 0.0,
                "possible_ambiguities": ["verifier unavailable"],
            },
            "reasoning_summary": [],
            "solution_steps": [],
            "final_answer": {
                "value": str(final_answer.get("value") or ""),
                "format": str(final_answer.get("format") or "text"),
            },
            "answer_confidence": 0.0,
            "consistency_checks": [],
            "needs_clarification": True,
        }

    def _normalize_entities(self, value) -> List[Dict]:
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            value = [value] if value else []

        normalized = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                text = self._normalize_text(item)
                if text:
                    normalized.append({"label": str(item), "type": "raw"})
        return normalized

    def _normalize_relations(self, value) -> List:
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            value = [value] if value else []

        normalized = []
        for item in value:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                text = str(item).strip()
                if text:
                    normalized.append(text)
        return normalized

    def _normalize_givens(self, value) -> List[Dict]:
        if isinstance(value, dict):
            normalized = []
            for key, item in value.items():
                statement = "%s: %s" % (key, item)
                normalized.append({"statement": statement})
            return normalized
        if not isinstance(value, list):
            value = [value] if value else []

        normalized = []
        for item in value:
            if isinstance(item, dict):
                if item.get("statement"):
                    normalized.append(item)
                else:
                    parts = []
                    for key, subvalue in item.items():
                        parts.append("%s: %s" % (key, subvalue))
                    statement = "; ".join(parts).strip()
                    if statement:
                        normalized.append({"statement": statement})
            else:
                text = str(item).strip()
                if text:
                    normalized.append({"statement": text})
        return normalized

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
        if not final_answer and llama is not None:
            final_answer = llama["final_answer"].get("value", "").strip()
        if not final_answer:
            final_answer = qwen["final_answer"].get("value", "").strip()
        if final_answer:
            return "1) %s" % final_answer

        return "1) Не удалось определить ответ"

    def _extract_json_object(self, raw_text: str) -> Dict:
        text = self._strip_json_wrappers(raw_text)
        if not text:
            raise ValueError("Empty JSON response")

        try:
            return json.loads(text)
        except ValueError:
            pass

        candidates = self._candidate_json_objects(text)
        last_error = None
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except ValueError as exc:
                last_error = exc
                repaired = self._repair_json(candidate)
                try:
                    return json.loads(repaired)
                except ValueError as repaired_exc:
                    last_error = repaired_exc
                    continue

        if candidates:
            raise ValueError("%s: %s" % (last_error.__class__.__name__ if last_error else "Invalid JSON", str(last_error or "unknown parse failure")))
        raise ValueError("No JSON object found in response: %s" % text[:200])

    def _candidate_json_objects(self, text: str) -> List[str]:
        candidates = []

        balanced = self._extract_balanced_object(text)
        if balanced:
            candidates.append(balanced)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            greedy = match.group(0).strip()
            if greedy and greedy not in candidates:
                candidates.append(greedy)

        first_brace = text.find("{")
        if first_brace != -1:
            suffix = text[first_brace:].strip()
            if suffix and suffix not in candidates:
                candidates.append(suffix)

        return candidates

    def _extract_balanced_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1].strip()

        return text[start:].strip()

    def _strip_json_wrappers(self, raw_text: str) -> str:
        text = (raw_text or "").strip()
        if not text:
            return ""
        text = text.replace("```json", "").replace("```", "").strip()
        return text

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
        repaired = self._strip_json_wrappers(repaired)
        if repaired != text:
            _log.warning("Applied local JSON repair before parsing model output")
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
            if isinstance(relation, dict):
                rel_type = relation.get("type", "")
                subject = relation.get("subject", "")
                obj = relation.get("object", "")
                key = "%s:%s:%s" % (rel_type, subject, obj)
            else:
                key = self._normalize_text(relation)
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
