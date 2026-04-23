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
    "You are the primary solver for mixed user tasks from text and images. "
    "Reason as fully as needed internally before deciding on the answer. "
    "Prioritize correct interpretation of the source over speed. "
    "Handle any domain, not only mathematics. "
    "If several independent tasks are present, solve all of them in source order. "
    "Do not let the requirement of concise final output reduce reasoning quality. "
    "If the task is solvable, final_answer.value must contain only the final answer content. "
    "For multiple answers, use a short numbered list like '1) ...\\n2) ...'. "
    + JSON_SCHEMA_NOTE
)

QWEN_SYSTEM_PROMPT = (
    "You are the parser and extractor for mixed user tasks from text and images. "
    "Extract OCR text, task boundaries, entities, relations, givens, targets, and ambiguities. "
    "Handle any domain, not only mathematics. "
    "If several independent tasks are present, preserve their order. "
    "Do not optimize for solving. Lower confidence instead of guessing. "
    "Leave final_answer empty unless the answer is explicitly printed in the source itself. "
    + JSON_SCHEMA_NOTE
)

LLAMA_SYSTEM_PROMPT = (
    "You are the independent verifier for mixed user tasks from text and images. "
    "Re-check interpretation, target, reasoning, and final answer without blindly copying the proposed result. "
    "Handle any domain, not only mathematics. "
    "If several independent tasks are present, verify all of them in source order. "
    "Do not let the requirement of concise final output reduce reasoning quality. "
    + JSON_SCHEMA_NOTE
)


class RecoverableProviderError(RuntimeError):
    """Provider failure that parser/verifier may degrade around."""


class GeometryPhotoSolver:
    def __init__(
        self,
        api_key: str,
        kimi_model: str = DEFAULT_KIMI_MODEL,
        qwen_model: str = DEFAULT_QWEN_MODEL,
        llama_model: str = DEFAULT_LLAMA_MODEL,
    ):
        self.api_key = api_key
        self.kimi_model = kimi_model
        self.qwen_model = qwen_model
        self.llama_model = llama_model
        _log.info(
            "GeometryPhotoSolver initialized: kimi=%s qwen=%s llama=%s",
            self.kimi_model,
            self.qwen_model,
            self.llama_model,
        )

    def solve_content_blocks(self, content_blocks: List[Dict]) -> str:
        image_urls, user_text = self._extract_image_payload(content_blocks)
        if not image_urls and not user_text.strip():
            return "1) Не удалось определить ответ"

        preprocessed = self._prepare_variants(image_urls)
        _log.info(
            "Prepared request variants: image_count=%d auxiliary_crops=%d",
            len(preprocessed),
            len([variant for variant in preprocessed if variant.get("text_crop") or variant.get("diagram_crop")]),
        )

        qwen_result = self._call_qwen(preprocessed, user_text)
        kimi_result = self._call_kimi(preprocessed, user_text, qwen_result, None)
        llama_result = self._call_llama(preprocessed, user_text, qwen_result, kimi_result, None)
        consensus = self._compare_results(kimi_result, qwen_result, llama_result)
        _log.info("Consensus after llama: status=%s score=%.3f", consensus["status"], consensus["score"])

        if self._should_run_self_check(consensus, kimi_result, llama_result):
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

    def _prepare_variants(self, image_urls: List[str]) -> List[Dict[str, Optional[str]]]:
        prepared = []
        for index, image_url in enumerate(image_urls, start=1):
            variants = {
                "full_image": image_url,
                "text_crop": None,
                "diagram_crop": None,
            }
            image_bytes = self._data_url_to_bytes(image_url)
            if image_bytes is None:
                _log.warning("Could not decode image data URL for image %d, using full image only", index)
                prepared.append(variants)
                continue

            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    image = img.convert("RGB")
                    width, height = image.size
                    _log.info("Preparing image variants: image=%d width=%d height=%d", index, width, height)
                    if height >= int(width * 1.15) and height >= 300:
                        split_y = int(height * 0.45)
                        text_crop = image.crop((0, 0, width, split_y))
                        diagram_crop = image.crop((0, split_y, width, height))
                        variants["text_crop"] = self._image_to_data_url(text_crop)
                        variants["diagram_crop"] = self._image_to_data_url(diagram_crop)
            except Exception as exc:
                _log.error("Image preprocessing failed for image %d: %s", index, exc)
            prepared.append(variants)
        return prepared

    def _call_qwen(self, variants: List[Dict[str, Optional[str]]], user_text: str) -> Dict:
        user_content = self._build_qwen_content(variants, user_text)
        try:
            result = self._request_json(self.qwen_model, QWEN_SYSTEM_PROMPT, user_content)
            return self._normalize_result(result, role="parser")
        except RecoverableProviderError as exc:
            _log.warning("Qwen parser unavailable, using degraded parser fallback: %s", exc)
            return self._fallback_parser_result(user_text, variants)

    def _call_kimi(
        self,
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        qwen_result: Dict,
        mismatch_summary: Optional[str],
    ) -> Dict:
        user_content = self._build_kimi_content(variants, user_text, qwen_result, mismatch_summary)
        try:
            result = self._request_json(self.kimi_model, KIMI_SYSTEM_PROMPT, user_content)
            return self._normalize_result(result, role="solver")
        except RecoverableProviderError as exc:
            _log.warning("Kimi primary solver unavailable, trying degraded solver fallback: %s", exc)
            if self.llama_model != self.kimi_model:
                try:
                    result = self._request_json(self.llama_model, KIMI_SYSTEM_PROMPT, user_content)
                    normalized = self._normalize_result(result, role="solver")
                    ambiguities = normalized["visual_interpretation"].get("possible_ambiguities") or []
                    ambiguities.append("primary solver unavailable")
                    ambiguities.append("solver used verifier model")
                    normalized["visual_interpretation"]["possible_ambiguities"] = ambiguities
                    normalized["needs_clarification"] = True
                    return normalized
                except RecoverableProviderError as degraded_exc:
                    _log.warning("Degraded solver fallback via verifier model failed: %s", degraded_exc)
            return self._fallback_solver_result(user_text, qwen_result)

    def _call_llama(
        self,
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        qwen_result: Dict,
        kimi_result: Dict,
        mismatch_summary: Optional[str],
    ) -> Dict:
        ambiguities = (kimi_result.get("visual_interpretation") or {}).get("possible_ambiguities") or []
        if "solver used verifier model" in ambiguities:
            _log.warning("Skipping verifier request because llama already served as degraded solver")
            return self._fallback_verifier_result(kimi_result)
        user_content = self._build_llama_content(variants, user_text, qwen_result, kimi_result, mismatch_summary)
        try:
            result = self._request_json(self.llama_model, LLAMA_SYSTEM_PROMPT, user_content)
            return self._normalize_result(result, role="verifier")
        except RecoverableProviderError as exc:
            _log.warning("Llama verifier unavailable, using degraded verifier fallback: %s", exc)
            return self._fallback_verifier_result(kimi_result)

    def _build_qwen_content(self, variants: List[Dict[str, Optional[str]]], user_text: str) -> List[Dict]:
        has_image = bool(variants)
        if has_image:
            text_prompt = (
                "Parse the attached task materials into strict JSON.\n"
                "Extract OCR text, task boundaries, entities, relations, givens, targets, ambiguities, and confidence.\n"
                "The images may contain one task or several independent tasks. Preserve their order.\n"
                "Do not solve unless needed for normalization.\n"
                "%s" % JSON_SCHEMA_NOTE
            )
        else:
            text_prompt = (
                "Parse the user text task into strict JSON.\n"
                "Extract task boundaries, givens, target, inferred entities, relations, ambiguities, and confidence.\n"
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
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        qwen_result: Dict,
        mismatch_summary: Optional[str],
    ) -> List[Dict]:
        has_image = bool(variants)
        if has_image:
            prompt = (
                "Solve the user task from the attached text and images and return strict JSON.\n"
                "Use the Qwen parse as a helper, but correct it if the source clearly disagrees.\n"
                "The materials may contain several independent tasks; solve all of them in source order.\n"
                "Prioritize correct interpretation over aggressive solving.\n"
                "Keep full reasoning inside solution_steps and reasoning_summary.\n"
                "%s\n" % JSON_SCHEMA_NOTE
            )
        else:
            prompt = (
                "Solve the user text task and return strict JSON.\n"
                "Use the Qwen parse as a helper, but reason independently.\n"
                "If several independent tasks are present, solve all of them in order.\n"
                "Prioritize faithful interpretation of the source.\n"
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
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        qwen_result: Dict,
        kimi_result: Dict,
        mismatch_summary: Optional[str],
    ) -> List[Dict]:
        has_image = bool(variants)
        if has_image:
            prompt = (
                "Verify the user task from the attached text and images and return strict JSON.\n"
                "Check the interpretation, targets, and final answer for all tasks in order.\n"
                "Do not blindly copy Kimi. If unsure, lower confidence or mark ambiguity.\n"
                "Keep full reasoning inside solution_steps and reasoning_summary.\n"
                "%s\n" % JSON_SCHEMA_NOTE
            )
        else:
            prompt = (
                "Verify the user text task and return strict JSON.\n"
                "Check the target, reasoning, and final answer.\n"
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

    def _build_image_blocks(self, variants: List[Dict[str, Optional[str]]]) -> List[Dict]:
        content = []
        for index, variant in enumerate(variants, start=1):
            if variant.get("full_image"):
                content.append({"type": "text", "text": "Source image %d." % index})
                content.append({"type": "image_url", "image_url": {"url": variant["full_image"]}})
            if variant.get("text_crop"):
                content.append({"type": "text", "text": "Auxiliary crop for image %d: likely text region." % index})
                content.append({"type": "image_url", "image_url": {"url": variant["text_crop"]}})
            if variant.get("diagram_crop"):
                content.append({"type": "text", "text": "Auxiliary crop for image %d: likely lower region or diagram." % index})
                content.append({"type": "image_url", "image_url": {"url": variant["diagram_crop"]}})
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
        _log.info("Requesting task JSON: model=%s blocks=%d", model, len(user_content))
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
                last_error = RecoverableProviderError("[Таймаут OpenRouter: модель %s не ответила вовремя]" % model)
                _log.error("Task solver timeout: model=%s attempt=%d elapsed_ms=%d timeout=%s error=%s", model, attempt, elapsed_ms, REQUEST_TIMEOUT, exc)
                if attempt < 2:
                    self._sleep_before_retry(attempt)
                    continue
                raise last_error
            except requests.RequestException as exc:
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                last_error = RecoverableProviderError("[Ошибка сети OpenRouter: %s]" % exc)
                _log.error("Task solver request failed: model=%s attempt=%d elapsed_ms=%d error=%s", model, attempt, elapsed_ms, exc)
                if attempt < 2:
                    self._sleep_before_retry(attempt)
                    continue
                raise last_error

            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            _log.info("Task solver HTTP response: model=%s attempt=%d status=%d elapsed_ms=%d", model, attempt, response.status_code, elapsed_ms)
            if response.ok:
                break

            body = response.text[:500]
            request_id = response.headers.get("x-request-id") or response.headers.get("cf-ray") or "-"
            _log.error("Task solver API error: model=%s attempt=%d status=%d request_id=%s body=%s", model, attempt, response.status_code, request_id, body)
            last_error = RecoverableProviderError("[Ошибка API %d: %s]" % (response.status_code, body))
            if self._is_retryable_status(response.status_code) and attempt < 2:
                self._sleep_before_retry(attempt)
                continue
            raise last_error

        if response is None:
            raise last_error or RecoverableProviderError("[Ошибка API: пустой ответ]")

        try:
            data = response.json()
        except ValueError as exc:
            raise RecoverableProviderError("[Ошибка API: невалидный JSON от %s]" % model) from exc

        message = (data.get("choices") or [{}])[0].get("message") or {}
        raw_text = self._message_to_text(message)
        try:
            parsed = self._extract_json_object(raw_text)
        except ValueError as exc:
            _log.warning("Primary JSON parse failed for model=%s chars=%d: %s", model, len(raw_text), exc)
            try:
                repaired_text = self._repair_non_json_response(model, raw_text)
                parsed = self._extract_json_object(repaired_text)
            except ValueError as repair_exc:
                raise RecoverableProviderError("[Ошибка JSON: модель %s вернула невалидный ответ]" % model) from repair_exc
            _log.info("JSON repair pass validated successfully: model=%s chars=%d", model, len(repaired_text))
        _log.info("Task JSON parsed: model=%s chars=%d", model, len(raw_text))
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
        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError("JSON repair returned invalid API payload") from exc
        message = (data.get("choices") or [{}])[0].get("message") or {}
        repaired_text = self._message_to_text(message)
        if not repaired_text.strip():
            raise ValueError("JSON repair returned empty content")
        return repaired_text

    def _normalize_result(self, raw: Dict, role: str = "generic") -> Dict:
        result = {
            "task_type": raw.get("task_type") or "mixed_task",
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

    def _fallback_parser_result(self, user_text: str, variants: List[Dict[str, Optional[str]]]) -> Dict:
        summary = "Parser fallback used because Qwen was temporarily unavailable."
        if variants:
            summary += " The attached images should still be checked directly by Kimi and Llama."
        return {
            "task_type": "mixed_task",
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
            "task_type": kimi_result.get("task_type") or "mixed_task",
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

    def _fallback_solver_result(self, user_text: str, qwen_result: Dict) -> Dict:
        final_answer = (qwen_result.get("final_answer") or {}).get("value", "")
        ambiguities = ["primary solver unavailable"]
        qwen_ambiguities = (qwen_result.get("visual_interpretation") or {}).get("possible_ambiguities") or []
        ambiguities.extend([item for item in qwen_ambiguities if item not in ambiguities])
        return {
            "task_type": qwen_result.get("task_type") or "mixed_task",
            "ocr_text": qwen_result.get("ocr_text") or user_text or "",
            "normalized_problem_text": qwen_result.get("normalized_problem_text") or user_text or "",
            "diagram_entities": qwen_result.get("diagram_entities") or [],
            "diagram_relations": qwen_result.get("diagram_relations") or [],
            "givens": qwen_result.get("givens") or [],
            "target": qwen_result.get("target") or {"statement": ""},
            "visual_interpretation": {
                "summary": "Solver fallback used because Kimi was temporarily unavailable.",
                "confidence": 0.0,
                "possible_ambiguities": ambiguities,
            },
            "reasoning_summary": [],
            "solution_steps": [],
            "final_answer": {
                "value": str(final_answer or ""),
                "format": "text",
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
        answer_agreement = 1.0 if self._normalize_answer_text(kimi["final_answer"].get("value", "")) == self._normalize_answer_text(llama["final_answer"].get("value", "")) else 0.0

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

    def _should_run_self_check(self, consensus: Dict, kimi: Dict, llama: Optional[Dict]) -> bool:
        if consensus["status"] == "accepted":
            return False
        if consensus["status"] == "self_check":
            return True
        kimi_answer = (kimi.get("final_answer") or {}).get("value", "").strip()
        llama_answer = ""
        if llama is not None:
            llama_answer = (llama.get("final_answer") or {}).get("value", "").strip()
        return bool(kimi_answer or llama_answer)

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
        final_answer = self._pick_user_answer(consensus, kimi, qwen, llama)
        if final_answer:
            _log.info(
                "Returning user answer: status=%s score=%.3f answer=%s",
                consensus["status"],
                consensus["score"],
                final_answer,
            )
            return self._render_answer_only(final_answer)

        _log.warning(
            "No safe final answer after consensus: status=%s score=%.3f reasons=%s",
            consensus["status"],
            consensus["score"],
            ", ".join(consensus.get("reasons") or []) or "-",
        )
        return "1) Не удалось определить ответ"

    def _pick_user_answer(self, consensus: Dict, kimi: Dict, qwen: Dict, llama: Optional[Dict]) -> str:
        kimi_answer = (kimi.get("final_answer") or {}).get("value", "").strip()
        llama_answer = ""
        llama_confidence = 0.0
        if llama is not None:
            llama_answer = (llama.get("final_answer") or {}).get("value", "").strip()
            llama_confidence = self._coerce_confidence(llama.get("answer_confidence"))
        kimi_confidence = self._coerce_confidence(kimi.get("answer_confidence"))

        if consensus["status"] == "accepted":
            return kimi_answer or llama_answer

        answers_match = bool(kimi_answer) and self._normalize_answer_text(kimi_answer) == self._normalize_answer_text(llama_answer)
        parser_unavailable = "parser unavailable" in (qwen.get("visual_interpretation") or {}).get("possible_ambiguities", [])
        verifier_unavailable = bool(
            llama and "verifier unavailable" in (llama.get("visual_interpretation") or {}).get("possible_ambiguities", [])
        )

        if answers_match:
            _log.info("Accepting answer despite non-accepted consensus because Kimi and Llama agree")
            return kimi_answer

        if kimi_answer and verifier_unavailable:
            _log.info("Accepting Kimi answer because verifier is unavailable")
            return kimi_answer

        if kimi_answer and not llama_answer:
            _log.info("Accepting Kimi answer because verifier did not provide a competing answer")
            return kimi_answer

        if llama_answer and not kimi_answer:
            _log.info("Accepting verifier answer because primary solver did not provide an answer")
            return llama_answer

        if kimi_answer and llama_answer:
            if kimi_confidence >= llama_confidence + 0.15:
                _log.info("Accepting Kimi answer on confidence tie-break: kimi=%.3f llama=%.3f", kimi_confidence, llama_confidence)
                return kimi_answer
            if llama_confidence >= kimi_confidence + 0.15:
                _log.info("Accepting verifier answer on confidence tie-break: kimi=%.3f llama=%.3f", kimi_confidence, llama_confidence)
                return llama_answer

        if kimi_answer and consensus["status"] == "self_check" and not parser_unavailable and not qwen.get("needs_clarification"):
            _log.info("Accepting Kimi answer after self-check because parser did not report unresolved ambiguity")
            return kimi_answer

        if kimi_answer and not parser_unavailable and not qwen.get("needs_clarification"):
            _log.info("Accepting Kimi answer despite ambiguous consensus because parser did not flag unresolved ambiguity")
            return kimi_answer

        return ""

    def _render_answer_only(self, final_answer: str) -> str:
        answer = (final_answer or "").strip()
        if not answer:
            return "1) Не удалось определить ответ"
        if re.match(r"^\d+\)\s*", answer):
            return answer
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if len(lines) > 1:
            return "\n".join(["%d) %s" % (index, line) for index, line in enumerate(lines, start=1)])
        return "1) %s" % answer

    def _message_to_text(self, message: Dict) -> str:
        content = message.get("content")
        text = self._coerce_message_content(content)
        if text:
            return text
        reasoning = message.get("reasoning")
        return self._coerce_message_content(reasoning)

    def _coerce_message_content(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            return "\n".join([part for part in parts if part])
        if isinstance(value, dict):
            text = value.get("text")
            if text:
                return str(text)
        return str(value)

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
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
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

    def _normalize_answer_text(self, text: str) -> str:
        if text is None:
            return ""
        lines = []
        for raw_line in str(text).splitlines():
            line = raw_line.strip().lower()
            if not line:
                continue
            line = re.sub(r"^\d+\)\s*", "", line)
            line = re.sub(r"^ответ[:\s-]*", "", line)
            line = " ".join(line.split())
            if line:
                lines.append(line)
        if lines:
            return "\n".join(lines)
        return self._normalize_text(text)

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
