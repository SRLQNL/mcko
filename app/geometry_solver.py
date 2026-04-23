from __future__ import annotations

import base64
import io
import json
import logging
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image
from requests.adapters import HTTPAdapter

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = (20, 90)
PARSER_MAX_TOKENS = 1200
SOLVER_MAX_TOKENS = 1400
VERIFIER_MAX_TOKENS = 1000
TEXT_ONLY_MAX_TOKENS = 900
REPAIR_MAX_TOKENS = 1000
ACCEPT_SCORE_THRESHOLD = 0.85
SELF_CHECK_SCORE_THRESHOLD = 0.65
DIRECT_SOLVER_CONFIDENCE_THRESHOLD = 0.85
TIE_BREAK_CONFIDENCE_GAP = 0.15
REPAIRED_MATCH_CONFIDENCE_THRESHOLD = 0.60

DEFAULT_KIMI_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_QWEN_MODEL = "qwen/qwen2.5-vl-72b-instruct"
DEFAULT_LLAMA_MODEL = "meta-llama/llama-4-maverick"
RETRYABLE_STATUSES = (408, 429, 502, 503, 504)

_log = logging.getLogger("mcko.geometry_solver")

PARSER_JSON_SCHEMA_NOTE = (
    'Return one valid JSON object with keys: '
    '"task_type","ocr_text","normalized_problem_text","diagram_entities","diagram_relations",'
    '"givens","target","visual_interpretation","reasoning_summary","solution_steps",'
    '"final_answer","answer_confidence","consistency_checks","needs_clarification". '
    'Use double quotes. No markdown. No prose outside JSON.'
)

SOLVER_JSON_SCHEMA_NOTE = (
    'Return one valid JSON object with keys: '
    '"task_type","normalized_problem_text","diagram_relations","givens","target",'
    '"visual_interpretation","final_answer","answer_confidence","consistency_checks","needs_clarification". '
    'Keep arrays short. Use double quotes. No markdown. No prose outside JSON.'
)

KIMI_SYSTEM_PROMPT = (
    "You are the primary solver for mixed user tasks from text and images. "
    "Reason as fully as needed internally before deciding on the answer. "
    "Prioritize correct interpretation of the source over speed. "
    "Handle any domain, not only mathematics. "
    "If several independent tasks are present, solve all of them in source order. "
    "Do not let the requirement of concise final output reduce reasoning quality. "
    "Do not emit full derivations or long reasoning text. "
    "If the task is solvable, final_answer.value must contain only the final answer content. "
    "For multiple answers, use a short numbered list like '1) ...\\n2) ...'. "
    + SOLVER_JSON_SCHEMA_NOTE
)

QWEN_SYSTEM_PROMPT = (
    "You are the parser and extractor for mixed user tasks from text and images. "
    "Extract OCR text, task boundaries, entities, relations, givens, targets, and ambiguities. "
    "Handle any domain, not only mathematics. "
    "If several independent tasks are present, preserve their order. "
    "Keep extracted summaries concise and avoid verbose restatement. "
    "Do not optimize for solving. Lower confidence instead of guessing. "
    "Leave final_answer empty unless the answer is explicitly printed in the source itself. "
    + PARSER_JSON_SCHEMA_NOTE
)

LLAMA_SYSTEM_PROMPT = (
    "You are the independent verifier for mixed user tasks from text and images. "
    "Re-check interpretation, target, reasoning, and final answer without blindly copying the proposed result. "
    "Handle any domain, not only mathematics. "
    "If several independent tasks are present, verify all of them in source order. "
    "Do not let the requirement of concise final output reduce reasoning quality. "
    "Do not emit full derivations or long reasoning text. "
    + SOLVER_JSON_SCHEMA_NOTE
)

KIMI_TEXT_ONLY_SYSTEM_PROMPT = (
    "You are the primary solver for user text tasks. "
    "Reason as fully as needed internally before deciding on the answer. "
    "Prioritize correctness over speed, but keep the output strictly structured. "
    "If several independent tasks are present, solve all of them in source order. "
    "Do not let concise final output reduce reasoning quality. "
    "Do not emit full derivations or long reasoning text. "
    "If the task is solvable, final_answer.value must contain only the final answer content. "
    "For multiple answers, use a short numbered list like '1) ...\\n2) ...'. "
    + SOLVER_JSON_SCHEMA_NOTE
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
        self._solve_lock = threading.Lock()
        self._http = self._build_http_session()
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

        started_at = time.monotonic()
        _log.info("Waiting for solver slot: has_images=%s", bool(image_urls))
        with self._solve_lock:
            _log.info("Solver slot acquired: has_images=%s", bool(image_urls))
            try:
                if not image_urls:
                    return self._solve_text_only(user_text)

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
            except RecoverableProviderError as exc:
                _log.warning("Recoverable provider failure at top-level solve path: %s", exc)
                return "1) Не удалось определить ответ"
            except Exception as exc:
                _log.error("Unexpected solver failure at top-level solve path: %s", exc, exc_info=True)
                return "1) Не удалось определить ответ"
            finally:
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                _log.info("Solve pipeline finished: has_images=%s elapsed_ms=%d", bool(image_urls), elapsed_ms)

    def _build_http_session(self) -> requests.Session:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _solve_text_only(self, user_text: str) -> str:
        _log.info("Using text-only fast path via Kimi")
        try:
            kimi_result = self._call_kimi_text_only(user_text)
        except RecoverableProviderError as exc:
            _log.warning("Text-only fast path failed: %s", exc)
            return "1) Не удалось определить ответ"
        final_answer = (kimi_result.get("final_answer") or {}).get("value", "").strip()
        if final_answer:
            _log.info("Returning text-only answer: %s", final_answer)
            return self._render_answer_only(final_answer)
        _log.warning("Text-only fast path produced no answer")
        return "1) Не удалось определить ответ"

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
            result = self._request_json(self.qwen_model, QWEN_SYSTEM_PROMPT, user_content, max_tokens=PARSER_MAX_TOKENS)
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
            result = self._request_json(self.kimi_model, KIMI_SYSTEM_PROMPT, user_content, max_tokens=SOLVER_MAX_TOKENS)
            return self._normalize_result(result, role="solver")
        except RecoverableProviderError as exc:
            _log.warning("Kimi primary solver unavailable, trying degraded solver fallback: %s", exc)
            if self.llama_model != self.kimi_model:
                try:
                    result = self._request_json(self.llama_model, KIMI_SYSTEM_PROMPT, user_content, max_tokens=SOLVER_MAX_TOKENS)
                    normalized = self._normalize_result(result, role="solver")
                    ambiguities = normalized["visual_interpretation"].get("possible_ambiguities") or []
                    ambiguities.append("primary solver unavailable")
                    ambiguities.append("solver used verifier model")
                    normalized["visual_interpretation"]["possible_ambiguities"] = ambiguities
                    normalized["needs_clarification"] = True
                    normalized["_solver_origin"] = "degraded_solver"
                    return normalized
                except RecoverableProviderError as degraded_exc:
                    _log.warning("Degraded solver fallback via verifier model failed: %s", degraded_exc)
            return self._fallback_solver_result(user_text, qwen_result)

    def _call_kimi_text_only(self, user_text: str) -> Dict:
        user_content = self._build_kimi_text_only_content(user_text)
        result = self._request_json(self.kimi_model, KIMI_TEXT_ONLY_SYSTEM_PROMPT, user_content, max_tokens=TEXT_ONLY_MAX_TOKENS)
        return self._normalize_result(result, role="solver")

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
            return self._fallback_verifier_result(
                kimi_result,
                reason="verifier skipped because solver already used verifier model",
                mirrors_solver=True,
            )
        user_content = self._build_llama_content(variants, user_text, qwen_result, kimi_result, mismatch_summary)
        try:
            result = self._request_json(self.llama_model, LLAMA_SYSTEM_PROMPT, user_content, max_tokens=VERIFIER_MAX_TOKENS)
            return self._normalize_result(result, role="verifier")
        except RecoverableProviderError as exc:
            _log.warning("Llama verifier unavailable, using degraded verifier fallback: %s", exc)
            return self._fallback_verifier_result(kimi_result, reason="verifier unavailable", mirrors_solver=False)

    def _build_qwen_content(self, variants: List[Dict[str, Optional[str]]], user_text: str) -> List[Dict]:
        has_image = bool(variants)
        if has_image:
            text_prompt = (
                "Parse the attached task materials into strict JSON.\n"
                "Extract OCR text, task boundaries, entities, relations, givens, targets, ambiguities, and confidence.\n"
                "The images may contain one task or several independent tasks. Preserve their order.\n"
                "Do not solve unless needed for normalization.\n"
                "%s" % PARSER_JSON_SCHEMA_NOTE
            )
        else:
            text_prompt = (
                "Parse the user text task into strict JSON.\n"
                "Extract task boundaries, givens, target, inferred entities, relations, ambiguities, and confidence.\n"
                "Do not optimize for solving.\n"
                "%s" % PARSER_JSON_SCHEMA_NOTE
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
                "Return a compact JSON object focused on target, constraints, confidence, and final answer.\n"
                "%s\n" % SOLVER_JSON_SCHEMA_NOTE
            )
        else:
            prompt = (
                "Solve the user text task and return strict JSON.\n"
                "Use the Qwen parse as a helper, but reason independently.\n"
                "If several independent tasks are present, solve all of them in order.\n"
                "Prioritize faithful interpretation of the source.\n"
                "Return a compact JSON object focused on target, constraints, confidence, and final answer.\n"
                "%s\n" % SOLVER_JSON_SCHEMA_NOTE
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
                "Return a compact JSON object focused on inconsistencies, confidence, and final answer.\n"
                "%s\n" % SOLVER_JSON_SCHEMA_NOTE
            )
        else:
            prompt = (
                "Verify the user text task and return strict JSON.\n"
                "Check the target, reasoning, and final answer.\n"
                "Do not blindly copy Kimi. If unsure, lower confidence or mark ambiguity.\n"
                "Return a compact JSON object focused on inconsistencies, confidence, and final answer.\n"
                "%s\n" % SOLVER_JSON_SCHEMA_NOTE
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

    def _build_kimi_text_only_content(self, user_text: str) -> List[Dict]:
        prompt = (
            "Solve the user text task and return strict JSON.\n"
            "If several independent tasks are present, solve all of them in order.\n"
            "Prioritize faithful interpretation of the text.\n"
            "Return a compact JSON object focused on target, constraints, confidence, and final answer.\n"
            "%s\n" % SOLVER_JSON_SCHEMA_NOTE
        )
        if user_text:
            prompt += "User task:\n%s\n" % user_text
        return [{"type": "text", "text": prompt}]

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

    def _request_json(self, model: str, system_prompt: str, user_content: List[Dict], max_tokens: int) -> Dict:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "provider": {"allow_fallbacks": True},
        }
        _log.info("Requesting task JSON: model=%s blocks=%d max_tokens=%d", model, len(user_content), max_tokens)
        response = None
        last_error = None
        for attempt in range(1, 3):
            started_at = time.monotonic()
            try:
                response = self._http.post(
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
        used_repair = False
        repair_model = ""
        try:
            parsed = self._extract_json_object(raw_text)
        except ValueError as exc:
            _log.warning("Primary JSON parse failed for model=%s chars=%d: %s", model, len(raw_text), exc)
            parsed = None
            repair_error = None
            salvaged = self._try_salvage_answer_only(raw_text)
            if salvaged is not None:
                parsed = salvaged
                used_repair = True
                repair_model = "local_answer_salvage"
                _log.info("Local answer salvage succeeded: source_model=%s", model)
            repair_models = self._repair_models_for_source(model)
            for candidate_model in repair_models:
                if parsed is not None:
                    break
                try:
                    repaired_text = self._repair_non_json_response(candidate_model, raw_text, source_model=model)
                    parsed = self._extract_json_object(repaired_text)
                    used_repair = True
                    repair_model = candidate_model
                    _log.info(
                        "JSON repair pass validated successfully: source_model=%s repair_model=%s chars=%d",
                        model,
                        candidate_model,
                        len(repaired_text),
                    )
                    break
                except ValueError as repair_exc:
                    repair_error = repair_exc
                    _log.warning(
                        "JSON repair pass failed: source_model=%s repair_model=%s error=%s",
                        model,
                        candidate_model,
                        repair_exc,
                    )
            if parsed is None:
                raise RecoverableProviderError("[Ошибка JSON: модель %s вернула невалидный ответ]" % model) from repair_error
        parsed["_request_meta"] = {
            "model": model,
            "used_repair": used_repair,
            "repair_model": repair_model,
            "raw_text_chars": len(raw_text),
        }
        _log.info(
            "Task JSON parsed: model=%s chars=%d used_repair=%s repair_model=%s",
            model,
            len(raw_text),
            used_repair,
            repair_model or "-",
        )
        return parsed

    def _repair_models_for_source(self, model: str) -> List[str]:
        if model == self.kimi_model:
            if self.llama_model == self.kimi_model:
                return [self.kimi_model]
            return [self.llama_model]
        repair_models = [model]
        if self.llama_model not in repair_models:
            repair_models.append(self.llama_model)
        return repair_models

    def _is_retryable_status(self, status_code: int) -> bool:
        return status_code in RETRYABLE_STATUSES

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = 1.0 if attempt <= 1 else 2.0
        time.sleep(delay)

    def _repair_non_json_response(self, model: str, raw_text: str, source_model: str) -> str:
        repair_prompt = (
            "Convert the following model output into one strict JSON object without losing meaning.\n"
            "%s\n"
            "Original output:\n%s" % (SOLVER_JSON_SCHEMA_NOTE, raw_text)
        )
        _log.info("Requesting JSON repair pass: source_model=%s repair_model=%s", source_model, model)
        response = self._http.post(
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
                "max_tokens": REPAIR_MAX_TOKENS,
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

    def _try_salvage_answer_only(self, raw_text: str) -> Optional[Dict]:
        text = (raw_text or "").strip()
        if not text:
            return None

        answer = ""
        numbered_block = self._extract_numbered_answer_block(text)
        if numbered_block:
            answer = numbered_block
        else:
            for pattern in (
                r"(?im)^\s*final[_\s-]*answer\s*[:=-]\s*(.+?)\s*$",
                r"(?im)^\s*answer\s*[:=-]\s*(.+?)\s*$",
                r"(?im)^\s*итог(?:овый)?\s*ответ\s*[:=-]\s*(.+?)\s*$",
                r"(?im)^\s*ответ\s*[:=-]\s*(.+?)\s*$",
            ):
                match = re.search(pattern, text)
                if match:
                    answer = match.group(1).strip()
                    break

        answer = self._normalize_salvaged_answer(answer)
        if not answer:
            return None

        confidence = self._extract_salvaged_confidence(text)
        needs_clarification = bool(re.search(r"(?i)\b(ambiguous|unclear|need clarification|insufficient data)\b", text))
        return {
            "task_type": "mixed_task",
            "normalized_problem_text": "",
            "diagram_relations": [],
            "givens": [],
            "target": {"statement": ""},
            "visual_interpretation": {
                "summary": "Recovered final answer from non-JSON model output.",
                "confidence": confidence,
                "possible_ambiguities": ["recovered from non-json output"],
            },
            "final_answer": {"value": answer, "format": "text"},
            "answer_confidence": confidence,
            "consistency_checks": [],
            "needs_clarification": needs_clarification,
        }

    def _extract_numbered_answer_block(self, text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]
        collected = []
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                if collected:
                    break
                continue
            if re.match(r"^\d+\)\s+\S", stripped):
                collected.append(stripped)
                continue
            if collected:
                break
        if not collected:
            return ""
        collected.reverse()
        return "\n".join(collected)

    def _normalize_salvaged_answer(self, answer: str) -> str:
        text = (answer or "").strip().strip("`").strip()
        if not text:
            return ""
        if len(text) > 200:
            return ""
        text = re.sub(r"(?i)^(final[_\s-]*answer|answer|итог(?:овый)?\s*ответ|ответ)\s*[:=-]\s*", "", text).strip()
        text = re.sub(r"\s+$", "", text)
        if not text:
            return ""
        if re.search(r"[{}\\[\\]]", text):
            return ""
        return text

    def _extract_salvaged_confidence(self, text: str) -> float:
        match = re.search(r"(?im)^\s*(?:answer_)?confidence\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)\s*$", text)
        if match:
            return self._coerce_confidence(match.group(1))
        if re.search(r"(?i)\bhigh confidence\b", text):
            return 0.85
        if re.search(r"(?i)\bmedium confidence\b", text):
            return 0.65
        if re.search(r"(?i)\blow confidence\b", text):
            return 0.35
        return 0.7

    def _normalize_result(self, raw: Dict, role: str = "generic") -> Dict:
        request_meta = raw.get("_request_meta") if isinstance(raw.get("_request_meta"), dict) else {}
        visual_value = raw.get("visual_interpretation")
        if not visual_value:
            visual_value = {
                "summary": raw.get("visual_summary") or "",
                "confidence": self._coerce_confidence(raw.get("visual_interpretation_confidence")),
                "possible_ambiguities": raw.get("possible_ambiguities") or [],
            }
        result = {
            "task_type": raw.get("task_type") or "mixed_task",
            "ocr_text": raw.get("ocr_text") or "",
            "normalized_problem_text": raw.get("normalized_problem_text") or "",
            "diagram_entities": self._normalize_entities(raw.get("diagram_entities") or raw.get("objects") or []),
            "diagram_relations": self._normalize_relations(raw.get("diagram_relations") or raw.get("relations") or []),
            "givens": self._normalize_givens(raw.get("givens") or []),
            "target": self._normalize_target(raw.get("target")),
            "visual_interpretation": visual_value,
            "reasoning_summary": self._normalize_text_list(raw.get("reasoning_summary") or []),
            "solution_steps": self._normalize_text_list(raw.get("solution_steps") or []),
            "final_answer": raw.get("final_answer") or {"value": "", "format": "text"},
            "answer_confidence": self._coerce_confidence(raw.get("answer_confidence")),
            "consistency_checks": self._normalize_text_list(raw.get("consistency_checks") or []),
            "needs_clarification": bool(raw.get("needs_clarification", False)),
            "_request_meta": {
                "model": str(request_meta.get("model") or ""),
                "used_repair": bool(request_meta.get("used_repair", False)),
                "repair_model": str(request_meta.get("repair_model") or ""),
                "raw_text_chars": int(request_meta.get("raw_text_chars") or 0),
            },
            "_solver_origin": str(raw.get("_solver_origin") or "model"),
            "_mirrors_solver": bool(raw.get("_mirrors_solver", False)),
        }
        if isinstance(result["final_answer"], str):
            result["final_answer"] = {"value": result["final_answer"], "format": "text"}
        visual = result["visual_interpretation"]
        if isinstance(visual, str):
            result["visual_interpretation"] = {
                "summary": visual,
                "confidence": 0.5,
                "possible_ambiguities": [],
            }
        elif isinstance(visual, list):
            result["visual_interpretation"] = {
                "summary": "; ".join([str(item).strip() for item in visual if str(item).strip()])[:500],
                "confidence": self._coerce_confidence(raw.get("visual_interpretation_confidence")),
                "possible_ambiguities": [],
            }
        elif not isinstance(visual, dict):
            result["visual_interpretation"] = {
                "summary": str(visual).strip(),
                "confidence": self._coerce_confidence(raw.get("visual_interpretation_confidence")),
                "possible_ambiguities": [],
            }
        else:
            visual["summary"] = visual.get("summary") or ""
            visual["confidence"] = self._coerce_confidence(visual.get("confidence"))
            visual["possible_ambiguities"] = self._normalize_text_list(visual.get("possible_ambiguities") or [])
        final_answer = result["final_answer"]
        if isinstance(final_answer, list):
            final_answer = {"value": "\n".join([str(item).strip() for item in final_answer if str(item).strip()]), "format": "text"}
            result["final_answer"] = final_answer
        elif not isinstance(final_answer, dict):
            final_answer = {"value": str(final_answer), "format": "text"}
            result["final_answer"] = final_answer
        final_answer["value"] = "" if final_answer.get("value") is None else str(final_answer.get("value"))
        final_answer["format"] = str(final_answer.get("format") or "text")

        if role == "parser":
            result["final_answer"] = {"value": "", "format": "text"}
            result["answer_confidence"] = 0.0
            result["reasoning_summary"] = []
            result["solution_steps"] = []

        return result

    def _normalize_target(self, value) -> Dict[str, str]:
        if isinstance(value, dict):
            statement = value.get("statement")
            if statement:
                return {"statement": str(statement)}
            parts = []
            for key, item in value.items():
                part = "%s: %s" % (key, item)
                if part.strip():
                    parts.append(part)
            return {"statement": "; ".join(parts)}
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    statement = item.get("statement")
                    if statement:
                        parts.append(str(statement))
                    else:
                        flattened = []
                        for key, subvalue in item.items():
                            flattened.append("%s: %s" % (key, subvalue))
                        if flattened:
                            parts.append("; ".join(flattened))
                else:
                    text = str(item).strip()
                    if text:
                        parts.append(text)
            return {"statement": " | ".join(parts)}
        if value is None:
            return {"statement": ""}
        return {"statement": str(value)}

    def _normalize_text_list(self, value) -> List[str]:
        if isinstance(value, list):
            items = value
        elif value:
            items = [value]
        else:
            items = []
        normalized = []
        for item in items:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized

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
            "_request_meta": {"model": self.qwen_model, "used_repair": False, "repair_model": "", "raw_text_chars": 0},
            "_solver_origin": "parser_fallback",
            "_mirrors_solver": False,
        }

    def _fallback_verifier_result(self, kimi_result: Dict, reason: str, mirrors_solver: bool) -> Dict:
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
                "summary": "Verifier fallback used because %s." % reason,
                "confidence": 0.0,
                "possible_ambiguities": [reason],
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
            "_request_meta": {"model": self.llama_model, "used_repair": False, "repair_model": "", "raw_text_chars": 0},
            "_solver_origin": "verifier_fallback",
            "_mirrors_solver": mirrors_solver,
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
            "_request_meta": {"model": self.kimi_model, "used_repair": False, "repair_model": "", "raw_text_chars": 0},
            "_solver_origin": "solver_fallback",
            "_mirrors_solver": False,
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
        qwen_givens = self._jaccard(self._given_keys(kimi), self._given_keys(qwen))
        target_agreement = 1.0 if self._normalize_text(kimi["target"].get("statement", "")) == self._normalize_text(qwen["target"].get("statement", "")) else 0.0
        verifier_independent = self._has_independent_verifier(llama)
        if verifier_independent:
            llama_diagram = self._jaccard(self._relation_keys(kimi), self._relation_keys(llama))
            llama_givens = self._jaccard(self._given_keys(kimi), self._given_keys(llama))
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
        else:
            answer_agreement = 0.0
            diagram_agreement = qwen_diagram
            givens_agreement = qwen_givens
            confidence_alignment = (
                kimi["visual_interpretation"]["confidence"] +
                qwen["visual_interpretation"]["confidence"]
            ) / 2.0
            score = (
                0.45 * diagram_agreement +
                0.25 * givens_agreement +
                0.20 * target_agreement +
                0.10 * confidence_alignment
            )
        reasons = []
        if diagram_agreement < 0.5:
            reasons.append("low diagram agreement")
        if givens_agreement < 0.6:
            reasons.append("low givens agreement")
        if verifier_independent and answer_agreement < 1.0:
            reasons.append("answer mismatch")
        if not verifier_independent:
            reasons.append("no independent verifier")
        if qwen["visual_interpretation"]["possible_ambiguities"]:
            reasons.append("diagram ambiguity detected")
        if qwen.get("needs_clarification"):
            reasons.append("parser requested clarification")
        if self._used_repair(kimi):
            reasons.append("solver JSON repaired")
            score -= 0.08
        if self._solver_is_degraded(kimi):
            reasons.append("solver fallback path")
            score -= 0.20

        score = max(0.0, min(1.0, score))

        status = "ambiguous"
        accepted = False
        if (
            verifier_independent and
            score >= ACCEPT_SCORE_THRESHOLD and
            diagram_agreement >= 0.5 and
            not qwen.get("needs_clarification") and
            not self._solver_is_degraded(kimi) and
            not self._used_repair(kimi)
        ):
            status = "accepted"
            accepted = True
        elif score >= SELF_CHECK_SCORE_THRESHOLD or self._used_repair(kimi) or self._solver_is_degraded(kimi):
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
        if not self._has_independent_verifier(llama):
            return False
        if self._can_accept_repaired_match_without_self_check(consensus, kimi, llama):
            return False
        if consensus["status"] == "accepted" and not self._requires_quality_escalation(consensus, kimi, llama):
            return False
        if consensus["status"] == "accepted":
            return True
        if self._answers_effectively_match(kimi, llama):
            return self._requires_quality_escalation(consensus, kimi, llama)
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

        answers_match = self._answers_effectively_match(kimi, llama)
        parser_clear = not qwen.get("needs_clarification") and not (
            (qwen.get("visual_interpretation") or {}).get("possible_ambiguities") or []
        )
        parser_unavailable = "parser unavailable" in (qwen.get("visual_interpretation") or {}).get("possible_ambiguities", [])
        verifier_independent = self._has_independent_verifier(llama)
        solver_degraded = self._solver_is_degraded(kimi)
        solver_repaired = self._used_repair(kimi)

        if not verifier_independent:
            if solver_degraded:
                _log.warning("Rejecting answer because only degraded solver output is available without an independent verifier")
                return ""
            if kimi_answer and parser_clear and not solver_repaired and kimi_confidence >= DIRECT_SOLVER_CONFIDENCE_THRESHOLD:
                _log.info("Accepting direct Kimi answer without verifier because parser is clear and solver confidence is high")
                return kimi_answer
            return ""

        if self._can_accept_repaired_match_without_self_check(consensus, kimi, llama):
            _log.info("Accepting repaired solver answer because independent verifier matches and parser is clear")
            return kimi_answer or llama_answer

        if answers_match and parser_clear and not solver_degraded and not solver_repaired and consensus["score"] >= SELF_CHECK_SCORE_THRESHOLD:
            _log.info("Accepting answer after non-accepted consensus because independent solver and verifier match under clear parse")
            return kimi_answer

        if kimi_answer and not llama_answer:
            if parser_clear and not solver_degraded and not solver_repaired and kimi_confidence >= DIRECT_SOLVER_CONFIDENCE_THRESHOLD:
                _log.info("Accepting Kimi answer because verifier did not provide a competing answer and parser is clear")
                return kimi_answer
            return ""

        if llama_answer and not kimi_answer:
            if parser_clear and consensus["score"] >= ACCEPT_SCORE_THRESHOLD:
                _log.info("Accepting verifier answer because primary solver did not provide an answer")
                return llama_answer
            return ""

        if kimi_answer and llama_answer and parser_clear and not solver_degraded and not solver_repaired:
            if kimi_confidence >= llama_confidence + TIE_BREAK_CONFIDENCE_GAP and consensus["score"] >= SELF_CHECK_SCORE_THRESHOLD:
                _log.info("Accepting Kimi answer on confidence tie-break: kimi=%.3f llama=%.3f", kimi_confidence, llama_confidence)
                return kimi_answer
            if llama_confidence >= kimi_confidence + TIE_BREAK_CONFIDENCE_GAP and consensus["score"] >= SELF_CHECK_SCORE_THRESHOLD:
                _log.info("Accepting verifier answer on confidence tie-break: kimi=%.3f llama=%.3f", kimi_confidence, llama_confidence)
                return llama_answer

        if kimi_answer and consensus["status"] == "self_check" and parser_clear and not parser_unavailable and not solver_degraded and not solver_repaired and kimi_confidence >= DIRECT_SOLVER_CONFIDENCE_THRESHOLD:
            _log.info("Accepting Kimi answer after self-check because parser is clear and solver output is direct")
            return kimi_answer

        return ""

    def _used_repair(self, result: Optional[Dict]) -> bool:
        if not result:
            return False
        request_meta = result.get("_request_meta") or {}
        return bool(request_meta.get("used_repair", False))

    def _solver_is_degraded(self, result: Optional[Dict]) -> bool:
        if not result:
            return False
        return result.get("_solver_origin") in ("degraded_solver", "solver_fallback")

    def _has_independent_verifier(self, verifier: Optional[Dict]) -> bool:
        if not verifier:
            return False
        if verifier.get("_mirrors_solver"):
            return False
        ambiguities = (verifier.get("visual_interpretation") or {}).get("possible_ambiguities") or []
        return "verifier unavailable" not in ambiguities

    def _requires_quality_escalation(self, consensus: Dict, kimi: Dict, llama: Optional[Dict]) -> bool:
        return bool(
            consensus["score"] < 0.75 or
            self._used_repair(kimi) or
            self._solver_is_degraded(kimi) or
            self._used_repair(llama)
        )

    def _can_accept_repaired_match_without_self_check(self, consensus: Dict, kimi: Dict, llama: Optional[Dict]) -> bool:
        if not self._used_repair(kimi):
            return False
        if self._solver_is_degraded(kimi):
            return False
        if not self._has_independent_verifier(llama):
            return False
        if not self._answers_effectively_match(kimi, llama):
            return False
        qwen_like_clear = True
        kimi_visual = kimi.get("visual_interpretation") or {}
        ambiguities = kimi_visual.get("possible_ambiguities") or []
        if "recovered from non-json output" in ambiguities:
            qwen_like_clear = False
        kimi_confidence = self._coerce_confidence(kimi.get("answer_confidence"))
        llama_confidence = self._coerce_confidence((llama or {}).get("answer_confidence"))
        if kimi_confidence < REPAIRED_MATCH_CONFIDENCE_THRESHOLD and llama_confidence < REPAIRED_MATCH_CONFIDENCE_THRESHOLD:
            return False
        if consensus.get("answer_agreement", 0.0) < 1.0:
            return False
        return qwen_like_clear

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

    def _answers_effectively_match(self, kimi: Dict, llama: Optional[Dict]) -> bool:
        kimi_answer = (kimi.get("final_answer") or {}).get("value", "").strip()
        if not kimi_answer or llama is None:
            return False
        llama_answer = (llama.get("final_answer") or {}).get("value", "").strip()
        if not llama_answer:
            return False
        return self._normalize_answer_text(kimi_answer) == self._normalize_answer_text(llama_answer)

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
