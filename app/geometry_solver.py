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
REQUEST_TIMEOUT = (20, 120)
PARSER_MAX_TOKENS = 2500
SOLVER_MAX_TOKENS = 1500
VERIFIER_MAX_TOKENS = 1000
TEXT_ONLY_MAX_TOKENS = 1200
REPAIR_MAX_TOKENS = 2000
ACCEPT_SCORE_THRESHOLD = 0.85
SELF_CHECK_SCORE_THRESHOLD = 0.65
DIRECT_SOLVER_CONFIDENCE_THRESHOLD = 0.85
VERIFIER_ONLY_CONFIDENCE_THRESHOLD = 0.75
QWEN_CHALLENGER_CONFIDENCE_THRESHOLD = 0.85
TIE_BREAK_CONFIDENCE_GAP = 0.15
REPAIRED_MATCH_CONFIDENCE_THRESHOLD = 0.60

DEFAULT_SOLVER_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_PARSER_MODEL = "qwen/qwen3-vl-32b-instruct"
DEFAULT_VERIFIER_MODEL = "meta-llama/llama-4-maverick"
RETRYABLE_STATUSES = (408, 429, 502, 503, 504)

_log = logging.getLogger("mcko.geometry_solver")

PARSER_JSON_SCHEMA_NOTE = (
    'Return one valid JSON object with keys: '
    '"task_type","ocr_text","normalized_problem_text","diagram_relations",'
    '"givens","target","visual_interpretation",'
    '"final_answer","answer_confidence","consistency_checks","needs_clarification". '
    'Use double quotes. No markdown. No prose outside JSON.'
)

SOLVER_JSON_SCHEMA_NOTE = (
    'Return one valid JSON object with keys: '
    '"task_type","normalized_problem_text","diagram_relations","givens","target",'
    '"visual_interpretation","final_answer","answer_confidence","consistency_checks","needs_clarification". '
    'Keep arrays short. Use double quotes. No markdown. No prose outside JSON.'
)

SOLVER_SYSTEM_PROMPT = (
    "You are a universal solver for any school or exam task, any subject, any language. "
    "Subjects include but are not limited to: mathematics, physics, chemistry, biology, "
    "history, geography, Russian language, literature, informatics, social science, "
    "foreign languages, and any other discipline. "
    "The task text and/or images may be in Russian, English, or any other language. "
    "Reason fully before deciding. Prioritize correct interpretation over speed. "
    "If several independent tasks are present, solve all in source order. "
    "Output only the final answer — no derivations, no reasoning text, no explanations. "
    "For matching tasks (установите соответствие): answer as digit sequence like '2341'. "
    "For multiple-choice tasks (запишите номера / укажите цифры): only the correct digits like '135'. "
    "For calculation/formula tasks: give numeric result or formula. "
    "For word-fill tasks: the exact word or short phrase. "
    "If the task is solvable, final_answer.value must contain only the final answer. "
    "For multiple independent answers, use: '1) ...\\n2) ...'. "
    + SOLVER_JSON_SCHEMA_NOTE
)

PARSER_SYSTEM_PROMPT = (
    "You are the parser and OCR extractor for any task from text and images, any subject, any language. "
    "Extract OCR text carefully — pay attention to Cyrillic, Latin, math formulas, tables, diagrams, "
    "chemical notation, musical notation, code snippets, and any other domain-specific content. "
    "Extract task boundaries, givens, targets, entities, relations, and ambiguities. "
    "Preserve the source order of multiple tasks. "
    "Do not optimize for solving. Lower confidence when unsure; do not guess. "
    "Leave final_answer empty unless explicitly printed in the source. "
    + PARSER_JSON_SCHEMA_NOTE
)

VERIFIER_SYSTEM_PROMPT = (
    "You are the independent verifier for any school or exam task, any subject, any language. "
    "Re-check interpretation, targets, and final answer without blindly copying the proposed result. "
    "If several independent tasks are present, verify all in source order. "
    "Output only the final answer — no derivations, no reasoning text. "
    + SOLVER_JSON_SCHEMA_NOTE
)

SOLVER_TEXT_ONLY_SYSTEM_PROMPT = (
    "You are a universal solver for any text task, any subject, any language. "
    "Reason fully before deciding. Prioritize correctness over speed. "
    "If several independent tasks are present, solve all in source order. "
    "Output only the final answer — no derivations, no reasoning text. "
    "For matching tasks: digit sequence like '2341'. "
    "For multiple-choice tasks: only the correct digits like '135'. "
    "If the task is solvable, final_answer.value must contain only the final answer. "
    "For multiple answers, use: '1) ...\\n2) ...'. "
    + SOLVER_JSON_SCHEMA_NOTE
)


class RecoverableProviderError(RuntimeError):
    """Provider failure that parser/verifier may degrade around."""


class GeometryPhotoSolver:
    def __init__(
        self,
        api_key: str,
        solver_model: str = DEFAULT_SOLVER_MODEL,
        parser_model: str = DEFAULT_PARSER_MODEL,
        verifier_model: str = DEFAULT_VERIFIER_MODEL,
    ):
        self.api_key = api_key
        self.solver_model = solver_model
        self.parser_model = parser_model
        self.verifier_model = verifier_model
        self._solve_lock = threading.Lock()
        self._http = self._build_http_session()
        _log.info(
            "GeometryPhotoSolver initialized: solver=%s parser=%s verifier=%s",
            self.solver_model,
            self.parser_model,
            self.verifier_model,
        )

    def solve_content_blocks(self, content_blocks: List[Dict], multi_model: bool = True) -> str:
        image_urls, user_text = self._extract_image_payload(content_blocks)
        if not image_urls and not user_text.strip():
            return "1) Не удалось определить ответ"

        started_at = time.monotonic()
        _log.info("Waiting for solver slot: has_images=%s multi_model=%s", bool(image_urls), multi_model)
        with self._solve_lock:
            _log.info("Solver slot acquired")
            try:
                if not multi_model:
                    return self._solve_single_model(image_urls, user_text)

                if not image_urls:
                    return self._solve_text_only(user_text)

                preprocessed = self._prepare_variants(image_urls)
                _log.info(
                    "Prepared request variants: image_count=%d auxiliary_crops=%d",
                    len(preprocessed),
                    len([v for v in preprocessed if v.get("text_crop") or v.get("diagram_crop")]),
                )

                # Step 1: parser OCR (sequential — provides context for both solvers)
                parser_result = self._call_parser(preprocessed, user_text)

                # Step 2: solver + verifier in parallel, verifier independent (no anchoring)
                solver_result, verifier_result = self._call_parallel_solvers(preprocessed, user_text, parser_result)

                consensus = self._compare_results(solver_result, parser_result, verifier_result)
                _log.info("Consensus after parallel solve: status=%s score=%.3f", consensus["status"], consensus["score"])

                option_arbiter_result = None
                if self._should_run_option_arbiter(consensus, parser_result, solver_result, verifier_result):
                    option_arbiter_result = self._call_option_arbiter(preprocessed, user_text, parser_result, None)

                if self._should_run_self_check(consensus, solver_result, verifier_result):
                    first_round = {
                        "consensus": consensus,
                        "solver": solver_result,
                        "verifier": verifier_result,
                        "parser": parser_result,
                    }
                    mismatch_summary = self._build_mismatch_summary(consensus, solver_result, parser_result, verifier_result)
                    _log.info("Running parallel self-check round: %s", mismatch_summary)
                    parser_check = self._call_parser(preprocessed, user_text, mismatch_summary)
                    solver_check, verifier_check = self._call_parallel_solvers(preprocessed, user_text, parser_check, mismatch_summary)
                    second_round = {
                        "consensus": self._compare_results(solver_check, parser_check, verifier_check),
                        "solver": solver_check,
                        "verifier": verifier_check,
                        "parser": parser_check,
                    }
                    chosen_answer = self._resolve_multi_round_answer(parser_result, first_round, second_round, option_arbiter_result)
                    if chosen_answer:
                        _log.info("Returning answer from parallel multi-round: %s", chosen_answer)
                        return self._render_answer_only(chosen_answer)
                    if not self._pick_user_answer(first_round["consensus"], first_round["solver"], first_round["parser"], first_round["verifier"], option_arbiter_result):
                        consensus = second_round["consensus"]
                        solver_result = second_round["solver"]
                        verifier_result = second_round["verifier"]
                        parser_result = second_round["parser"]
                    else:
                        consensus = first_round["consensus"]
                        solver_result = first_round["solver"]
                        verifier_result = first_round["verifier"]
                        parser_result = first_round["parser"]
                    _log.info("Consensus after self-check: status=%s score=%.3f", consensus["status"], consensus["score"])

                return self._format_user_result(consensus, solver_result, parser_result, verifier_result, option_arbiter_result)
            except RecoverableProviderError as exc:
                _log.warning("Recoverable provider failure at top-level solve path: %s", exc)
                return "1) Не удалось определить ответ"
            except Exception as exc:
                _log.error("Unexpected solver failure at top-level solve path: %s", exc, exc_info=True)
                return "1) Не удалось определить ответ"
            finally:
                elapsed_ms = int((time.monotonic() - started_at) * 1000)
                _log.info("Solve pipeline finished: multi_model=%s elapsed_ms=%d", multi_model, elapsed_ms)

    def _build_http_session(self) -> requests.Session:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _solve_single_model(self, image_urls: List[str], user_text: str) -> str:
        """Fast single-model path. Parser for images (vision), solver for text."""
        if image_urls:
            _log.info("Single-model mode: Parser direct solve (vision+reasoning)")
            preprocessed = self._prepare_variants(image_urls)
            content = self._build_single_model_content(preprocessed, user_text)
            try:
                raw = self._request_json(self.parser_model, SOLVER_SYSTEM_PROMPT, content, max_tokens=SOLVER_MAX_TOKENS)
                result = self._normalize_result(raw, role="solver")
            except RecoverableProviderError as exc:
                _log.warning("Single-model image solve failed: %s", exc)
                return "1) Не удалось определить ответ"
        else:
            _log.info("Single-model mode: solver direct solve (text)")
            try:
                result = self._call_solver_text_only(user_text)
            except RecoverableProviderError as exc:
                _log.warning("Single-model text solve failed: %s", exc)
                return "1) Не удалось определить ответ"

        answer = (result.get("final_answer") or {}).get("value", "").strip()
        if answer and self._looks_like_final_answer(answer):
            _log.info("Single-model answer: %s", answer)
            return self._render_answer_only(answer)
        _log.warning("Single-model produced no valid answer")
        return "1) Не удалось определить ответ"

    def _build_single_model_content(self, variants: List[Dict], user_text: str) -> List[Dict]:
        prompt = (
            "Solve the user task from the images and return strict JSON.\n"
            "Read all text from the images, identify the problem, solve it.\n"
            "Return only the final answer in final_answer.value.\n"
            "%s\n" % SOLVER_JSON_SCHEMA_NOTE
        )
        if user_text:
            prompt += "User note:\n%s\n" % user_text
        content = [{"type": "text", "text": prompt}]
        content.extend(self._build_image_blocks(variants))
        return content

    # Confidence threshold at which we skip the verifier (EMS: Early Majority Stopping).
    # Only applied for non-option-selection tasks to avoid missing multi-digit answer disagreements.
    EMS_SKIP_VERIFIER_THRESHOLD = 0.95

    def _call_parallel_solvers(
        self,
        variants: List[Dict],
        user_text: str,
        parser_result: Dict,
        mismatch_summary: Optional[str] = None,
    ) -> Tuple[Dict, Dict]:
        """Run solver and verifier in parallel threads. Verifier solves independently (no solver anchoring)."""
        solver_holder = [None]
        verifier_holder = [None]
        verifier_skip_event = threading.Event()

        def run_solver():
            try:
                solver_holder[0] = self._call_solver(variants, user_text, parser_result, mismatch_summary)
            except RecoverableProviderError as exc:
                _log.warning("Parallel solver failed: %s", exc)
            finally:
                verifier_skip_event.set()

        def run_verifier():
            verifier_skip_event.wait()
            if self._ems_should_skip_verifier(solver_holder[0], parser_result):
                _log.info("EMS: skipping verifier — solver confidence >= %.2f on non-option task", self.EMS_SKIP_VERIFIER_THRESHOLD)
                return
            try:
                content = self._build_verifier_independent_content(variants, user_text, parser_result, mismatch_summary)
                raw = self._request_json(
                    self.verifier_model, VERIFIER_SYSTEM_PROMPT, content, max_tokens=VERIFIER_MAX_TOKENS
                )
                verifier_holder[0] = self._normalize_result(raw, role="verifier")
            except RecoverableProviderError as exc:
                _log.warning("Parallel verifier failed: %s", exc)

        t_solver = threading.Thread(target=run_solver, name="parallel-solver", daemon=True)
        t_verifier = threading.Thread(target=run_verifier, name="parallel-verifier", daemon=True)
        t_solver.start()
        t_verifier.start()
        t_solver.join()
        t_verifier.join()

        solver_result = solver_holder[0]
        verifier_result = verifier_holder[0]

        if solver_result is None:
            solver_result = self._fallback_solver_result(user_text, parser_result)
        if verifier_result is None:
            verifier_result = self._fallback_verifier_result(
                solver_result, reason="parallel verifier unavailable", mirrors_solver=False
            )

        _log.info(
            "Parallel solvers done: solver_answer=%s verifier_answer=%s",
            (solver_result.get("final_answer") or {}).get("value", ""),
            (verifier_result.get("final_answer") or {}).get("value", ""),
        )
        return solver_result, verifier_result

    def _ems_should_skip_verifier(self, solver_result: Optional[Dict], parser_result: Dict) -> bool:
        if solver_result is None:
            return False
        if self._is_option_selection_task(parser_result):
            return False
        confidence = self._coerce_confidence((solver_result.get("final_answer") or {}).get("confidence") or solver_result.get("answer_confidence"))
        answer = (solver_result.get("final_answer") or {}).get("value", "").strip()
        if not answer or not self._looks_like_final_answer(answer):
            return False
        return confidence >= self.EMS_SKIP_VERIFIER_THRESHOLD

    def _build_verifier_independent_content(
        self,
        variants: List[Dict],
        user_text: str,
        parser_result: Dict,
        mismatch_summary: Optional[str] = None,
    ) -> List[Dict]:
        """Build verifier prompt WITHOUT solver's answer — true independent solve."""
        has_image = bool(variants)
        if has_image:
            prompt = (
                "Solve the user task from the attached images and return strict JSON.\n"
                "You are solving INDEPENDENTLY — reason from source images directly.\n"
                "Do not try to confirm or deny any other model's answer.\n"
                "Use the parser OCR only as a text extraction aid.\n"
                "Return a compact JSON with your own final answer and confidence.\n"
                "%s\n" % SOLVER_JSON_SCHEMA_NOTE
            )
        else:
            prompt = (
                "Solve the user text task and return strict JSON.\n"
                "Reason independently. Return your own final answer.\n"
                "%s\n" % SOLVER_JSON_SCHEMA_NOTE
            )
        if user_text:
            prompt += "User hint:\n%s\n" % user_text
        prompt += "Parser OCR extract:\n%s\n" % json.dumps(
            self._compact_result_for_prompt(parser_result, role="parser"),
            ensure_ascii=False,
        )
        if mismatch_summary:
            prompt += (
                "Self-check note (treat as hint only, not ground truth):\n%s\n" % mismatch_summary
            )
        content = [{"type": "text", "text": prompt}]
        content.extend(self._build_image_blocks(variants))
        return content

    def _solve_text_only(self, user_text: str) -> str:
        _log.info("Using text-only fast path via solver")
        try:
            solver_result = self._call_solver_text_only(user_text)
        except RecoverableProviderError as exc:
            _log.warning("Text-only fast path failed: %s", exc)
            return "1) Не удалось определить ответ"

        solver_answer = (solver_result.get("final_answer") or {}).get("value", "").strip()
        solver_confidence = self._coerce_confidence(solver_result.get("answer_confidence"))

        if not solver_answer:
            _log.warning("Text-only fast path produced no answer")
            return "1) Не удалось определить ответ"

        if not self._looks_like_final_answer(solver_answer):
            _log.warning("Text-only solver answer rejected as non-final: %s", solver_answer)
            return "1) Не удалось определить ответ"

        if solver_confidence >= DIRECT_SOLVER_CONFIDENCE_THRESHOLD:
            _log.info("Returning high-confidence text-only answer: conf=%.3f answer=%s", solver_confidence, solver_answer)
            return self._render_answer_only(solver_answer)

        _log.info("Text-only solver confidence low (%.3f), verifying with verifier", solver_confidence)
        try:
            verifier_content = self._build_verifier_text_only_content(user_text, solver_result)
            verifier_raw = self._request_json(
                self.verifier_model, VERIFIER_SYSTEM_PROMPT, verifier_content, max_tokens=TEXT_ONLY_MAX_TOKENS
            )
            verifier_result = self._normalize_result(verifier_raw, role="verifier")
            verifier_answer = (verifier_result.get("final_answer") or {}).get("value", "").strip()
            verifier_confidence = self._coerce_confidence(verifier_result.get("answer_confidence"))

            if verifier_answer and self._looks_like_final_answer(verifier_answer):
                if self._normalize_answer_text(solver_answer) == self._normalize_answer_text(verifier_answer):
                    _log.info("Text-only solver+verifier agree: answer=%s", solver_answer)
                    return self._render_answer_only(solver_answer)
                if verifier_confidence >= solver_confidence + TIE_BREAK_CONFIDENCE_GAP:
                    _log.info(
                        "Text-only verifier preferred over solver: solver_conf=%.3f verifier_conf=%.3f answer=%s",
                        solver_confidence, verifier_confidence, verifier_answer,
                    )
                    return self._render_answer_only(verifier_answer)
                _log.info(
                    "Text-only disagreement, keeping solver: solver_conf=%.3f verifier_conf=%.3f solver=%s verifier=%s",
                    solver_confidence, verifier_confidence, solver_answer, verifier_answer,
                )
        except RecoverableProviderError as exc:
            _log.warning("Text-only verifier failed, using solver answer: %s", exc)

        _log.info("Returning text-only answer: %s", solver_answer)
        return self._render_answer_only(solver_answer)

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

    def _call_parser(
        self,
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        mismatch_summary: Optional[str] = None,
    ) -> Dict:
        user_content = self._build_parser_content(variants, user_text, mismatch_summary)
        try:
            result = self._request_json(self.parser_model, PARSER_SYSTEM_PROMPT, user_content, max_tokens=PARSER_MAX_TOKENS)
            return self._normalize_result(result, role="parser")
        except RecoverableProviderError as exc:
            _log.warning("Parser unavailable, using degraded parser fallback: %s", exc)
            return self._fallback_parser_result(user_text, variants)

    def _call_solver(
        self,
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        parser_result: Dict,
        mismatch_summary: Optional[str],
    ) -> Dict:
        user_content = self._build_solver_content(variants, user_text, parser_result, mismatch_summary)
        try:
            result = self._request_json(self.solver_model, SOLVER_SYSTEM_PROMPT, user_content, max_tokens=SOLVER_MAX_TOKENS)
            return self._normalize_result(result, role="solver")
        except RecoverableProviderError as exc:
            _log.warning("Primary solver unavailable, trying degraded solver fallback: %s", exc)
            if self.verifier_model != self.solver_model:
                try:
                    result = self._request_json(self.verifier_model, SOLVER_SYSTEM_PROMPT, user_content, max_tokens=SOLVER_MAX_TOKENS)
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
            return self._fallback_solver_result(user_text, parser_result)

    def _call_solver_text_only(self, user_text: str) -> Dict:
        user_content = self._build_solver_text_only_content(user_text)
        result = self._request_json(self.solver_model, SOLVER_TEXT_ONLY_SYSTEM_PROMPT, user_content, max_tokens=TEXT_ONLY_MAX_TOKENS)
        return self._normalize_result(result, role="solver")

    def _call_option_arbiter(
        self,
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        parser_result: Dict,
        mismatch_summary: Optional[str],
    ) -> Optional[Dict]:
        prompt = (
            "You are adjudicating a multiple-choice task from source images.\n"
            "Independently evaluate each numbered option from the source image and OCR.\n"
            "Do not summarize. Return strict JSON.\n"
            "final_answer.value must contain only the concatenated digits of all correct options in ascending order.\n"
            "If none are correct, return an empty string.\n"
            "%s\n" % SOLVER_JSON_SCHEMA_NOTE
        )
        if user_text:
            prompt += "User hint:\n%s\n" % user_text
        prompt += "Parser parse:\n%s\n" % json.dumps(
            self._compact_result_for_prompt(parser_result, role="parser"),
            ensure_ascii=False,
        )
        if mismatch_summary:
            prompt += "Mismatch summary:\n%s\n" % mismatch_summary
        user_content = [{"type": "text", "text": prompt}]
        user_content.extend(self._build_image_blocks(variants))
        try:
            result = self._request_json(self.parser_model, SOLVER_SYSTEM_PROMPT, user_content, max_tokens=SOLVER_MAX_TOKENS)
            normalized = self._normalize_result(result, role="solver")
            normalized["_solver_origin"] = "option_arbiter"
            return normalized
        except RecoverableProviderError as exc:
            _log.warning("Option arbiter unavailable on hard case: %s", exc)
            return None

    def _call_verifier(
        self,
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        parser_result: Dict,
        solver_result: Dict,
        mismatch_summary: Optional[str],
    ) -> Dict:
        ambiguities = (solver_result.get("visual_interpretation") or {}).get("possible_ambiguities") or []
        if "solver used verifier model" in ambiguities:
            _log.warning("Skipping verifier request because verifier already served as degraded solver")
            return self._fallback_verifier_result(
                solver_result,
                reason="verifier skipped because solver already used verifier model",
                mirrors_solver=True,
            )
        user_content = self._build_verifier_content(variants, user_text, parser_result, solver_result, mismatch_summary)
        try:
            result = self._request_json(self.verifier_model, VERIFIER_SYSTEM_PROMPT, user_content, max_tokens=VERIFIER_MAX_TOKENS)
            return self._normalize_result(result, role="verifier")
        except RecoverableProviderError as exc:
            _log.warning("Verifier unavailable, using degraded verifier fallback: %s", exc)
            return self._fallback_verifier_result(solver_result, reason="verifier unavailable", mirrors_solver=False)

    def _build_parser_content(
        self,
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        mismatch_summary: Optional[str],
    ) -> List[Dict]:
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
        if mismatch_summary:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Self-check instruction:\n"
                        "Re-parse the source from scratch. Treat the mismatch summary only as a warning about possible OCR, "
                        "diagram, or target-extraction mistakes.\n"
                        "Self-check mismatch summary:\n%s" % mismatch_summary
                    ),
                }
            )
        content.extend(self._build_image_blocks(variants))
        return content

    def _build_solver_content(
        self,
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        parser_result: Dict,
        mismatch_summary: Optional[str],
    ) -> List[Dict]:
        prompt = (
            "Solve the user task and return strict JSON.\n"
            "Use the parser OCR extract as the source of truth for text, values, and structure.\n"
            "The materials may contain several independent tasks; solve all of them in source order.\n"
            "Prioritize faithful interpretation of the parser extract over assumptions.\n"
            "Return a compact JSON object focused on target, constraints, confidence, and final answer.\n"
            "%s\n" % SOLVER_JSON_SCHEMA_NOTE
        )
        if user_text:
            prompt += "User hint:\n%s\n" % user_text
        prompt += "Parser OCR extract:\n%s\n" % json.dumps(
            self._compact_result_for_prompt(parser_result, role="parser"),
            ensure_ascii=False,
        )
        if mismatch_summary:
            prompt += (
                "Self-check instruction:\n"
                "Resolve the task again from scratch. "
                "Treat the mismatch summary only as a warning about possible failure modes, not as ground truth.\n"
            )
            prompt += "Self-check mismatch summary:\n%s\n" % mismatch_summary

        return [{"type": "text", "text": prompt}]

    def _build_verifier_content(
        self,
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        parser_result: Dict,
        solver_result: Dict,
        mismatch_summary: Optional[str],
    ) -> List[Dict]:
        has_image = bool(variants)
        if has_image:
            prompt = (
                "Verify the user task from the attached text and images and return strict JSON.\n"
                "Check the interpretation, targets, and final answer for all tasks in order.\n"
                "Prefer literal OCR text, visible labels, and explicit numeric values from the source over inferred structure when they conflict.\n"
                "Do not blindly copy the solver. If unsure, lower confidence or mark ambiguity.\n"
                "Return a compact JSON object focused on inconsistencies, confidence, and final answer.\n"
                "%s\n" % SOLVER_JSON_SCHEMA_NOTE
            )
        else:
            prompt = (
                "Verify the user text task and return strict JSON.\n"
                "Check the target, reasoning, and final answer.\n"
                "Do not blindly copy the solver. If unsure, lower confidence or mark ambiguity.\n"
                "Return a compact JSON object focused on inconsistencies, confidence, and final answer.\n"
                "%s\n" % SOLVER_JSON_SCHEMA_NOTE
            )
        if user_text:
            prompt += "User hint:\n%s\n" % user_text
        prompt += "Parser parse:\n%s\n" % json.dumps(
            self._compact_result_for_prompt(parser_result, role="parser"),
            ensure_ascii=False,
        )
        prompt += "Solver result:\n%s\n" % json.dumps(
            self._compact_result_for_prompt(solver_result, role="solver"),
            ensure_ascii=False,
        )
        if mismatch_summary:
            prompt += (
                "Self-check instruction:\n"
                "Re-verify the task from the source and do not anchor on the previous answer if the source suggests otherwise.\n"
            )
            prompt += "Self-check mismatch summary:\n%s\n" % mismatch_summary

        content = [{"type": "text", "text": prompt}]
        content.extend(self._build_image_blocks(variants))
        return content

    def _build_solver_text_only_content(self, user_text: str) -> List[Dict]:
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

    def _build_verifier_text_only_content(self, user_text: str, solver_result: Dict) -> List[Dict]:
        prompt = (
            "Independently verify the solution to this text task and return strict JSON.\n"
            "Do not blindly copy the solver. Solve from scratch, then compare.\n"
            "If unsure, lower confidence. Output only the final answer.\n"
            "%s\n" % SOLVER_JSON_SCHEMA_NOTE
        )
        if user_text:
            prompt += "User task:\n%s\n" % user_text
        prompt += "Solver proposed answer (verify independently):\n%s\n" % json.dumps(
            self._compact_result_for_prompt(solver_result, role="solver"),
            ensure_ascii=False,
        )
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
            "provider": {"allow_fallbacks": True, "data_collection": "allow", "sort": "throughput"},
            "plugins": [{"id": "response-healing"}],
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
                self._sleep_before_retry(attempt, status_code=response.status_code)
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
            repair_models = self._repair_models_for_source(model)
            prefer_remote_repair = self._should_prefer_remote_repair(raw_text)
            salvaged = None
            if not prefer_remote_repair:
                salvaged = self._try_salvage_answer_only(raw_text)
                if salvaged is not None:
                    parsed = salvaged
                    used_repair = True
                    repair_model = "local_answer_salvage"
                    _log.info("Local answer salvage succeeded: source_model=%s", model)
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
            if parsed is None and prefer_remote_repair:
                salvaged = self._try_salvage_answer_only(raw_text)
                if salvaged is not None:
                    parsed = salvaged
                    used_repair = True
                    repair_model = "local_answer_salvage"
                    _log.info("Local answer salvage succeeded after remote-repair attempts: source_model=%s", model)
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
        # Parser (vision parser) → repair with fast text models, not Parser itself
        if model == self.parser_model:
            candidates = []
            if self.solver_model != self.parser_model:
                candidates.append(self.solver_model)
            if self.verifier_model not in candidates:
                candidates.append(self.verifier_model)
            return candidates or [self.solver_model]
        # Solver → repair with verifier
        if model == self.solver_model:
            return [self.verifier_model] if self.verifier_model != self.solver_model else [self.solver_model]
        # Verifier or other → repair with solver
        return [self.solver_model]

    def _is_retryable_status(self, status_code: int) -> bool:
        return status_code in RETRYABLE_STATUSES

    def _sleep_before_retry(self, attempt: int, status_code: int = 0) -> None:
        if status_code == 429:
            delay = 4.0 if attempt <= 1 else 8.0
        else:
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
                "provider": {"allow_fallbacks": True, "data_collection": "allow", "sort": "throughput"},
                "plugins": [{"id": "response-healing"}],
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
        if not answer:
            answer = self._extract_loose_terminal_answer(text)

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

    def _should_prefer_remote_repair(self, raw_text: str) -> bool:
        text = raw_text or ""
        if not text.strip():
            return False
        if re.search(r"(?im)^\s*(?:final[_\s-]*answer|answer|итог(?:овый)?\s*ответ|ответ)\s*[:=-]", text):
            return False
        numbered_lines = re.findall(r"(?im)^\s*\d+\)\s+\S", text)
        if len(numbered_lines) >= 3:
            return True
        return False

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

    def _extract_loose_terminal_answer(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        tail_lines = lines[-12:]

        for line in reversed(tail_lines):
            match = re.search(r"=\s*([-+A-Za-z0-9.,/]+)\s*[.]?$", line)
            if match:
                candidate = match.group(1).strip()
                if self._is_simple_answer_candidate(candidate):
                    return candidate

        cue_patterns = (
            r"(?i)\b(?:therefore|thus|hence|so|finally|result)\b[^.\n]{0,80}?\b(?:is|=)\s*([-+A-Za-z0-9.,/]+)",
            r"(?i)\b(?:итак|следовательно|значит|получаем)\b[^.\n]{0,80}?\b(?:=|это)\s*([-+A-Za-z0-9.,/]+)",
        )
        tail_text = "\n".join(tail_lines)
        for pattern in cue_patterns:
            match = re.search(pattern, tail_text)
            if match:
                candidate = match.group(1).strip()
                if self._is_simple_answer_candidate(candidate):
                    return candidate

        return ""

    def _normalize_salvaged_answer(self, answer: str) -> str:
        text = (answer or "").strip().strip("`").strip()
        if not text:
            return ""
        if len(text) > 200:
            return ""
        text = re.sub(r"(?i)^(final[_\s-]*answer|answer|итог(?:овый)?\s*ответ|ответ)\s*[:=-]\s*", "", text).strip()
        text = text.rstrip(".;,")
        text = re.sub(r"\s+$", "", text)
        if not text:
            return ""
        if re.search(r"[{}\\[\\]]", text):
            return ""
        return text

    def _is_simple_answer_candidate(self, value: str) -> bool:
        candidate = (value or "").strip().strip(".;,")
        if not candidate or len(candidate) > 40:
            return False
        if re.match(r"^[A-Za-z]$", candidate):
            return False
        return bool(re.match(r"^[-+A-Za-z0-9.,/]+$", candidate))

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

    def _compact_result_for_prompt(self, result: Dict, role: str) -> Dict:
        compact = {
            "task_type": result.get("task_type") or "mixed_task",
            "ocr_text": self._truncate_text(result.get("ocr_text") or "", 450),
            "normalized_problem_text": self._truncate_text(result.get("normalized_problem_text") or "", 700),
            "target": self._normalize_target(result.get("target")),
            "givens": self._truncate_statements(result.get("givens") or [], limit=6, width=120),
            "diagram_relations": self._truncate_text_list(result.get("diagram_relations") or [], limit=6, width=80),
            "visual_interpretation": {
                "summary": self._truncate_text(((result.get("visual_interpretation") or {}).get("summary") or ""), 220),
                "confidence": self._coerce_confidence((result.get("visual_interpretation") or {}).get("confidence")),
                "possible_ambiguities": self._truncate_text_list(
                    ((result.get("visual_interpretation") or {}).get("possible_ambiguities") or []),
                    limit=4,
                    width=80,
                ),
            },
            "needs_clarification": bool(result.get("needs_clarification", False)),
        }
        if role != "parser":
            compact["final_answer"] = {
                "value": self._truncate_text(((result.get("final_answer") or {}).get("value") or ""), 80),
                "format": str(((result.get("final_answer") or {}).get("format") or "text")),
            }
            compact["answer_confidence"] = self._coerce_confidence(result.get("answer_confidence"))
            compact["consistency_checks"] = self._truncate_text_list(result.get("consistency_checks") or [], limit=4, width=100)
        return compact

    def _truncate_text(self, text: str, width: int) -> str:
        normalized = str(text or "").strip()
        if len(normalized) <= width:
            return normalized
        return normalized[: max(0, width - 1)].rstrip() + "…"

    def _truncate_text_list(self, value, limit: int, width: int) -> List[str]:
        items = self._normalize_text_list(value)
        return [self._truncate_text(item, width) for item in items[:limit]]

    def _truncate_statements(self, givens, limit: int, width: int) -> List[Dict]:
        normalized = self._normalize_givens(givens)
        trimmed = []
        for item in normalized[:limit]:
            trimmed.append({"statement": self._truncate_text(item.get("statement") or "", width)})
        return trimmed

    def _fallback_parser_result(self, user_text: str, variants: List[Dict[str, Optional[str]]]) -> Dict:
        summary = "Parser fallback used because Parser was temporarily unavailable."
        if variants:
            summary += " The attached images should still be checked directly by solver and verifier."
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
            "_request_meta": {"model": self.parser_model, "used_repair": False, "repair_model": "", "raw_text_chars": 0},
            "_solver_origin": "parser_fallback",
            "_mirrors_solver": False,
        }

    def _fallback_verifier_result(self, solver_result: Dict, reason: str, mirrors_solver: bool) -> Dict:
        final_answer = solver_result.get("final_answer") or {"value": "", "format": "text"}
        return {
            "task_type": solver_result.get("task_type") or "mixed_task",
            "ocr_text": solver_result.get("ocr_text") or "",
            "normalized_problem_text": solver_result.get("normalized_problem_text") or "",
            "diagram_entities": solver_result.get("diagram_entities") or [],
            "diagram_relations": solver_result.get("diagram_relations") or [],
            "givens": solver_result.get("givens") or [],
            "target": solver_result.get("target") or {"statement": ""},
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
            "_request_meta": {"model": self.verifier_model, "used_repair": False, "repair_model": "", "raw_text_chars": 0},
            "_solver_origin": "verifier_fallback",
            "_mirrors_solver": mirrors_solver,
        }

    def _fallback_solver_result(self, user_text: str, parser_result: Dict) -> Dict:
        final_answer = (parser_result.get("final_answer") or {}).get("value", "")
        ambiguities = ["primary solver unavailable"]
        parser_ambiguities = (parser_result.get("visual_interpretation") or {}).get("possible_ambiguities") or []
        ambiguities.extend([item for item in parser_ambiguities if item not in ambiguities])
        return {
            "task_type": parser_result.get("task_type") or "mixed_task",
            "ocr_text": parser_result.get("ocr_text") or user_text or "",
            "normalized_problem_text": parser_result.get("normalized_problem_text") or user_text or "",
            "diagram_entities": parser_result.get("diagram_entities") or [],
            "diagram_relations": parser_result.get("diagram_relations") or [],
            "givens": parser_result.get("givens") or [],
            "target": parser_result.get("target") or {"statement": ""},
            "visual_interpretation": {
                "summary": "Solver fallback used because primary solver was temporarily unavailable.",
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
            "_request_meta": {"model": self.solver_model, "used_repair": False, "repair_model": "", "raw_text_chars": 0},
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

    def _compare_results(self, solver: Dict, parser: Dict, verifier: Dict) -> Dict:
        parser_diagram = self._jaccard(self._relation_keys(solver), self._relation_keys(parser))
        parser_givens = self._jaccard(self._given_keys(solver), self._given_keys(parser))
        target_agreement = 1.0 if self._normalize_text(solver["target"].get("statement", "")) == self._normalize_text(parser["target"].get("statement", "")) else 0.0
        verifier_independent = self._has_independent_verifier(verifier)
        if verifier_independent:
            verifier_diagram = self._jaccard(self._relation_keys(solver), self._relation_keys(verifier))
            verifier_givens = self._jaccard(self._given_keys(solver), self._given_keys(verifier))
            answer_agreement = 1.0 if self._normalize_answer_text(solver["final_answer"].get("value", "")) == self._normalize_answer_text(verifier["final_answer"].get("value", "")) else 0.0
            diagram_agreement = (parser_diagram + verifier_diagram) / 2.0
            givens_agreement = (parser_givens + verifier_givens) / 2.0
            confidence_alignment = (
                solver["visual_interpretation"]["confidence"] +
                parser["visual_interpretation"]["confidence"] +
                verifier["visual_interpretation"]["confidence"]
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
            diagram_agreement = parser_diagram
            givens_agreement = parser_givens
            confidence_alignment = (
                solver["visual_interpretation"]["confidence"] +
                parser["visual_interpretation"]["confidence"]
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
        if parser["visual_interpretation"]["possible_ambiguities"]:
            reasons.append("diagram ambiguity detected")
        if parser.get("needs_clarification"):
            reasons.append("parser requested clarification")
        if self._used_repair(solver):
            reasons.append("solver JSON repaired")
            score -= 0.08
        if self._solver_is_degraded(solver):
            reasons.append("solver fallback path")
            score -= 0.20

        score = max(0.0, min(1.0, score))

        # When solver and independent verifier agree on a non-empty answer, that's
        # strong evidence regardless of structural field similarity (diagram/givens
        # fields are often empty for non-geometry tasks).
        solver_ans = self._normalize_answer_text(solver["final_answer"].get("value", ""))
        verifier_ans = self._normalize_answer_text(verifier["final_answer"].get("value", ""))
        answers_agree = (
            verifier_independent and
            solver_ans and verifier_ans and
            solver_ans == verifier_ans and
            not self._solver_is_degraded(solver) and
            not parser.get("needs_clarification")
        )
        if answers_agree:
            score = max(score, ACCEPT_SCORE_THRESHOLD)

        status = "ambiguous"
        accepted = False
        if (
            verifier_independent and
            score >= ACCEPT_SCORE_THRESHOLD and
            not parser.get("needs_clarification") and
            not self._solver_is_degraded(solver) and
            not self._used_repair(solver)
        ):
            status = "accepted"
            accepted = True
        elif score >= SELF_CHECK_SCORE_THRESHOLD or self._used_repair(solver) or self._solver_is_degraded(solver):
            status = "self_check"

        return {
            "accepted": accepted,
            "score": score,
            "status": status,
            "final_answer": solver["final_answer"].get("value", ""),
            "diagram_agreement": diagram_agreement,
            "givens_agreement": givens_agreement,
            "answer_agreement": answer_agreement,
            "reasons": reasons,
        }

    def _should_run_self_check(self, consensus: Dict, solver: Dict, verifier: Optional[Dict]) -> bool:
        if not self._has_independent_verifier(verifier):
            return False
        if self._can_accept_verifier_over_local_salvage(consensus, solver, verifier):
            return False
        if self._can_accept_repaired_match_without_self_check(consensus, solver, verifier):
            return False
        if consensus["status"] == "accepted" and not self._requires_quality_escalation(consensus, solver, verifier):
            return False
        if consensus["status"] == "accepted":
            return True
        if self._answers_effectively_match(solver, verifier):
            return self._requires_quality_escalation(consensus, solver, verifier)
        if consensus["status"] == "self_check":
            return True
        solver_answer = (solver.get("final_answer") or {}).get("value", "").strip()
        verifier_answer = ""
        if verifier is not None:
            verifier_answer = (verifier.get("final_answer") or {}).get("value", "").strip()
        return bool(solver_answer or verifier_answer)

    def _build_mismatch_summary(self, consensus: Dict, solver: Dict, parser: Dict, verifier: Dict) -> str:
        parts = [
            "consensus_score=%.3f" % consensus["score"],
            "reasons=%s" % ", ".join(consensus["reasons"]),
            "solver_answer=%s" % solver["final_answer"].get("value", ""),
            "verifier_answer=%s" % verifier["final_answer"].get("value", ""),
            "parser_ambiguities=%s" % "; ".join(parser["visual_interpretation"]["possible_ambiguities"]),
        ]
        return "\n".join([part for part in parts if part.strip()])

    def _format_user_result(self, consensus: Dict, solver: Dict, parser: Dict, verifier: Optional[Dict], option_arbiter: Optional[Dict] = None) -> str:
        final_answer = self._pick_user_answer(consensus, solver, parser, verifier, option_arbiter)
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

    def _pick_user_answer(self, consensus: Dict, solver: Dict, parser: Dict, verifier: Optional[Dict], option_arbiter: Optional[Dict] = None) -> str:
        solver_answer = (solver.get("final_answer") or {}).get("value", "").strip()
        if solver_answer and not self._looks_like_final_answer(solver_answer):
            _log.warning("Rejecting solver answer that does not look like a final answer: %s", solver_answer)
            solver_answer = ""
        verifier_answer = ""
        verifier_confidence = 0.0
        if verifier is not None:
            verifier_answer = (verifier.get("final_answer") or {}).get("value", "").strip()
            if verifier_answer and not self._looks_like_final_answer(verifier_answer):
                _log.warning("Rejecting verifier answer that does not look like a final answer: %s", verifier_answer)
                verifier_answer = ""
            verifier_confidence = self._coerce_confidence(verifier.get("answer_confidence"))
        solver_confidence = self._coerce_confidence(solver.get("answer_confidence"))
        option_arbiter_answer = ""
        option_arbiter_confidence = 0.0
        if option_arbiter is not None:
            option_arbiter_answer = (option_arbiter.get("final_answer") or {}).get("value", "").strip()
            if option_arbiter_answer and not self._looks_like_final_answer(option_arbiter_answer):
                _log.warning("Rejecting Parser challenger answer that does not look like a final answer: %s", option_arbiter_answer)
                option_arbiter_answer = ""
            option_arbiter_confidence = self._coerce_confidence(option_arbiter.get("answer_confidence"))

        if consensus["status"] == "accepted":
            return solver_answer or verifier_answer

        answers_match = self._answers_effectively_match(solver, verifier)
        parser_unavailable = "parser unavailable" in (parser.get("visual_interpretation") or {}).get("possible_ambiguities", [])
        parser_explicit_ambiguity = self._parser_has_explicit_ambiguity(parser)
        parser_clear = not parser_explicit_ambiguity
        verifier_independent = self._has_independent_verifier(verifier)
        solver_degraded = self._solver_is_degraded(solver)
        solver_repaired = self._used_repair(solver)
        if self._can_accept_option_arbiter(
            consensus,
            parser,
            solver,
            verifier,
            option_arbiter,
            option_arbiter_answer,
            option_arbiter_confidence,
            parser_clear,
            parser_unavailable,
        ):
            _log.info(
                "Accepting Parser challenger answer on hard case: score=%.3f solver_repaired=%s arbiter_confidence=%.3f answer=%s",
                consensus["score"],
                solver_repaired,
                option_arbiter_confidence,
                option_arbiter_answer,
            )
            return option_arbiter_answer

        if not verifier_independent:
            if solver_degraded:
                _log.warning("Rejecting answer because only degraded solver output is available without an independent verifier")
                return ""
            if solver_answer and parser_clear and not parser_unavailable and solver_repaired and solver_confidence >= REPAIRED_MATCH_CONFIDENCE_THRESHOLD:
                _log.info(
                    "Accepting repaired solver answer without independent verifier because parser is clear "
                    "and solver produced a stable final answer: solver_confidence=%.3f",
                    solver_confidence,
                )
                return solver_answer
            if solver_answer and parser_clear and not solver_repaired and solver_confidence >= DIRECT_SOLVER_CONFIDENCE_THRESHOLD:
                _log.info("Accepting direct solver answer without verifier because parser is clear and solver confidence is high")
                return solver_answer
            return ""

        if self._can_accept_repaired_match_without_self_check(consensus, solver, verifier):
            _log.info("Accepting repaired solver answer because independent verifier matches and parser is clear")
            return solver_answer or verifier_answer

        if self._can_accept_verifier_over_local_salvage(consensus, solver, verifier):
            _log.info("Accepting verifier answer because local answer salvage conflicts with independent verifier")
            return verifier_answer

        if answers_match and parser_clear and not solver_degraded and not solver_repaired and consensus["score"] >= SELF_CHECK_SCORE_THRESHOLD:
            _log.info("Accepting answer after non-accepted consensus because independent solver and verifier match under clear parse")
            return solver_answer

        if solver_answer and not verifier_answer:
            if parser_clear and not solver_degraded and not solver_repaired and solver_confidence >= DIRECT_SOLVER_CONFIDENCE_THRESHOLD:
                _log.info("Accepting solver answer because verifier did not provide a competing answer and parser is clear")
                return solver_answer
            return ""

        if verifier_answer and not solver_answer:
            if (
                parser_clear
                and not parser_unavailable
                and verifier_independent
                and not self._used_repair(verifier or {})
                and verifier_confidence >= VERIFIER_ONLY_CONFIDENCE_THRESHOLD
            ):
                _log.info(
                    "Accepting verifier-only answer because primary solver omitted final_answer, "
                    "parser is clear, and verifier confidence is high: score=%.3f verifier_confidence=%.3f",
                    consensus["score"],
                    verifier_confidence,
                )
                return verifier_answer
            _log.warning(
                "Rejecting verifier-only answer because primary solver did not provide a stable answer: "
                "status=%s score=%.3f verifier_confidence=%.3f",
                consensus["status"],
                consensus["score"],
                verifier_confidence,
            )
            return ""

        if solver_answer and verifier_answer and parser_clear and not solver_degraded and not solver_repaired:
            if solver_confidence >= verifier_confidence + TIE_BREAK_CONFIDENCE_GAP and consensus["score"] >= SELF_CHECK_SCORE_THRESHOLD:
                _log.info("Accepting solver answer on confidence tie-break: solver=%.3f verifier=%.3f", solver_confidence, verifier_confidence)
                return solver_answer
            if verifier_confidence >= solver_confidence + TIE_BREAK_CONFIDENCE_GAP and consensus["score"] >= SELF_CHECK_SCORE_THRESHOLD:
                _log.info("Accepting verifier answer on confidence tie-break: solver=%.3f verifier=%.3f", solver_confidence, verifier_confidence)
                return verifier_answer

        if solver_answer and consensus["status"] == "self_check" and parser_clear and not parser_unavailable and not solver_degraded and not solver_repaired and solver_confidence >= DIRECT_SOLVER_CONFIDENCE_THRESHOLD:
            _log.info("Accepting solver answer after self-check because parser is clear and solver output is direct")
            return solver_answer

        return ""

    def _resolve_multi_round_answer(self, default_parser: Dict, first_round: Dict, second_round: Dict, option_arbiter: Optional[Dict] = None) -> str:
        first_parser = first_round.get("parser") or default_parser
        second_parser = second_round.get("parser") or default_parser
        first_answer = self._pick_user_answer(first_round["consensus"], first_round["solver"], first_parser, first_round["verifier"], option_arbiter)
        second_answer = self._pick_user_answer(second_round["consensus"], second_round["solver"], second_parser, second_round["verifier"], option_arbiter)

        if first_answer and not second_answer:
            _log.info("Keeping primary-round answer because self-check did not produce a safe answer")
            return first_answer
        if second_answer and not first_answer:
            _log.info("Using self-check answer because primary round did not produce a safe answer")
            return second_answer
        if not first_answer and not second_answer:
            _log.info("Neither primary nor self-check round produced a safe answer")
            return ""

        normalized_first = self._normalize_answer_text(first_answer)
        normalized_second = self._normalize_answer_text(second_answer)
        if normalized_first == normalized_second:
            if second_round["consensus"]["score"] > first_round["consensus"]["score"]:
                _log.info("Primary and self-check rounds agree on the answer; keeping higher-score self-check round")
                return second_answer
            _log.info("Primary and self-check rounds agree on the answer; keeping primary round")
            return first_answer

        first_quality = self._round_quality_score(first_round)
        second_quality = self._round_quality_score(second_round)
        if first_quality >= second_quality + 0.20:
            _log.info(
                "Keeping primary-round answer after conflicting self-check answers: primary_quality=%.3f self_check_quality=%.3f",
                first_quality,
                second_quality,
            )
            return first_answer
        if second_quality >= first_quality + 0.20:
            _log.info(
                "Using self-check answer after conflicting round answers: primary_quality=%.3f self_check_quality=%.3f",
                first_quality,
                second_quality,
            )
            return second_answer

        _log.warning(
            "Rejecting conflicting answers across primary and self-check rounds: primary=%s self_check=%s primary_quality=%.3f self_check_quality=%.3f",
            first_answer,
            second_answer,
            first_quality,
            second_quality,
        )
        return ""

    def _round_quality_score(self, round_data: Dict) -> float:
        consensus = round_data["consensus"]
        solver = round_data["solver"]
        verifier = round_data["verifier"]

        score = float(consensus.get("score", 0.0))
        if consensus.get("status") == "accepted":
            score += 0.20
        elif consensus.get("status") == "self_check":
            score += 0.05

        if self._answers_effectively_match(solver, verifier):
            score += 0.10
        if not self._used_repair(solver):
            score += 0.05
        if not self._solver_is_degraded(solver):
            score += 0.05
        if self._has_independent_verifier(verifier):
            score += 0.05
        return score

    def _used_repair(self, result: Optional[Dict]) -> bool:
        if not result:
            return False
        request_meta = result.get("_request_meta") or {}
        return bool(request_meta.get("used_repair", False))

    def _used_local_salvage(self, result: Optional[Dict]) -> bool:
        if not result:
            return False
        request_meta = result.get("_request_meta") or {}
        return request_meta.get("repair_model") == "local_answer_salvage"

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

    def _requires_quality_escalation(self, consensus: Dict, solver: Dict, verifier: Optional[Dict]) -> bool:
        return bool(
            consensus["score"] < 0.75 or
            self._used_repair(solver) or
            self._solver_is_degraded(solver) or
            self._used_repair(verifier)
        )

    def _should_run_option_arbiter(self, consensus: Dict, parser: Dict, solver: Dict, verifier: Optional[Dict]) -> bool:
        if not self._is_option_selection_task(parser):
            return False
        if "parser unavailable" in ((parser.get("visual_interpretation") or {}).get("possible_ambiguities") or []):
            return False
        if self._parser_has_explicit_ambiguity(parser):
            return False
        if not self._has_option_answer_disagreement(solver, verifier):
            return False
        if consensus.get("score", 0.0) >= ACCEPT_SCORE_THRESHOLD:
            return False
        return True

    def _can_accept_option_arbiter(
        self,
        consensus: Dict,
        parser: Dict,
        solver: Dict,
        verifier: Optional[Dict],
        option_arbiter: Optional[Dict],
        option_arbiter_answer: str,
        option_arbiter_confidence: float,
        parser_clear: bool,
        parser_unavailable: bool,
    ) -> bool:
        if not option_arbiter or not option_arbiter_answer:
            return False
        if option_arbiter.get("_solver_origin") != "option_arbiter":
            return False
        if parser_unavailable or not parser_clear:
            return False
        if self._used_repair(option_arbiter):
            return False
        if self._solver_is_degraded(option_arbiter):
            return False
        if option_arbiter_confidence < QWEN_CHALLENGER_CONFIDENCE_THRESHOLD:
            return False
        if not self._looks_like_option_answer(option_arbiter_answer):
            return False
        if not self._has_option_answer_disagreement(solver, verifier):
            return False
        return True

    def _is_option_selection_task(self, parser: Dict) -> bool:
        texts = [
            parser.get("ocr_text") or "",
            parser.get("normalized_problem_text") or "",
            (parser.get("target") or {}).get("statement") or "",
        ]
        combined = "\n".join(texts).lower()
        if not combined.strip():
            return False
        numbered_items = len(re.findall(r"(?m)(?:^|\s)\d+\)", combined))
        if numbered_items < 3:
            return False
        option_markers = (
            "запишите номера",
            "укажите номера",
            "запишите цифры",
            "укажите цифры",
            "запишите в ответ цифры",
            "в ответ запишите цифры",
            "запишите в ответ номера",
            "выберите из предложенного списка",
            "выберите все",
            "установите соответствие",
            "выберите верные",
            "какие из",
            "selected pairs",
            "write the numbers",
            "select from the proposed list",
        )
        return any(marker in combined for marker in option_markers)

    def _looks_like_option_answer(self, answer: str) -> bool:
        text = (answer or "").strip()
        return bool(re.match(r"^[1-9]{1,9}$", text))

    def _has_option_answer_disagreement(self, solver: Dict, verifier: Optional[Dict]) -> bool:
        solver_answer = ((solver.get("final_answer") or {}).get("value") or "").strip()
        verifier_answer = ""
        if verifier is not None:
            verifier_answer = ((verifier.get("final_answer") or {}).get("value") or "").strip()
        option_like = [answer for answer in (solver_answer, verifier_answer) if self._looks_like_option_answer(answer)]
        if len(option_like) >= 2 and len(set(option_like)) >= 2:
            return True
        if len(option_like) == 1:
            return True
        return False

    def _can_accept_repaired_match_without_self_check(self, consensus: Dict, solver: Dict, verifier: Optional[Dict]) -> bool:
        if not self._used_repair(solver):
            return False
        if self._solver_is_degraded(solver):
            return False
        if not self._has_independent_verifier(verifier):
            return False
        if not self._answers_effectively_match(solver, verifier):
            return False
        solver_answer = ((solver.get("final_answer") or {}).get("value") or "").strip()
        if not self._looks_like_final_answer(solver_answer):
            return False
        match_clear = True
        solver_visual = solver.get("visual_interpretation") or {}
        ambiguities = solver_visual.get("possible_ambiguities") or []
        if "recovered from non-json output" in ambiguities:
            if self._coerce_confidence((verifier or {}).get("answer_confidence")) < 0.75:
                return False
        solver_confidence = self._coerce_confidence(solver.get("answer_confidence"))
        verifier_confidence = self._coerce_confidence((verifier or {}).get("answer_confidence"))
        if solver_confidence < REPAIRED_MATCH_CONFIDENCE_THRESHOLD and verifier_confidence < REPAIRED_MATCH_CONFIDENCE_THRESHOLD:
            return False
        if consensus.get("answer_agreement", 0.0) < 1.0:
            return False
        return match_clear

    def _looks_like_final_answer(self, answer: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return False
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if any(re.match(r"^\d+\)\s*", line) for line in lines):
            normalized_lines = [re.sub(r"^\d+\)\s*", "", line).strip() for line in lines]
            return all(self._looks_like_final_answer(line) for line in normalized_lines if line)

        if len(text) > 160:
            return False
        lowered = text.lower()
        forbidden_markers = (
            "dependent on",
            "unknown",
            "cannot determine",
            "insufficient",
            "need clarification",
            "зависит от",
            "неизвест",
            "невозможно определить",
            "нельзя определить",
            "уточн",
        )
        for marker in forbidden_markers:
            if marker in lowered:
                return False
        if len(text.split()) > 15:
            return False
        return True

    def _can_accept_verifier_over_local_salvage(self, consensus: Dict, solver: Dict, verifier: Optional[Dict]) -> bool:
        if not self._used_local_salvage(solver):
            return False
        if not self._has_independent_verifier(verifier):
            return False
        if self._answers_effectively_match(solver, verifier):
            return False
        verifier_answer = ((verifier or {}).get("final_answer") or {}).get("value", "").strip()
        if not verifier_answer:
            return False
        if self._coerce_confidence((verifier or {}).get("answer_confidence")) < 0.75:
            return False
        return True

    def _parser_has_explicit_ambiguity(self, parser: Dict) -> bool:
        ambiguities = ((parser.get("visual_interpretation") or {}).get("possible_ambiguities") or [])
        if ambiguities:
            return True
        return False

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

    def _answers_effectively_match(self, solver: Dict, verifier: Optional[Dict]) -> bool:
        solver_answer = (solver.get("final_answer") or {}).get("value", "").strip()
        if not solver_answer or verifier is None:
            return False
        verifier_answer = (verifier.get("final_answer") or {}).get("value", "").strip()
        if not verifier_answer:
            return False
        return self._normalize_answer_text(solver_answer) == self._normalize_answer_text(verifier_answer)

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
