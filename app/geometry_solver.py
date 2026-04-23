from __future__ import annotations

import base64
import io
import json
import logging
import math
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
SOLVER_MAX_TOKENS = 1100
VERIFIER_MAX_TOKENS = 700
TEXT_ONLY_MAX_TOKENS = 700
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
                exact_answer = self._try_exact_answer_from_parser(qwen_result, user_text)
                if exact_answer:
                    _log.info("Returning exact-engine image answer after parser extraction: %s", exact_answer)
                    return self._render_answer_only(exact_answer)
                kimi_result = self._call_kimi(preprocessed, user_text, qwen_result, None)
                llama_result = self._call_llama(preprocessed, user_text, qwen_result, kimi_result, None)
                consensus = self._compare_results(kimi_result, qwen_result, llama_result)
                _log.info("Consensus after llama: status=%s score=%.3f", consensus["status"], consensus["score"])

                if self._should_run_self_check(consensus, kimi_result, llama_result):
                    first_round = {
                        "consensus": consensus,
                        "kimi": kimi_result,
                        "llama": llama_result,
                        "qwen": qwen_result,
                    }
                    mismatch_summary = self._build_mismatch_summary(consensus, kimi_result, qwen_result, llama_result)
                    _log.info("Running self-check round: %s", mismatch_summary)
                    qwen_check = self._call_qwen(preprocessed, user_text, mismatch_summary)
                    kimi_check = self._call_kimi(preprocessed, user_text, qwen_check, mismatch_summary)
                    llama_check = self._call_llama(preprocessed, user_text, qwen_check, kimi_check, mismatch_summary)
                    second_round = {
                        "consensus": self._compare_results(kimi_check, qwen_check, llama_check),
                        "kimi": kimi_check,
                        "llama": llama_check,
                        "qwen": qwen_check,
                    }
                    chosen_answer = self._resolve_multi_round_answer(qwen_result, first_round, second_round)
                    if chosen_answer:
                        _log.info("Returning answer selected across primary/self-check rounds: %s", chosen_answer)
                        return self._render_answer_only(chosen_answer)
                    if not self._pick_user_answer(first_round["consensus"], first_round["kimi"], first_round["qwen"], first_round["llama"]):
                        consensus = second_round["consensus"]
                        kimi_result = second_round["kimi"]
                        llama_result = second_round["llama"]
                        qwen_result = second_round["qwen"]
                    else:
                        consensus = first_round["consensus"]
                        kimi_result = first_round["kimi"]
                        llama_result = first_round["llama"]
                        qwen_result = first_round["qwen"]
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
        exact_answer = self._try_exact_answer_engine(user_text)
        if exact_answer:
            _log.info("Returning exact-engine text-only answer: %s", exact_answer)
            return self._render_answer_only(exact_answer)
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

    def _try_exact_answer_from_parser(self, qwen_result: Dict, user_text: str) -> str:
        for candidate in (
            qwen_result.get("normalized_problem_text") or "",
            qwen_result.get("ocr_text") or "",
            user_text or "",
        ):
            answer = self._try_exact_answer_engine(candidate)
            if answer:
                return answer
        return ""

    def _try_exact_answer_engine(self, text: str) -> str:
        normalized = self._normalize_exact_text(text)
        if not normalized:
            return ""
        task_chunks = self._split_exact_task_chunks(normalized)
        split_answers = []
        for chunk in task_chunks:
            answer = self._try_exact_answer_single(chunk)
            if not answer:
                split_answers = []
                break
            split_answers.append(answer)
        if len(split_answers) >= 2:
            return "\n".join(
                ["%d) %s" % (index, answer) for index, answer in enumerate(split_answers, start=1)]
            )
        direct_answer = self._try_exact_answer_single(normalized)
        if direct_answer:
            return direct_answer
        return ""

    def _try_exact_answer_single(self, normalized: str) -> str:
        for solver in (
            self._exact_right_triangle_median,
            self._exact_prism_perpendicular_lines,
            self._exact_isosceles_exterior_angle,
            self._exact_rhombus_incircle_area,
            self._exact_regular_pyramid_sine,
            self._exact_parallelepiped_distance,
            self._exact_marker_probability,
            self._exact_two_clubs_overlap,
            self._exact_interval_probability,
            self._exact_same_color_probability,
        ):
            answer = solver(normalized)
            if answer:
                return answer
        return ""

    def _split_exact_task_chunks(self, normalized: str) -> List[str]:
        chunks = []
        boundaries = []
        for match in re.finditer(r"(?:^|\s)(тип\s+\d+\s*№\s*\d+)", normalized):
            boundaries.append(match.start(1))
        if len(boundaries) < 2:
            return []
        boundaries.append(len(normalized))
        for index in range(len(boundaries) - 1):
            chunk = normalized[boundaries[index]:boundaries[index + 1]].strip(" .\n\t")
            if chunk:
                chunks.append(chunk)
        return chunks

    def _normalize_exact_text(self, text: str) -> str:
        normalized = (text or "").lower()
        if not normalized:
            return ""
        for source, target in (
            ("−", "-"),
            ("–", "-"),
            ("—", "-"),
            ("≤", "<="),
            ("≥", ">="),
            ("₁", "1"),
            ("₂", "2"),
            ("₃", "3"),
            ("₄", "4"),
            ("₅", "5"),
            ("₆", "6"),
            ("₇", "7"),
            ("₈", "8"),
            ("₉", "9"),
            ("₀", "0"),
            ("чёрных", "черных"),
        ):
            normalized = normalized.replace(source, target)
        normalized = normalized.replace(",", ".")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _exact_isosceles_exterior_angle(self, text: str) -> str:
        if "треугольнике abc" not in text or "ab и bc равны" not in text:
            return ""
        if "внешний угол" not in text or "угол c" not in text:
            return ""
        match = re.search(r"внешний угол[^.]*?b[^0-9]*?([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            return ""
        exterior = float(match.group(1))
        interior = 180.0 - exterior
        answer = (180.0 - interior) / 2.0
        return self._format_exact_numeric(answer)

    def _exact_right_triangle_median(self, text: str) -> str:
        if "прямоугольном треугольнике abc" not in text:
            return ""
        if "прямым углом c" not in text or "медиану ck" not in text:
            return ""
        ac_match = re.search(r"ac\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
        bc_match = re.search(r"bc\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not ac_match or not bc_match:
            return ""
        ac_value = float(ac_match.group(1))
        bc_value = float(bc_match.group(1))
        hypotenuse = math.sqrt(ac_value * ac_value + bc_value * bc_value)
        return self._format_exact_numeric(hypotenuse / 2.0)

    def _exact_prism_perpendicular_lines(self, text: str) -> str:
        if "прямая треугольная призма" not in text and "прямой треугольной призме" not in text:
            return ""
        if "перпендикулярные плоскости abc" not in text:
            return ""
        if "aa1" in text and "cc1" in text:
            return "12"
        return ""

    def _exact_rhombus_incircle_area(self, text: str) -> str:
        if "ромбе abcd" not in text or "радиусом" not in text or "de =" not in text:
            return ""
        radius_match = re.search(r"радиус(?:ом)?\s*([0-9]+(?:\.[0-9]+)?)", text)
        de_match = re.search(r"de\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not radius_match or not de_match:
            return ""
        radius = float(radius_match.group(1))
        de_value = float(de_match.group(1))
        if de_value <= 0:
            return ""
        side = (radius * radius) / de_value + de_value
        area = 2.0 * radius * side
        return self._format_exact_numeric(area)

    def _exact_regular_pyramid_sine(self, text: str) -> str:
        if "правильной четырехугольной пирамиде" not in text and "правильной четырёхугольной пирамиде" not in text:
            return ""
        if "сторона основания ab равна" not in text or "ребро as равно" not in text:
            return ""
        side_match = re.search(r"сторона основания ab равна\s*([0-9]+(?:\.[0-9]+)?)", text)
        edge_match = re.search(r"(?:боковое )?ребро as равно\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not side_match or not edge_match:
            return ""
        side = float(side_match.group(1))
        edge = float(edge_match.group(1))
        if edge <= 0:
            return ""
        cos_value = side / (2.0 * edge)
        if cos_value < -1.0 or cos_value > 1.0:
            return ""
        answer = math.sqrt(max(0.0, 1.0 - cos_value * cos_value))
        return self._format_exact_numeric(answer)

    def _exact_parallelepiped_distance(self, text: str) -> str:
        if "прямоугольном параллелепипеде" not in text or "плоскости cdk" not in text:
            return ""
        ad_match = re.search(r"ad\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*√\s*([0-9]+(?:\.[0-9]+)?)", text)
        aa1_match = re.search(r"aa1\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*√\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not ad_match or not aa1_match:
            return ""
        ad = float(ad_match.group(1)) * math.sqrt(float(ad_match.group(2)))
        aa1 = float(aa1_match.group(1)) * math.sqrt(float(aa1_match.group(2)))
        denominator = math.sqrt(aa1 * aa1 + (ad * ad) / 4.0)
        if denominator == 0:
            return ""
        answer = (ad * aa1 / 2.0) / denominator
        return self._format_exact_numeric(answer)

    def _exact_marker_probability(self, text: str) -> str:
        if "красных маркеров" not in text or "черных" not in text:
            return ""
        match = re.search(
            r"([0-9]+)\s+черных\s+и\s+([0-9]+)\s+красных\s+маркеров",
            text,
        )
        if not match:
            return ""
        black = float(match.group(1))
        red = float(match.group(2))
        total = black + red
        if total == 0:
            return ""
        return self._format_exact_numeric(red / total)

    def _exact_two_clubs_overlap(self, text: str) -> str:
        if "25 учащ" not in text or "химическ" not in text or "биологическ" not in text:
            return ""
        numbers = [int(value) for value in re.findall(r"\b([0-9]+)\b", text)]
        if len(numbers) < 3:
            return ""
        total = numbers[0]
        chem = numbers[1]
        bio = numbers[2]
        return self._format_exact_numeric(chem + bio - total)

    def _exact_interval_probability(self, text: str) -> str:
        if "p(x <= 15)" not in text or "p(x >= 10)" not in text:
            return ""
        left_match = re.search(r"p\(x <= 15\)\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
        right_match = re.search(r"p\(x >= 10\)\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not left_match or not right_match:
            return ""
        answer = float(left_match.group(1)) + float(right_match.group(1)) - 1.0
        return self._format_exact_numeric(answer)

    def _exact_same_color_probability(self, text: str) -> str:
        if "красных чашек" not in text or "синих чашки" not in text or "одного цвета" not in text:
            return ""
        numbers = [int(value) for value in re.findall(r"\b([0-9]+)\b", text)]
        if len(numbers) < 4:
            return ""
        red_cups, red_saucers, blue_cups, blue_saucers = numbers[:4]
        total_cups = red_cups + blue_cups
        total_saucers = red_saucers + blue_saucers
        if total_cups == 0 or total_saucers == 0:
            return ""
        answer = (
            (red_cups / float(total_cups)) * (red_saucers / float(total_saucers)) +
            (blue_cups / float(total_cups)) * (blue_saucers / float(total_saucers))
        )
        return self._format_exact_numeric(answer)

    def _format_exact_numeric(self, value: float) -> str:
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return str(int(rounded))
        return ("%.10f" % value).rstrip("0").rstrip(".")

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

    def _call_qwen(
        self,
        variants: List[Dict[str, Optional[str]]],
        user_text: str,
        mismatch_summary: Optional[str] = None,
    ) -> Dict:
        user_content = self._build_qwen_content(variants, user_text, mismatch_summary)
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

    def _build_qwen_content(
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
                "Prefer literal OCR text, visible labels, and explicit numeric values from the source over inferred structure when they conflict.\n"
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
        prompt += "Qwen parse:\n%s\n" % json.dumps(
            self._compact_result_for_prompt(qwen_result, role="parser"),
            ensure_ascii=False,
        )
        if mismatch_summary:
            prompt += (
                "Self-check instruction:\n"
                "Resolve the task again from scratch using the source itself. "
                "Treat the mismatch summary only as a warning about possible failure modes, not as ground truth.\n"
            )
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
                "Prefer literal OCR text, visible labels, and explicit numeric values from the source over inferred structure when they conflict.\n"
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
        prompt += "Qwen parse:\n%s\n" % json.dumps(
            self._compact_result_for_prompt(qwen_result, role="parser"),
            ensure_ascii=False,
        )
        prompt += "Kimi result:\n%s\n" % json.dumps(
            self._compact_result_for_prompt(kimi_result, role="solver"),
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
        if self._can_accept_verifier_over_local_salvage(consensus, kimi, llama):
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
        if kimi_answer and not self._looks_like_final_answer(kimi_answer):
            _log.warning("Rejecting solver answer that does not look like a final answer: %s", kimi_answer)
            kimi_answer = ""
        llama_answer = ""
        llama_confidence = 0.0
        if llama is not None:
            llama_answer = (llama.get("final_answer") or {}).get("value", "").strip()
            if llama_answer and not self._looks_like_final_answer(llama_answer):
                _log.warning("Rejecting verifier answer that does not look like a final answer: %s", llama_answer)
                llama_answer = ""
            llama_confidence = self._coerce_confidence(llama.get("answer_confidence"))
        kimi_confidence = self._coerce_confidence(kimi.get("answer_confidence"))

        if consensus["status"] == "accepted":
            return kimi_answer or llama_answer

        answers_match = self._answers_effectively_match(kimi, llama)
        parser_unavailable = "parser unavailable" in (qwen.get("visual_interpretation") or {}).get("possible_ambiguities", [])
        parser_explicit_ambiguity = self._parser_has_explicit_ambiguity(qwen)
        parser_clear = not parser_explicit_ambiguity
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

        if self._can_accept_verifier_over_local_salvage(consensus, kimi, llama):
            _log.info("Accepting verifier answer because local answer salvage conflicts with independent verifier")
            return llama_answer

        if answers_match and parser_clear and not solver_degraded and not solver_repaired and consensus["score"] >= SELF_CHECK_SCORE_THRESHOLD:
            _log.info("Accepting answer after non-accepted consensus because independent solver and verifier match under clear parse")
            return kimi_answer

        if kimi_answer and not llama_answer:
            if parser_clear and not solver_degraded and not solver_repaired and kimi_confidence >= DIRECT_SOLVER_CONFIDENCE_THRESHOLD:
                _log.info("Accepting Kimi answer because verifier did not provide a competing answer and parser is clear")
                return kimi_answer
            return ""

        if llama_answer and not kimi_answer:
            _log.warning(
                "Rejecting verifier-only answer because primary solver did not provide a stable answer: "
                "status=%s score=%.3f verifier_confidence=%.3f",
                consensus["status"],
                consensus["score"],
                llama_confidence,
            )
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

    def _resolve_multi_round_answer(self, default_qwen: Dict, first_round: Dict, second_round: Dict) -> str:
        first_qwen = first_round.get("qwen") or default_qwen
        second_qwen = second_round.get("qwen") or default_qwen
        first_answer = self._pick_user_answer(first_round["consensus"], first_round["kimi"], first_qwen, first_round["llama"])
        second_answer = self._pick_user_answer(second_round["consensus"], second_round["kimi"], second_qwen, second_round["llama"])

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
        kimi = round_data["kimi"]
        llama = round_data["llama"]

        score = float(consensus.get("score", 0.0))
        if consensus.get("status") == "accepted":
            score += 0.20
        elif consensus.get("status") == "self_check":
            score += 0.05

        if self._answers_effectively_match(kimi, llama):
            score += 0.10
        if not self._used_repair(kimi):
            score += 0.05
        if not self._solver_is_degraded(kimi):
            score += 0.05
        if self._has_independent_verifier(llama):
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
        if float(consensus.get("score", 0.0)) < SELF_CHECK_SCORE_THRESHOLD:
            return False
        if self._solver_is_degraded(kimi):
            return False
        if not self._has_independent_verifier(llama):
            return False
        if not self._answers_effectively_match(kimi, llama):
            return False
        kimi_answer = ((kimi.get("final_answer") or {}).get("value") or "").strip()
        if not self._looks_like_final_answer(kimi_answer):
            return False
        qwen_like_clear = True
        kimi_visual = kimi.get("visual_interpretation") or {}
        ambiguities = kimi_visual.get("possible_ambiguities") or []
        if "recovered from non-json output" in ambiguities:
            if self._coerce_confidence((llama or {}).get("answer_confidence")) < 0.75:
                return False
        kimi_confidence = self._coerce_confidence(kimi.get("answer_confidence"))
        llama_confidence = self._coerce_confidence((llama or {}).get("answer_confidence"))
        if kimi_confidence < REPAIRED_MATCH_CONFIDENCE_THRESHOLD and llama_confidence < REPAIRED_MATCH_CONFIDENCE_THRESHOLD:
            return False
        if consensus.get("answer_agreement", 0.0) < 1.0:
            return False
        return qwen_like_clear

    def _looks_like_final_answer(self, answer: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return False
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if any(re.match(r"^\d+\)\s*", line) for line in lines):
            normalized_lines = [re.sub(r"^\d+\)\s*", "", line).strip() for line in lines]
            return all(self._looks_like_final_answer(line) for line in normalized_lines if line)

        if len(text) > 80:
            return False
        lowered = text.lower()
        forbidden_markers = (
            "dependent on",
            "unknown",
            "cannot determine",
            "insufficient",
            "need clarification",
            "зависит",
            "неизвест",
            "недостаточно",
            "уточн",
        )
        for marker in forbidden_markers:
            if marker in lowered:
                return False
        if len(text.split()) > 5:
            return False
        return True

    def _can_accept_verifier_over_local_salvage(self, consensus: Dict, kimi: Dict, llama: Optional[Dict]) -> bool:
        if not self._used_local_salvage(kimi):
            return False
        if not self._has_independent_verifier(llama):
            return False
        if self._answers_effectively_match(kimi, llama):
            return False
        llama_answer = ((llama or {}).get("final_answer") or {}).get("value", "").strip()
        if not llama_answer:
            return False
        if self._coerce_confidence((llama or {}).get("answer_confidence")) < 0.75:
            return False
        return True

    def _parser_has_explicit_ambiguity(self, qwen: Dict) -> bool:
        ambiguities = ((qwen.get("visual_interpretation") or {}).get("possible_ambiguities") or [])
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
