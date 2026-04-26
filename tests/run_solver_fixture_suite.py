from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Config
from app.geometry_solver import GeometryPhotoSolver


FIXTURE_DIR = Path("/tmp/mcko_fixture_suite")


def _load_fonts() -> Tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return (
                ImageFont.truetype(path, 32),
                ImageFont.truetype(path, 24),
            )
        except Exception:
            continue
    default = ImageFont.load_default()
    return default, default


FONT_BIG, FONT = _load_fonts()


def _canvas() -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (1280, 760), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 35, 1110, 70], outline="black", width=2)
    draw.text(
        (120, 38),
        "Ответ на задание запишите в виде целого числа или конечной десятичной дроби.",
        fill="black",
        font=FONT_BIG,
    )
    return img, draw


def _draw_wrapped_text(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, max_width: int) -> int:
    words = text.split()
    line = ""
    for word in words:
        candidate = (line + " " + word).strip()
        if line and draw.textlength(candidate, font=FONT) > max_width:
            draw.text((x, y), line, fill="black", font=FONT)
            y += 40
            line = word
        else:
            line = candidate
    if line:
        draw.text((x, y), line, fill="black", font=FONT)
        y += 40
    return y


def _draw_answer_box(draw: ImageDraw.ImageDraw, y: int) -> None:
    draw.rectangle([40, y, 860, y + 70], fill=(228, 244, 248), outline=None)
    draw.text((55, y + 18), "Ответ:", fill="black", font=FONT)
    draw.rectangle([135, y + 15, 235, y + 52], outline="black", width=2)


def _save(img: Image.Image, name: str) -> Path:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / name
    img.save(path)
    return path


def make_fixture_1() -> Path:
    img, draw = _canvas()
    text = (
        "Известно, что в треугольнике ABC стороны AB и BC равны. "
        "Внешний угол при вершине B равен 138°. Найдите угол C. "
        "Ответ дайте в градусах."
    )
    y = _draw_wrapped_text(draw, text, 30, 110, 760)
    _draw_answer_box(draw, y + 10)
    A, B, C = (860, 270), (1070, 270), (910, 150)
    draw.line([A, B], fill="black", width=3)
    draw.line([B, C], fill="black", width=3)
    draw.line([C, A], fill="black", width=3)
    draw.line([B, (1150, 270)], fill="black", width=2)
    draw.line([(958, 226), (973, 237)], fill="black", width=3)
    draw.line([(960, 270), (960, 290)], fill="black", width=3)
    for point, label in ((A, "A"), (B, "B"), (C, "C")):
        draw.text((point[0] - 10, point[1] + 5), label, fill="black", font=FONT)
    return _save(img, "fixture1.png")


def make_fixture_2() -> Path:
    img, draw = _canvas()
    text = (
        "В ромбе ABCD диагонали пересекаются в точке O. "
        "Окружность радиусом 4 вписана в ромб и касается стороны AD в точке E. "
        "Найдите площадь ромба, если известно, что DE = 2."
    )
    y = _draw_wrapped_text(draw, text, 30, 110, 760)
    _draw_answer_box(draw, y + 10)
    A, B, C, D, O, E = (980, 670), (885, 410), (980, 150), (1080, 410), (980, 410), (1045, 495)
    for p, q in ((A, B), (B, C), (C, D), (D, A), (A, C), (B, D), (O, E)):
        draw.line([p, q], fill="black", width=3)
    draw.ellipse([890, 330, 1070, 510], outline="black", width=3)
    for point, label in ((A, "A"), (B, "B"), (C, "C"), (D, "D"), (O, "O"), (E, "E")):
        draw.text((point[0] + 8, point[1] + 8), label, fill="black", font=FONT)
    return _save(img, "fixture2.png")


def make_fixture_3() -> Path:
    img, draw = _canvas()
    text = (
        "В правильной четырёхугольной пирамиде SABCD сторона основания AB равна 18, "
        "а боковое ребро AS равно 15. Найдите синус угла между прямыми AB и SD."
    )
    y = _draw_wrapped_text(draw, text, 30, 110, 820)
    _draw_answer_box(draw, y + 10)
    A, D, C, B, S = (860, 360), (1050, 360), (1120, 250), (950, 250), (980, 145)
    for p, q in ((A, D), (D, C), (C, S), (S, A), (S, D)):
        draw.line([p, q], fill="black", width=3)
    draw.line([B, C], fill="black", width=2)
    draw.line([A, B], fill="black", width=2, joint="curve")
    draw.line([S, B], fill="black", width=2)
    for point, label in ((A, "A"), (B, "B"), (C, "C"), (D, "D"), (S, "S")):
        draw.text((point[0] - 10, point[1] + 5), label, fill="black", font=FONT)
    return _save(img, "fixture3.png")


def make_fixture_4() -> Path:
    img, draw = _canvas()
    text = (
        "В прямоугольном параллелепипеде ABCDA1B1C1D1 точка K — середина ребра B1C1. "
        "Известно, что AD = 4√11, AA1 = 3√22. Найдите расстояние от точки A1 до плоскости CDK."
    )
    y = _draw_wrapped_text(draw, text, 30, 110, 820)
    _draw_answer_box(draw, y + 10)
    A, D, C, B = (870, 500), (1030, 500), (1080, 450), (920, 450)
    A1, D1, C1, B1 = (870, 260), (1030, 260), (1080, 210), (920, 210)
    K = (1005, 210)
    for p, q in ((A, D), (D, C), (C, B), (B, A), (A1, D1), (D1, C1), (C1, B1), (B1, A1), (A, A1), (D, D1), (C, C1), (B, B1)):
        draw.line([p, q], fill="black", width=2)
    draw.line([B, B1], fill="black", width=2)
    draw.ellipse([K[0] - 4, K[1] - 4, K[0] + 4, K[1] + 4], fill="black")
    draw.text((K[0] - 5, K[1] - 35), "K", fill="black", font=FONT)
    for point, label in ((A, "A"), (B, "B"), (C, "C"), (D, "D"), (A1, "A1"), (B1, "B1"), (C1, "C1"), (D1, "D1")):
        draw.text((point[0] - 20, point[1] - 10), label, fill="black", font=FONT)
    return _save(img, "fixture4.png")


def make_text_fixture(name: str, text: str) -> Path:
    img, draw = _canvas()
    y = _draw_wrapped_text(draw, text, 30, 110, 1080)
    _draw_answer_box(draw, y + 10)
    return _save(img, name)


def build_fixtures() -> List[Dict]:
    return [
        {"id": 1, "path": make_fixture_1(), "expected": "69"},
        {"id": 2, "path": make_fixture_2(), "expected": "80"},
        {"id": 3, "path": make_fixture_3(), "expected": "0.8"},
        {"id": 4, "path": make_fixture_4(), "expected": "6"},
        {"id": 5, "path": make_text_fixture("fixture5.png", "Из коробки, в которой лежат 15 чёрных и 5 красных маркеров, достают один случайный маркер. Найдите вероятность того, что он окажется красным."), "expected": "0.25"},
        {"id": 6, "path": make_text_fixture("fixture6.png", "Каждый из 25 учащихся в классе посещает хотя бы один из двух кружков. Известно, что 10 человек занимаются в химическом кружке, а 18 — в биологическом. Сколько учащихся посещают оба кружка?"), "expected": "3"},
        {"id": 7, "path": make_text_fixture("fixture7.png", "В некотором случайном эксперименте рассматривается случайная величина X. Известно, что P(X ≤ 15) = 0,77 и P(X ≥ 10) = 0,58. Найдите вероятность события (10 ≤ X ≤ 15)."), "expected": "0.35"},
        {"id": 8, "path": make_text_fixture("fixture8.png", "На полке стоят 6 красных чашек и 6 красных блюдец, 4 синих чашки и 4 синих блюдца. Случайным образом выбирают одно блюдце и одну чашку. Какова вероятность того, что они окажутся одного цвета?"), "expected": "0.52"},
    ]


def build_user_regression_fixtures() -> List[Dict]:
    return [
        {
            "id": 101,
            "path": make_text_fixture(
                "user_fixture_101.png",
                "Углы B и C треугольника ABC равны соответственно 67° и 83°. "
                "Найдите BC, если радиус окружности, описанной около треугольника ABC, равен 16.",
            ),
            "expected": "16",
        },
        {
            "id": 102,
            "path": make_text_fixture(
                "user_fixture_102.png",
                "Биссектрисы углов A и B при боковой стороне AB трапеции ABCD пересекаются в точке F. "
                "Найдите AB, если AF = 20, BF = 15.",
            ),
            "expected": "25",
        },
        {
            "id": 103,
            "path": make_text_fixture(
                "user_fixture_103.png",
                "Дана четырёхугольная пирамида SABCD с вершиной S. Основание ABCD является прямоугольной "
                "трапецией с прямыми углами A и D. Отрезок SD перпендикулярен плоскости основания. "
                "Выберите из предложенного списка пары скрещивающихся прямых. "
                "1) прямые AB и CD "
                "2) прямые SA и DC "
                "3) прямые AC и SB "
                "4) прямые BD и AC "
                "В ответе запишите номера выбранных пар прямых без пробелов, запятых и других дополнительных символов.",
            ),
            "expected": "23",
        },
        {
            "id": 104,
            "path": make_text_fixture(
                "user_fixture_104.png",
                "Дана прямая четырёхугольная призма ABCDA1B1C1D1. Выберите из предложенного списка пары прямых, "
                "которые лежат в одной плоскости. "
                "1) прямые CD и C1D1 "
                "2) прямые BC и AD "
                "3) прямые AB и CC1 "
                "4) прямые AB и CD "
                "В ответе запишите номера выбранных пар прямых без пробелов, запятых и других дополнительных символов.",
            ),
            "expected": "124",
        },
        {
            "id": 105,
            "path": make_text_fixture(
                "user_fixture_105.png",
                "В основании прямой призмы лежит прямоугольный треугольник с катетами 8 и 15. "
                "Найдите расстояние между гипотенузой основания и скрещивающимся с ней ребром.",
            ),
            "expected": "120/17",
        },
        {
            "id": 106,
            "path": make_text_fixture(
                "user_fixture_106.png",
                "В кубе ABCDA1B1C1D1, ребро которого равно a. "
                "Найдите расстояние от вершины B до плоскости AB1C.",
            ),
            "expected": "a/sqrt(3)",
        },
        {
            "id": 107,
            "path": make_fixture_1(),
            "expected": "69",
        },
        {
            "id": 108,
            "path": make_fixture_2(),
            "expected": "80",
        },
        {
            "id": 109,
            "path": make_fixture_3(),
            "expected": "0.8",
        },
        {
            "id": 110,
            "path": make_fixture_4(),
            "expected": "6",
        },
    ]


def _path_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return "data:image/png;base64,%s" % encoded


def _normalize_answer(text: str) -> str:
    answer = (text or "").strip()
    answer = answer.replace("1) ", "").replace("1)\n", "").strip()
    return answer


def _canonical_math_answer(text: str) -> str:
    answer = _normalize_answer(text)
    if not answer:
        return ""
    lowered = answer.lower()
    lowered = lowered.replace(" ", "")
    lowered = lowered.replace("\\left", "").replace("\\right", "")
    lowered = lowered.strip("$")
    lowered = lowered.replace("\\sqrt{3}", "sqrt(3)")
    lowered = lowered.replace("√3", "sqrt(3)")
    lowered = lowered.replace("{", "").replace("}", "")
    lowered = lowered.replace("\\fracasqrt(3)3", "a/sqrt(3)")
    lowered = lowered.replace("\\fraca*sqrt(3)3", "a/sqrt(3)")
    lowered = lowered.replace("(a*sqrt(3))/3", "a/sqrt(3)")
    lowered = lowered.replace("a*sqrt(3)/3", "a/sqrt(3)")
    lowered = lowered.replace("asqrt(3)/3", "a/sqrt(3)")
    lowered = lowered.replace("\\fracasqrt(3)3", "a/sqrt(3)")
    return lowered


def _select_fixtures(argv: List[str]) -> List[Dict]:
    if len(argv) > 1 and argv[1] == "user-regression":
        return build_user_regression_fixtures()
    return build_fixtures()


def main() -> None:
    cfg = Config()
    cfg.load()
    solver = GeometryPhotoSolver(
        cfg.api_key,
        model=cfg.model,
    )
    fixtures = _select_fixtures(sys.argv)
    results = []
    for fixture in fixtures:
        blocks = [
            {"type": "image_url", "image_url": {"url": _path_to_data_url(fixture["path"])}},
        ]
        started = time.monotonic()
        result = solver.solve_content_blocks(blocks)
        elapsed = round(time.monotonic() - started, 2)
        normalized = _normalize_answer(result)
        passed = _canonical_math_answer(result) == _canonical_math_answer(fixture["expected"])
        payload = {
            "id": fixture["id"],
            "expected": fixture["expected"],
            "result": result,
            "normalized": normalized,
            "passed": passed,
            "seconds": elapsed,
            "path": str(fixture["path"]),
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        results.append(payload)

    summary = {
        "passed": sum(1 for item in results if item["passed"]),
        "total": len(results),
        "avg_seconds": round(sum(item["seconds"] for item in results) / float(len(results)), 2),
    }
    print("SUMMARY %s" % json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
