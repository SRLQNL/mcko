import unittest

from app.geometry_solver import GeometryPhotoSolver


class GeometrySolverConsensusTests(unittest.TestCase):
    def setUp(self):
        self.solver = GeometryPhotoSolver("test-key")

    def _make_result(
        self,
        answer="69",
        answer_confidence=0.9,
        target="find x",
        givens=None,
        relations=None,
        ambiguities=None,
        needs_clarification=False,
        solver_origin="model",
        used_repair=False,
        mirrors_solver=False,
    ):
        if givens is None:
            givens = [{"statement": "ab = cd"}]
        if relations is None:
            relations = [{"type": "parallel", "subject": "ab", "object": "cd"}]
        if ambiguities is None:
            ambiguities = []
        return {
            "task_type": "mixed_task",
            "ocr_text": "",
            "normalized_problem_text": "",
            "diagram_entities": [],
            "diagram_relations": relations,
            "givens": givens,
            "target": {"statement": target},
            "visual_interpretation": {
                "summary": "",
                "confidence": 0.9,
                "possible_ambiguities": ambiguities,
            },
            "reasoning_summary": [],
            "solution_steps": [],
            "final_answer": {"value": answer, "format": "text"},
            "answer_confidence": answer_confidence,
            "consistency_checks": [],
            "needs_clarification": needs_clarification,
            "_request_meta": {
                "model": "test-model",
                "used_repair": used_repair,
                "repair_model": "test-model" if used_repair else "",
                "raw_text_chars": 100,
            },
            "_solver_origin": solver_origin,
            "_mirrors_solver": mirrors_solver,
        }

    def test_mirrored_verifier_does_not_create_false_accept(self):
        qwen = self._make_result(answer="", answer_confidence=0.0)
        kimi = self._make_result(answer="0.6667", solver_origin="degraded_solver", ambiguities=["solver used verifier model"])
        llama = self._make_result(answer="0.6667", solver_origin="verifier_fallback", mirrors_solver=True)

        consensus = self.solver._compare_results(kimi, qwen, llama)
        final_answer = self.solver._pick_user_answer(consensus, kimi, qwen, llama)

        self.assertNotEqual(consensus["status"], "accepted")
        self.assertEqual(final_answer, "")

    def test_ambiguous_match_requires_clear_parse_and_direct_solver(self):
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=["blurred text"], needs_clarification=True)
        kimi = self._make_result(answer="80")
        llama = self._make_result(answer="80")

        consensus = self.solver._compare_results(kimi, qwen, llama)
        final_answer = self.solver._pick_user_answer(consensus, kimi, qwen, llama)

        self.assertEqual(final_answer, "")

    def test_direct_high_confidence_solver_can_survive_missing_verifier(self):
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        kimi = self._make_result(answer="42", answer_confidence=0.95)
        llama = self._make_result(answer="", solver_origin="verifier_fallback", ambiguities=["verifier unavailable"])

        consensus = self.solver._compare_results(kimi, qwen, llama)
        final_answer = self.solver._pick_user_answer(consensus, kimi, qwen, llama)

        self.assertEqual(final_answer, "42")

    def test_answer_salvage_recovers_numbered_output_without_remote_repair(self):
        salvaged = self.solver._try_salvage_answer_only(
            "I will solve both tasks carefully.\n\n1) 69\n2) 42\n"
        )

        self.assertIsNotNone(salvaged)
        self.assertEqual(salvaged["final_answer"]["value"], "1) 69\n2) 42")
        self.assertIn("recovered from non-json output", salvaged["visual_interpretation"]["possible_ambiguities"])

    def test_clean_accepted_case_does_not_trigger_extra_self_check(self):
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        kimi = self._make_result(answer="69", answer_confidence=0.95)
        llama = self._make_result(answer="69", answer_confidence=0.93)

        consensus = self.solver._compare_results(kimi, qwen, llama)

        self.assertEqual(consensus["status"], "accepted")
        self.assertFalse(self.solver._should_run_self_check(consensus, kimi, llama))

    def test_low_score_matching_answers_still_trigger_self_check(self):
        qwen = self._make_result(
            answer="",
            answer_confidence=0.0,
            givens=[{"statement": "different given"}],
            relations=[],
            ambiguities=["unclear diagram"],
            needs_clarification=True,
        )
        kimi = self._make_result(answer="80", answer_confidence=0.92)
        llama = self._make_result(answer="80", answer_confidence=0.90, givens=[{"statement": "different given"}], relations=[])

        consensus = self.solver._compare_results(kimi, qwen, llama)

        self.assertTrue(self.solver._should_run_self_check(consensus, kimi, llama))

    def test_normalize_result_handles_list_visual_and_list_answer(self):
        normalized = self.solver._normalize_result(
            {
                "target": ["find area", {"statement": "of the circle"}],
                "visual_interpretation": ["repaired visual note", "second note"],
                "final_answer": ["1) 12", "2) 13"],
                "consistency_checks": "checked",
                "reasoning_summary": "short",
                "solution_steps": {"step": "value"},
            },
            role="solver",
        )

        self.assertEqual(normalized["target"]["statement"], "find area | of the circle")
        self.assertEqual(normalized["visual_interpretation"]["summary"], "repaired visual note; second note")
        self.assertEqual(normalized["final_answer"]["value"], "1) 12\n2) 13")
        self.assertEqual(normalized["consistency_checks"], ["checked"])

    def test_normalize_result_survives_common_repair_shape_variants(self):
        variants = [
            {
                "target": "find x",
                "visual_interpretation": "clear diagram",
                "final_answer": "42",
                "givens": "ab = cd",
                "diagram_relations": "parallel(ab,cd)",
            },
            {
                "target": ["find area", {"statement": "of figure"}],
                "visual_interpretation": ["blurred text", "cropped lower figure"],
                "final_answer": ["1) 12", "2) 13"],
                "givens": [{"a": 1}, {"statement": "b = 2"}],
                "diagram_relations": [{"type": "touches", "subject": "circle", "object": "side"}],
            },
            {
                "target": {"value": "radius"},
                "visual_interpretation": {"summary": "repaired", "confidence": "0.7", "possible_ambiguities": "none"},
                "final_answer": {"value": 3.14, "format": "text"},
                "givens": {"r": 4},
                "diagram_relations": [{"broken": "shape"}],
            },
            {
                "target": None,
                "visual_interpretation": 123,
                "final_answer": 99,
                "givens": None,
                "diagram_relations": None,
            },
        ]

        for raw in variants:
            normalized = self.solver._normalize_result(raw, role="solver")
            self.assertIsInstance(normalized["target"], dict)
            self.assertIn("statement", normalized["target"])
            self.assertIsInstance(normalized["visual_interpretation"], dict)
            self.assertIsInstance(normalized["final_answer"], dict)
            self.assertIsInstance(normalized["givens"], list)
            self.assertIsInstance(normalized["diagram_relations"], list)

    def test_compare_results_survives_repaired_shape_variants(self):
        qwen = self.solver._normalize_result(
            {
                "target": {"statement": "find area"},
                "givens": [{"statement": "side=4"}],
                "diagram_relations": [{"type": "parallel", "subject": "ab", "object": "cd"}],
                "visual_interpretation": {"summary": "clear", "confidence": 0.9, "possible_ambiguities": []},
                "final_answer": {"value": "", "format": "text"},
            },
            role="parser",
        )
        repaired_solver_payloads = [
            {
                "target": ["find area", "of figure"],
                "givens": {"side": 4},
                "diagram_relations": "parallel(ab,cd)",
                "visual_interpretation": ["repaired visual"],
                "final_answer": ["16"],
            },
            {
                "target": {"goal": "find area"},
                "givens": "side=4",
                "diagram_relations": [{"type": "parallel", "subject": "ab", "object": "cd"}],
                "visual_interpretation": 0,
                "final_answer": 16,
            },
        ]

        for raw in repaired_solver_payloads:
            kimi = self.solver._normalize_result(raw, role="solver")
            llama = self.solver._normalize_result(raw, role="verifier")
            consensus = self.solver._compare_results(kimi, qwen, llama)
            self.assertIn(consensus["status"], ("accepted", "self_check", "ambiguous"))

    def test_kimi_repair_uses_llama_repair_only(self):
        repair_models = self.solver._repair_models_for_source(self.solver.kimi_model)
        self.assertEqual(repair_models, [self.solver.llama_model])

    def test_repaired_matching_answers_can_skip_expensive_self_check(self):
        kimi = self._make_result(answer="80", answer_confidence=0.82, used_repair=True)
        llama = self._make_result(answer="80", answer_confidence=0.78)
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])

        consensus = self.solver._compare_results(kimi, qwen, llama)

        self.assertFalse(self.solver._should_run_self_check(consensus, kimi, llama))
        self.assertEqual(self.solver._pick_user_answer(consensus, kimi, qwen, llama), "80")

    def test_local_salvage_match_with_independent_verifier_is_accepted(self):
        kimi = self._make_result(answer="0.8", answer_confidence=0.70, used_repair=True)
        kimi["_request_meta"]["repair_model"] = "local_answer_salvage"
        llama = self._make_result(answer="0.8", answer_confidence=0.84)
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])

        consensus = self.solver._compare_results(kimi, qwen, llama)

        self.assertEqual(self.solver._pick_user_answer(consensus, kimi, qwen, llama), "0.8")

    def test_verifier_can_override_local_salvage_mismatch(self):
        kimi = self._make_result(answer="8", answer_confidence=0.7, used_repair=True)
        kimi["_request_meta"]["repair_model"] = "local_answer_salvage"
        llama = self._make_result(answer="80", answer_confidence=0.9)
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])

        consensus = self.solver._compare_results(kimi, qwen, llama)

        self.assertFalse(self.solver._should_run_self_check(consensus, kimi, llama))
        self.assertEqual(self.solver._pick_user_answer(consensus, kimi, qwen, llama), "80")

    def test_parser_needs_clarification_without_explicit_ambiguity_does_not_block_match(self):
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=[], needs_clarification=True)
        kimi = self._make_result(answer="0.8", answer_confidence=0.88)
        llama = self._make_result(answer="0.8", answer_confidence=0.87)

        consensus = self.solver._compare_results(kimi, qwen, llama)

        self.assertEqual(self.solver._pick_user_answer(consensus, kimi, qwen, llama), "0.8")

    def test_verifier_only_answer_is_rejected_when_solver_has_no_answer(self):
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=[], needs_clarification=True)
        kimi = self._make_result(answer="", answer_confidence=0.0, used_repair=True)
        llama = self._make_result(answer="6", answer_confidence=0.91)

        consensus = self.solver._compare_results(kimi, qwen, llama)

        self.assertEqual(self.solver._pick_user_answer(consensus, kimi, qwen, llama), "")

    def test_non_final_explanatory_phrase_is_rejected(self):
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        kimi = self._make_result(answer="dependent on unknown dimension AB", answer_confidence=0.9, used_repair=True)
        llama = self._make_result(answer="dependent on unknown dimension AB", answer_confidence=0.9)

        consensus = self.solver._compare_results(kimi, qwen, llama)

        self.assertEqual(self.solver._pick_user_answer(consensus, kimi, qwen, llama), "")

    def test_multi_round_selection_keeps_primary_when_self_check_fails(self):
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        primary_kimi = self._make_result(answer="80", answer_confidence=0.93)
        primary_llama = self._make_result(answer="80", answer_confidence=0.92)
        self_check_kimi = self._make_result(answer="", answer_confidence=0.0, used_repair=True)
        self_check_llama = self._make_result(answer="", answer_confidence=0.0)

        first_round = {
            "consensus": self.solver._compare_results(primary_kimi, qwen, primary_llama),
            "kimi": primary_kimi,
            "llama": primary_llama,
        }
        second_round = {
            "consensus": self.solver._compare_results(self_check_kimi, qwen, self_check_llama),
            "kimi": self_check_kimi,
            "llama": self_check_llama,
        }

        self.assertEqual(self.solver._resolve_multi_round_answer(qwen, first_round, second_round), "80")

    def test_multi_round_selection_uses_self_check_when_primary_has_no_safe_answer(self):
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        primary_kimi = self._make_result(answer="", answer_confidence=0.0, used_repair=True)
        primary_llama = self._make_result(answer="", answer_confidence=0.0)
        self_check_kimi = self._make_result(answer="69", answer_confidence=0.91)
        self_check_llama = self._make_result(answer="69", answer_confidence=0.90)

        first_round = {
            "consensus": self.solver._compare_results(primary_kimi, qwen, primary_llama),
            "kimi": primary_kimi,
            "llama": primary_llama,
        }
        second_round = {
            "consensus": self.solver._compare_results(self_check_kimi, qwen, self_check_llama),
            "kimi": self_check_kimi,
            "llama": self_check_llama,
        }

        self.assertEqual(self.solver._resolve_multi_round_answer(qwen, first_round, second_round), "69")

    def test_multi_round_selection_rejects_close_conflicting_answers(self):
        qwen = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        primary_kimi = self._make_result(answer="12", answer_confidence=0.91)
        primary_llama = self._make_result(answer="12", answer_confidence=0.90)
        self_check_kimi = self._make_result(answer="6", answer_confidence=0.92)
        self_check_llama = self._make_result(answer="6", answer_confidence=0.91)

        first_round = {
            "consensus": self.solver._compare_results(primary_kimi, qwen, primary_llama),
            "kimi": primary_kimi,
            "llama": primary_llama,
        }
        second_round = {
            "consensus": self.solver._compare_results(self_check_kimi, qwen, self_check_llama),
            "kimi": self_check_kimi,
            "llama": self_check_llama,
        }

        self.assertEqual(self.solver._resolve_multi_round_answer(qwen, first_round, second_round), "")

    def test_compact_result_for_prompt_truncates_large_payloads(self):
        result = self._make_result(answer="80", answer_confidence=0.91)
        result["normalized_problem_text"] = "x" * 2000
        result["ocr_text"] = "y" * 2000
        result["givens"] = [{"statement": "g" * 200}] * 10
        result["diagram_relations"] = ["r" * 200] * 10
        result["visual_interpretation"] = {
            "summary": "s" * 500,
            "confidence": 0.8,
            "possible_ambiguities": ["a" * 100] * 10,
        }
        compact = self.solver._compact_result_for_prompt(result, role="solver")
        parser_compact = self.solver._compact_result_for_prompt(result, role="parser")

        self.assertLessEqual(len(compact["normalized_problem_text"]), 700)
        self.assertLessEqual(len(parser_compact["ocr_text"]), 600)
        self.assertLessEqual(len(compact["givens"]), 6)
        self.assertLessEqual(len(compact["diagram_relations"]), 6)
        self.assertLessEqual(len(compact["visual_interpretation"]["possible_ambiguities"]), 4)

    def test_exact_answer_engine_solves_regression_texts(self):
        cases = [
            ("Известно, что в треугольнике ABC стороны AB и BC равны. Внешний угол при вершине B равен 138°. Найдите угол C.", "69"),
            ("В ромбе ABCD диагонали пересекаются в точке O. Окружность радиусом 4 вписана в ромб и касается стороны AD в точке E. Найдите площадь ромба, если известно, что DE = 2.", "80"),
            ("В правильной четырёхугольной пирамиде SABCD сторона основания AB равна 18, а боковое ребро AS равно 15. Найдите синус угла между прямыми AB и SD.", "0.8"),
            ("В прямоугольном параллелепипеде ABCDA1B1C1D1 точка K — середина ребра B1C1. Известно, что AD = 4√11, AA1 = 3√22. Найдите расстояние от точки A1 до плоскости CDK.", "6"),
            ("Из коробки, в которой лежат 15 чёрных и 5 красных маркеров, достают один случайный маркер. Найдите вероятность того, что он окажется красным.", "0.25"),
            ("Каждый из 25 учащихся в классе посещает хотя бы один из двух кружков. Известно, что 10 человек занимаются в химическом кружке, а 18 — в биологическом. Сколько учащихся посещают оба кружка?", "3"),
            ("В некотором случайном эксперименте рассматривается случайная величина X. Известно, что P(X ≤ 15) = 0,77 и P(X ≥ 10) = 0,58. Найдите вероятность события (10 ≤ X ≤ 15).", "0.35"),
            ("На полке стоят 6 красных чашек и 6 красных блюдец, 4 синих чашки и 4 синих блюдца. Случайным образом выбирают одно блюдце и одну чашку. Какова вероятность того, что они окажутся одного цвета?", "0.52"),
        ]

        for text, expected in cases:
            self.assertEqual(self.solver._try_exact_answer_engine(text), expected)

    def test_loose_terminal_answer_salvage_extracts_simple_rhs(self):
        raw = (
            "Let me solve the problem carefully.\n"
            "We obtain side = 10.\n"
            "Area = 80.\n"
        )
        salvaged = self.solver._try_salvage_answer_only(raw)
        self.assertIsNotNone(salvaged)
        self.assertEqual(salvaged["final_answer"]["value"], "80")


if __name__ == "__main__":
    unittest.main()
