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
        parser = self._make_result(answer="", answer_confidence=0.0)
        solver = self._make_result(answer="0.6667", solver_origin="degraded_solver", ambiguities=["solver used verifier model"])
        verifier = self._make_result(answer="0.6667", solver_origin="verifier_fallback", mirrors_solver=True)

        consensus = self.solver._compare_results(solver, parser, verifier)
        final_answer = self.solver._pick_user_answer(consensus, solver, parser, verifier)

        self.assertNotEqual(consensus["status"], "accepted")
        self.assertEqual(final_answer, "")

    def test_ambiguous_match_requires_clear_parse_and_direct_solver(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=["blurred text"], needs_clarification=True)
        solver = self._make_result(answer="80")
        verifier = self._make_result(answer="80")

        consensus = self.solver._compare_results(solver, parser, verifier)
        final_answer = self.solver._pick_user_answer(consensus, solver, parser, verifier)

        self.assertEqual(final_answer, "")

    def test_direct_high_confidence_solver_can_survive_missing_verifier(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        solver = self._make_result(answer="42", answer_confidence=0.95)
        verifier = self._make_result(answer="", solver_origin="verifier_fallback", ambiguities=["verifier unavailable"])

        consensus = self.solver._compare_results(solver, parser, verifier)
        final_answer = self.solver._pick_user_answer(consensus, solver, parser, verifier)

        self.assertEqual(final_answer, "42")

    def test_repaired_solver_answer_can_survive_missing_verifier_when_parser_is_clear(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        solver = self._make_result(answer="25", answer_confidence=0.70, used_repair=True)
        verifier = self._make_result(answer="", solver_origin="verifier_fallback", ambiguities=["verifier unavailable"])

        consensus = self.solver._compare_results(solver, parser, verifier)
        final_answer = self.solver._pick_user_answer(consensus, solver, parser, verifier)

        self.assertEqual(final_answer, "25")

    def test_high_confidence_verifier_only_answer_can_be_accepted_when_solver_omits_answer(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        solver = self._make_result(answer="", answer_confidence=0.0)
        verifier = self._make_result(answer="16", answer_confidence=0.95)

        consensus = self.solver._compare_results(solver, parser, verifier)
        final_answer = self.solver._pick_user_answer(consensus, solver, parser, verifier)

        self.assertEqual(final_answer, "16")

    def test_verifier_only_answer_is_still_rejected_when_parser_is_ambiguous(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=["cropped figure"], needs_clarification=True)
        solver = self._make_result(answer="", answer_confidence=0.0)
        verifier = self._make_result(answer="16", answer_confidence=0.95)

        consensus = self.solver._compare_results(solver, parser, verifier)
        final_answer = self.solver._pick_user_answer(consensus, solver, parser, verifier)

        self.assertEqual(final_answer, "")

    def test_answer_salvage_recovers_numbered_output_without_remote_repair(self):
        salvaged = self.solver._try_salvage_answer_only(
            "I will solve both tasks carefully.\n\n1) 69\n2) 42\n"
        )

        self.assertIsNotNone(salvaged)
        self.assertEqual(salvaged["final_answer"]["value"], "1) 69\n2) 42")
        self.assertIn("recovered from non-json output", salvaged["visual_interpretation"]["possible_ambiguities"])

    def test_clean_accepted_case_does_not_trigger_extra_self_check(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        solver = self._make_result(answer="69", answer_confidence=0.95)
        verifier = self._make_result(answer="69", answer_confidence=0.93)

        consensus = self.solver._compare_results(solver, parser, verifier)

        self.assertEqual(consensus["status"], "accepted")
        self.assertFalse(self.solver._should_run_self_check(consensus, solver, verifier))

    def test_low_score_matching_answers_still_trigger_self_check(self):
        parser = self._make_result(
            answer="",
            answer_confidence=0.0,
            givens=[{"statement": "different given"}],
            relations=[],
            ambiguities=["unclear diagram"],
            needs_clarification=True,
        )
        solver = self._make_result(answer="80", answer_confidence=0.92)
        verifier = self._make_result(answer="80", answer_confidence=0.90, givens=[{"statement": "different given"}], relations=[])

        consensus = self.solver._compare_results(solver, parser, verifier)

        self.assertTrue(self.solver._should_run_self_check(consensus, solver, verifier))

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
        parser = self.solver._normalize_result(
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
            solver = self.solver._normalize_result(raw, role="solver")
            verifier = self.solver._normalize_result(raw, role="verifier")
            consensus = self.solver._compare_results(solver, parser, verifier)
            self.assertIn(consensus["status"], ("accepted", "self_check", "ambiguous"))

    def test_solver_repair_uses_verifier_repair_only(self):
        repair_models = self.solver._repair_models_for_source(self.solver.solver_model)
        self.assertEqual(repair_models, [self.solver.verifier_model])

    def test_repaired_matching_answers_can_skip_expensive_self_check(self):
        solver = self._make_result(answer="80", answer_confidence=0.82, used_repair=True)
        verifier = self._make_result(answer="80", answer_confidence=0.78)
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])

        consensus = self.solver._compare_results(solver, parser, verifier)

        self.assertFalse(self.solver._should_run_self_check(consensus, solver, verifier))
        self.assertEqual(self.solver._pick_user_answer(consensus, solver, parser, verifier), "80")

    def test_repaired_matching_answers_can_be_accepted_below_general_self_check_threshold(self):
        solver = self._make_result(answer="25", answer_confidence=0.82, used_repair=True)
        verifier = self._make_result(answer="25", answer_confidence=0.90)
        consensus = {
            "status": "self_check",
            "score": 0.177,
            "answer_agreement": 1.0,
            "reasons": ["low givens agreement", "solver JSON repaired"],
        }

        self.assertTrue(self.solver._can_accept_repaired_match_without_self_check(consensus, solver, verifier))

    def test_local_salvage_match_with_independent_verifier_is_accepted(self):
        solver = self._make_result(answer="0.8", answer_confidence=0.70, used_repair=True)
        solver["_request_meta"]["repair_model"] = "local_answer_salvage"
        verifier = self._make_result(answer="0.8", answer_confidence=0.84)
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])

        consensus = self.solver._compare_results(solver, parser, verifier)

        self.assertEqual(self.solver._pick_user_answer(consensus, solver, parser, verifier), "0.8")

    def test_verifier_can_override_local_salvage_mismatch(self):
        solver = self._make_result(answer="8", answer_confidence=0.7, used_repair=True)
        solver["_request_meta"]["repair_model"] = "local_answer_salvage"
        verifier = self._make_result(answer="80", answer_confidence=0.9)
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])

        consensus = self.solver._compare_results(solver, parser, verifier)

        self.assertFalse(self.solver._should_run_self_check(consensus, solver, verifier))
        self.assertEqual(self.solver._pick_user_answer(consensus, solver, parser, verifier), "80")

    def test_parser_needs_clarification_without_explicit_ambiguity_does_not_block_match(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[], needs_clarification=True)
        solver = self._make_result(answer="0.8", answer_confidence=0.88)
        verifier = self._make_result(answer="0.8", answer_confidence=0.87)

        consensus = self.solver._compare_results(solver, parser, verifier)

        self.assertEqual(self.solver._pick_user_answer(consensus, solver, parser, verifier), "0.8")

    def test_verifier_only_answer_is_rejected_when_solver_has_no_answer_and_parser_is_not_clear(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=["cropped figure"], needs_clarification=True)
        solver = self._make_result(answer="", answer_confidence=0.0, used_repair=True)
        verifier = self._make_result(answer="6", answer_confidence=0.91)

        consensus = self.solver._compare_results(solver, parser, verifier)

        self.assertEqual(self.solver._pick_user_answer(consensus, solver, parser, verifier), "")

    def test_non_option_option_arbiter_does_not_override_regular_consensus(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        solver = self._make_result(answer="6√2", answer_confidence=0.92, used_repair=True)
        verifier = self._make_result(answer="6√2", answer_confidence=0.93)
        option_arbiter = self._make_result(answer="6", answer_confidence=0.96)
        option_arbiter["_solver_origin"] = "option_arbiter"
        option_arbiter["_request_meta"]["used_repair"] = False
        option_arbiter["_request_meta"]["repair_model"] = ""
        consensus = {
            "status": "self_check",
            "score": 0.267,
            "answer_agreement": 1.0,
            "reasons": ["solver JSON repaired"],
        }

        self.assertEqual(self.solver._pick_user_answer(consensus, solver, parser, verifier, option_arbiter), "6√2")

    def test_non_option_option_arbiter_does_not_override_local_salvage_case(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        solver = self._make_result(answer="35", answer_confidence=0.92, used_repair=True)
        solver["_request_meta"]["repair_model"] = "local_answer_salvage"
        verifier = self._make_result(answer="25", answer_confidence=0.93)
        option_arbiter = self._make_result(answer="35", answer_confidence=0.96)
        option_arbiter["_solver_origin"] = "option_arbiter"
        consensus = {
            "status": "self_check",
            "score": 0.177,
            "answer_agreement": 0.0,
            "reasons": ["solver JSON repaired", "answer mismatch"],
        }

        self.assertEqual(self.solver._pick_user_answer(consensus, solver, parser, verifier, option_arbiter), "35")

    def test_option_arbiter_can_override_disagreeing_option_answers(self):
        parser = self._make_result(
            answer="",
            answer_confidence=0.0,
            target="write the numbers of the selected pairs",
        )
        parser["ocr_text"] = "1) ... 2) ... 3) ... 4) ... В ответе запишите номера выбранных пар"
        parser["normalized_problem_text"] = "select from the proposed list 1) ... 2) ... 3) ... 4) ..."
        solver = self._make_result(answer="12", answer_confidence=0.7, used_repair=True)
        verifier = self._make_result(answer="14", answer_confidence=0.9)
        option_arbiter = self._make_result(answer="124", answer_confidence=0.95)
        option_arbiter["_solver_origin"] = "option_arbiter"
        consensus = {
            "status": "self_check",
            "score": 0.0,
            "answer_agreement": 0.0,
            "reasons": ["answer mismatch"],
        }

        self.assertTrue(self.solver._is_option_selection_task(parser))
        self.assertTrue(self.solver._has_option_answer_disagreement(solver, verifier))
        self.assertEqual(self.solver._pick_user_answer(consensus, solver, parser, verifier, option_arbiter), "124")

    def test_non_final_explanatory_phrase_is_rejected(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        solver = self._make_result(answer="dependent on unknown dimension AB", answer_confidence=0.9, used_repair=True)
        verifier = self._make_result(answer="dependent on unknown dimension AB", answer_confidence=0.9)

        consensus = self.solver._compare_results(solver, parser, verifier)

        self.assertEqual(self.solver._pick_user_answer(consensus, solver, parser, verifier), "")

    def test_multi_round_selection_keeps_primary_when_self_check_fails(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        primary_solver = self._make_result(answer="80", answer_confidence=0.93)
        primary_verifier = self._make_result(answer="80", answer_confidence=0.92)
        self_check_solver = self._make_result(answer="", answer_confidence=0.0, used_repair=True)
        self_check_verifier = self._make_result(answer="", answer_confidence=0.0)

        first_round = {
            "consensus": self.solver._compare_results(primary_solver, parser, primary_verifier),
            "solver": primary_solver,
            "verifier": primary_verifier,
        }
        second_round = {
            "consensus": self.solver._compare_results(self_check_solver, parser, self_check_verifier),
            "solver": self_check_solver,
            "verifier": self_check_verifier,
        }

        self.assertEqual(self.solver._resolve_multi_round_answer(parser, first_round, second_round), "80")

    def test_multi_round_selection_uses_self_check_when_primary_has_no_safe_answer(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        primary_solver = self._make_result(answer="", answer_confidence=0.0, used_repair=True)
        primary_verifier = self._make_result(answer="", answer_confidence=0.0)
        self_check_solver = self._make_result(answer="69", answer_confidence=0.91)
        self_check_verifier = self._make_result(answer="69", answer_confidence=0.90)

        first_round = {
            "consensus": self.solver._compare_results(primary_solver, parser, primary_verifier),
            "solver": primary_solver,
            "verifier": primary_verifier,
        }
        second_round = {
            "consensus": self.solver._compare_results(self_check_solver, parser, self_check_verifier),
            "solver": self_check_solver,
            "verifier": self_check_verifier,
        }

        self.assertEqual(self.solver._resolve_multi_round_answer(parser, first_round, second_round), "69")

    def test_multi_round_selection_rejects_close_conflicting_answers(self):
        parser = self._make_result(answer="", answer_confidence=0.0, ambiguities=[])
        primary_solver = self._make_result(answer="12", answer_confidence=0.91)
        primary_verifier = self._make_result(answer="12", answer_confidence=0.90)
        self_check_solver = self._make_result(answer="6", answer_confidence=0.92)
        self_check_verifier = self._make_result(answer="6", answer_confidence=0.91)

        first_round = {
            "consensus": self.solver._compare_results(primary_solver, parser, primary_verifier),
            "solver": primary_solver,
            "verifier": primary_verifier,
        }
        second_round = {
            "consensus": self.solver._compare_results(self_check_solver, parser, self_check_verifier),
            "solver": self_check_solver,
            "verifier": self_check_verifier,
        }

        self.assertEqual(self.solver._resolve_multi_round_answer(parser, first_round, second_round), "")

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

    def test_loose_terminal_answer_salvage_extracts_simple_rhs(self):
        raw = (
            "Let me solve the problem carefully.\n"
            "We obtain side = 10.\n"
            "Area = 80.\n"
        )
        salvaged = self.solver._try_salvage_answer_only(raw)
        self.assertIsNotNone(salvaged)
        self.assertEqual(salvaged["final_answer"]["value"], "80")

    def test_option_style_prose_prefers_remote_repair_before_local_salvage(self):
        raw = (
            "1) option one analysis\n"
            "2) option two analysis\n"
            "3) option three analysis\n"
            "4) option four analysis\n"
        )
        self.assertTrue(self.solver._should_prefer_remote_repair(raw))
        self.assertFalse(self.solver._should_prefer_remote_repair("Final answer: 124"))


if __name__ == "__main__":
    unittest.main()
