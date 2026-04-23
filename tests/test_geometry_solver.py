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
                "visual_interpretation": ["repaired visual note", "second note"],
                "final_answer": ["1) 12", "2) 13"],
                "consistency_checks": "checked",
                "reasoning_summary": "short",
                "solution_steps": {"step": "value"},
            },
            role="solver",
        )

        self.assertEqual(normalized["visual_interpretation"]["summary"], "repaired visual note; second note")
        self.assertEqual(normalized["final_answer"]["value"], "1) 12\n2) 13")
        self.assertEqual(normalized["consistency_checks"], ["checked"])


if __name__ == "__main__":
    unittest.main()
