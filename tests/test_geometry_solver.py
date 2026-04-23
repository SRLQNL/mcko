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


if __name__ == "__main__":
    unittest.main()
