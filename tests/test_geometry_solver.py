import unittest

from app.geometry_solver import GeometryPhotoSolver, FALLBACK_MESSAGE


class GeometrySolverLiteTests(unittest.TestCase):
    def setUp(self):
        self.solver = GeometryPhotoSolver("test-key", model="some/model")

    def test_requires_api_key(self):
        with self.assertRaises(ValueError):
            GeometryPhotoSolver("", model="some/model")

    def test_empty_blocks_return_fallback(self):
        self.assertEqual(self.solver.solve_content_blocks([]), FALLBACK_MESSAGE)

    def test_unset_model_returns_fallback(self):
        solver = GeometryPhotoSolver("test-key")
        result = solver.solve_content_blocks([{"type": "text", "text": "hi"}])
        self.assertEqual(result, FALLBACK_MESSAGE)

    def test_strip_markdown_removes_bold_italic_and_code(self):
        text = "**bold** and *italic* and `code` and __also__ and _i_"
        cleaned = GeometryPhotoSolver._strip_markdown(text)
        self.assertEqual(cleaned, "bold and italic and code and also and i")

    def test_strip_markdown_removes_headers_and_blockquotes(self):
        text = "# Title\n## Sub\n> quote\nbody"
        cleaned = GeometryPhotoSolver._strip_markdown(text)
        self.assertEqual(cleaned, "Title\nSub\nquote\nbody")

    def test_strip_markdown_removes_code_fences(self):
        text = "```python\nprint(1)\n```\nplain"
        cleaned = GeometryPhotoSolver._strip_markdown(text)
        self.assertIn("print(1)", cleaned)
        self.assertNotIn("```", cleaned)

    def test_strip_markdown_handles_empty(self):
        self.assertEqual(GeometryPhotoSolver._strip_markdown(""), "")

    def test_extract_text_handles_string_content(self):
        data = {"choices": [{"message": {"content": "hello"}}]}
        self.assertEqual(self.solver._extract_text(data), "hello")

    def test_extract_text_handles_list_content(self):
        data = {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}
            ]
        }
        self.assertEqual(self.solver._extract_text(data), "ab")

    def test_extract_text_handles_missing_choices(self):
        self.assertEqual(self.solver._extract_text({}), "")
        self.assertEqual(self.solver._extract_text({"choices": []}), "")

    def test_solve_returns_fallback_on_exception(self):
        def boom(_blocks):
            raise RuntimeError("network down")

        self.solver._call_model = boom
        result = self.solver.solve_content_blocks([{"type": "text", "text": "x"}])
        self.assertEqual(result, FALLBACK_MESSAGE)

    def test_solve_strips_markdown_from_response(self):
        self.solver._call_model = lambda _blocks: "**Answer:** 42"
        result = self.solver.solve_content_blocks([{"type": "text", "text": "x"}])
        self.assertEqual(result, "Answer: 42")

    def test_solve_returns_fallback_for_empty_response(self):
        self.solver._call_model = lambda _blocks: ""
        result = self.solver.solve_content_blocks([{"type": "text", "text": "x"}])
        self.assertEqual(result, FALLBACK_MESSAGE)


if __name__ == "__main__":
    unittest.main()
