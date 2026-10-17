"""Run without downloading model weights: python -m unittest discover -s tests -v."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "SmolLM2-360M-Instruct" / "model.py"
spec = importlib.util.spec_from_file_location("smollm_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResponseChecks(unittest.TestCase):
    def test_single_word_instruction_rejects_explanations(self):
        case = {"check": "positive"}
        self.assertEqual(module.check_response(case, "positive\n")["status"], "pass")
        self.assertEqual(module.check_response(case, "positive because it says loved")["status"], "fail")

    def test_json_checks_structure_values_and_types(self):
        case = {"check": "json"}
        self.assertEqual(module.check_response(case, '{"age":24,"name":"Alice"}')["status"], "pass")
        for bad in ('{"age":"24","name":"Alice"}',
                    '{"age":24.0,"name":"Alice"}',
                    '{"age":24,"name":"Alice","extra":1}',
                    '```json\n{"age":24,"name":"Alice"}\n```', '[]'):
            with self.subTest(response=bad):
                self.assertEqual(module.check_response(case, bad)["status"], "fail")

    def test_free_form_answers_are_not_marked_correct(self):
        self.assertEqual(module.check_response({"check": None}, "anything")["status"], "manual_review")

    def test_repetition_metric_detects_repeated_spans(self):
        metrics = module.token_metrics([1, 2, 3, 1, 2, 3])
        self.assertEqual(metrics["unique_token_ratio"], 0.5)
        self.assertEqual(metrics["repeated_trigram_ratio"], 0.25)
        self.assertIsNone(module.token_metrics([1, 2])["repeated_trigram_ratio"])
        self.assertIsNone(module.token_metrics([])["unique_token_ratio"])


if __name__ == "__main__":
    unittest.main()
