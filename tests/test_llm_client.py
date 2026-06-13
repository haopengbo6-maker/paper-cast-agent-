import unittest

from src.llm_client import retry_call


class LlmClientTests(unittest.TestCase):
    def test_retry_call_retries_until_success(self):
        attempts = {"count": 0}

        def flaky():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("temporary failure")
            return "ok"

        self.assertEqual(retry_call(flaky, max_attempts=3, delay_seconds=0), "ok")
        self.assertEqual(attempts["count"], 3)

    def test_retry_call_raises_after_max_attempts(self):
        attempts = {"count": 0}

        def always_fails():
            attempts["count"] += 1
            raise RuntimeError("still failing")

        with self.assertRaisesRegex(RuntimeError, "still failing"):
            retry_call(always_fails, max_attempts=2, delay_seconds=0)

        self.assertEqual(attempts["count"], 2)


if __name__ == "__main__":
    unittest.main()
