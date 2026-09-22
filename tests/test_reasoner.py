import io
import json
import unittest
from unittest.mock import patch

from adahvla.harness import JSONReasoner


class ReasonerTests(unittest.TestCase):
    def client(self):
        return JSONReasoner(
            base_url="https://reasoner.invalid/v1/", model="test-model",
            api_key="offline-test-key", timeout=17,
        )

    def test_json_request_and_response_contract_without_network(self):
        messages = [{"role": "user", "content": "Observe the object"}]
        result = {"progress": "hold"}
        response = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": json.dumps(result)}}],
        }).encode())
        with patch("adahvla.harness.urlopen", return_value=response) as transport:
            self.assertEqual(self.client().complete(messages), result)
        request = transport.call_args.args[0]
        self.assertEqual(request.full_url, "https://reasoner.invalid/v1/chat/completions")
        self.assertEqual(transport.call_args.kwargs["timeout"], 17)
        payload = json.loads(request.data)
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_malformed_or_non_object_content_is_rejected(self):
        for content in ["not JSON", "[]", "null"]:
            with self.subTest(content=content):
                response = io.BytesIO(json.dumps({
                    "choices": [{"message": {"content": content}}],
                }).encode())
                with patch("adahvla.harness.urlopen", return_value=response):
                    with self.assertRaises(ValueError):
                        self.client().complete([])

    def test_transport_failure_propagates_without_hidden_retries(self):
        with patch("adahvla.harness.urlopen", side_effect=TimeoutError("Unavailable")) as transport:
            with self.assertRaises(TimeoutError):
                self.client().complete([])
        self.assertEqual(transport.call_count, 1)


if __name__ == "__main__":
    unittest.main()
