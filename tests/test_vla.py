import json
import socket
import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from adahvla.harness import Frame
from adahvla.vla import (
    NaVILAClient, _decode_generation, build_query, read_message, strip_prompt_prefix, temporary_no_init,
)

try:
    import torch
except (ImportError, OSError):
    torch = None


@contextmanager
def fake_server(response="move forward", *, truncate=False, wait=False):
    requests = []
    errors = []
    done = threading.Event()
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(2)

    def serve():
        try:
            with server:
                connection, _ = server.accept()
                with connection:
                    requests.append(read_message(connection))
                    if wait:
                        done.wait(1)
                        return
                    body = json.dumps(response).encode()
                    packet = len(body).to_bytes(8, "big") + body
                    if truncate:
                        connection.sendall(packet[:10])
                    else:
                        for byte in packet:
                            connection.sendall(bytes([byte]))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield server.getsockname()[1], requests
    finally:
        done.set()
        thread.join(2)
        if thread.is_alive():
            raise AssertionError("Fake server did not terminate")
        if errors:
            raise errors[0]


class VLAClientTests(unittest.TestCase):
    def test_fixed_frames_and_raw_action_round_trip(self):
        frames = [
            Frame("data:image/png;base64,ZWdvMQ==", 1.0),
            Frame("ZWdvMg==", 2.0),
            Frame("d3Jpc3Q=", 3.0, view="wrist"),
        ]
        with fake_server(" turn left 30 degrees ") as (port, requests):
            result = NaVILAClient(port=port, num_frames=4).predict("Reach the target", frames)
        self.assertEqual(result, " turn left 30 degrees ")
        self.assertEqual(requests, [{
            "images": ["ZWdvMQ==", "ZWdvMQ==", "ZWdvMQ==", "ZWdvMg=="],
            "query": "Reach the target", "num_video_frames": 4,
        }])

    def test_uniform_sampling_preserves_latest_observation(self):
        frames = [Frame(str(index), float(index)) for index in range(10)]
        with fake_server() as (port, requests):
            NaVILAClient(port=port, num_frames=4).predict("Continue", frames)
        self.assertEqual(requests[0]["images"], ["0", "3", "6", "9"])

    def test_backend_errors_never_become_actions(self):
        frames = [Frame("ZWdv", 0.0)]
        for response, error in [("server_error: malformed image", RuntimeError), ("", ValueError), ({}, ValueError)]:
            with self.subTest(response=response), fake_server(response) as (port, _):
                with self.assertRaises(error):
                    NaVILAClient(port=port).predict("Continue", frames)

    def test_truncated_response_raises(self):
        with fake_server(truncate=True) as (port, _):
            with self.assertRaises(ConnectionError):
                NaVILAClient(port=port).predict("Continue", [Frame("ZWdv", 0.0)])

    def test_read_timeout_is_not_retried(self):
        with fake_server(wait=True) as (port, requests):
            with self.assertRaises(TimeoutError):
                NaVILAClient(port=port, timeout=0.05).predict("Continue", [Frame("ZWdv", 0.0)])
        self.assertEqual(len(requests), 1)

    def test_absent_view_and_invalid_image_fail_before_connecting(self):
        with self.assertRaisesRegex(ValueError, "No executor frames"):
            NaVILAClient().predict("Continue", [Frame("ZWdv", 0.0, view="wrist")])
        with self.assertRaisesRegex(ValueError, "base64"):
            NaVILAClient().predict("Continue", [Frame("data:image/png,raw", 0.0)])


class FrozenInterfaceTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is optional; install it to test model output decoding")
    def test_generation_decoding_trims_only_a_complete_matching_token_prefix(self):
        class Tokenizer:
            def batch_decode(self, token_ids, *, skip_special_tokens):
                self.assert_skip_special_tokens = skip_special_tokens
                return [" ".join(map(str, row)) for row in token_ids.tolist()]

        tokenizer = Tokenizer()
        prompt = torch.tensor([[10, 11, 12]])
        for output, expected in (
            ([20, 21, 22, 23, 24], "20 21 22 23 24"),  # generated-only, longer than prompt
            ([10, 11, 12, 20, 21], "20 21"),  # prompt-prefixed backend
            ([10, 11, 20, 21, 22], "10 11 20 21 22"),  # partial prefix is still generated-only
            ([20, 21], "20 21"),
            ([10, 11, 12], ""),  # no generated tokens
        ):
            with self.subTest(output=output):
                self.assertEqual(_decode_generation(tokenizer, prompt, torch.tensor([output])), expected)
                self.assertTrue(tokenizer.assert_skip_special_tokens)

    def test_prompt_keeps_checkpoint_contract(self):
        prompt = build_query("Find the door", 8)
        self.assertEqual(prompt.count("<image>"), 8)
        self.assertIn('Your assigned task is: "Find the door"', prompt)
        self.assertTrue(prompt.startswith("Imagine you are a robot programmed for navigation tasks."))
        self.assertEqual(strip_prompt_prefix("prefix prompt turn left", "prompt"), "turn left")
        self.assertEqual(strip_prompt_prefix("turn left", "prompt"), "turn left")

    def test_initializers_restored_after_load_failure(self):
        original = lambda *args, **kwargs: "original"
        torch = SimpleNamespace(nn=SimpleNamespace(
            Linear=SimpleNamespace(reset_parameters=original),
            LayerNorm=SimpleNamespace(reset_parameters=original),
            init=SimpleNamespace(**{
                name: original for name in ("kaiming_uniform_", "kaiming_normal_", "uniform_", "normal_")
            }),
        ))
        with self.assertRaises(RuntimeError):
            with temporary_no_init(torch):
                self.assertIsNone(torch.nn.Linear.reset_parameters())
                raise RuntimeError("Checkpoint unavailable")
        self.assertIs(torch.nn.Linear.reset_parameters, original)
        self.assertIs(torch.nn.LayerNorm.reset_parameters, original)
        self.assertTrue(all(value is original for value in vars(torch.nn.init).values()))


if __name__ == "__main__":
    unittest.main()
