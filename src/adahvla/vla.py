"""Frozen VLA interface, NaVILA client, and model serving entry point."""

from __future__ import annotations

import argparse
import base64
import json
import socket
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .harness import Frame


class VLA(Protocol):
    def predict(self, query: str, frames: Sequence[Frame]) -> Any:
        """Predict a native action; the caller alone executes it in the environment."""
        ...


def recv_exactly(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError(f"Socket closed with {remaining} bytes still expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(connection: socket.socket, max_bytes: int = 64 * 1024 * 1024) -> Any:
    size = int.from_bytes(recv_exactly(connection, 8), "big")
    if not 0 < size <= max_bytes:
        raise ValueError(f"Invalid message size: {size}")
    return json.loads(recv_exactly(connection, size).decode("utf-8"))


def write_message(connection: socket.socket, payload: Any) -> None:
    body = json.dumps(payload).encode("utf-8")
    connection.sendall(len(body).to_bytes(8, "big") + body)


@dataclass
class NaVILAClient:
    host: str = "127.0.0.1"
    port: int = 54321
    num_frames: int = 8
    timeout: float = 300.0
    view: str = "ego"

    def __post_init__(self) -> None:
        if self.num_frames < 2:
            raise ValueError("NaVILA requires at least two frames")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def predict(self, query: str, frames: Sequence[Frame]) -> str:
        if not query.strip():
            raise ValueError("An executor query is required")
        selected = sorted(
            (frame for frame in frames if frame.view == self.view),
            key=lambda frame: frame.timestamp,
        )
        if not selected:
            raise ValueError(f"No executor frames for view {self.view!r}")
        # Repeat the earliest real observation instead of inventing unseen history.
        if len(selected) < self.num_frames:
            selected = [selected[0]] * (self.num_frames - len(selected)) + selected
        indices = [
            index * (len(selected) - 1) // (self.num_frames - 1)
            for index in range(self.num_frames)
        ]
        images = []
        for index in indices:
            image = selected[index].image
            if image.startswith("data:"):
                header, separator, image = image.partition(",")
                if not separator or not header.endswith(";base64"):
                    raise ValueError("Frame data URLs must contain base64 images")
            if not image:
                raise ValueError("Executor images must be nonempty")
            images.append(image)
        request = {"images": images, "query": query, "num_video_frames": self.num_frames}
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as connection:
            write_message(connection, request)
            response = read_message(connection, max_bytes=1024 * 1024)
        if not isinstance(response, str) or not response.strip():
            raise ValueError("NaVILA returned an empty or non-text action")
        if response.lstrip().startswith("server_error:"):
            raise RuntimeError(response)
        return response


# NaVILA model backend; heavy dependencies load only when the server starts.

def strip_prompt_prefix(decoded_text: str, decoded_prompt: str) -> str:
    candidate = decoded_text.strip()
    prompt = decoded_prompt.strip()
    position = candidate.find(prompt) if prompt else -1
    if position != -1:
        answer = candidate[position + len(prompt) :].strip()
        if answer:
            return answer
    return candidate


def _decode_generation(tokenizer: Any, input_ids: Any, output_ids: Any) -> str:
    """Accept generated-only NaVILA tokens and prompt-prefixed backend outputs."""
    import torch

    prompt_length = input_ids.shape[1]
    if output_ids.shape[1] >= prompt_length and torch.equal(output_ids[:, :prompt_length], input_ids):
        output_ids = output_ids[:, prompt_length:]
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def build_query(query: str, num_frames: int) -> str:
    # This wording belongs to the frozen checkpoint interface, not harness policy.
    image_token = "<image>\n"
    return (
        "Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f"of historical observations {image_token * (num_frames - 1)}, and current observation <image>\n. "
        f'Your assigned task is: "{query}" '
        "Analyze this series of images to decide your next action, which could be turning left or right by a specific "
        "degree, moving forward a certain distance, or stop if the task is completed."
    )


@contextmanager
def temporary_no_init(torch: Any):
    targets = [
        (torch.nn.Linear, "reset_parameters"),
        (torch.nn.LayerNorm, "reset_parameters"),
        *[(torch.nn.init, name) for name in ("kaiming_uniform_", "kaiming_normal_", "uniform_", "normal_")],
    ]
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    try:
        for owner, name, _ in originals:
            setattr(owner, name, lambda *args, **kwargs: None)
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


class NaVILAServer:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        num_video_frames: int = 8,
        conv_mode: str = "llama_3",
    ) -> None:
        import torch
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        if num_video_frames < 2:
            raise ValueError("NaVILA requires at least two frames")
        self.device = device
        self.num_video_frames = num_video_frames
        self.conv_mode = conv_mode
        with temporary_no_init(torch):
            self.tokenizer, model, self.image_processor, _ = load_pretrained_model(
                model_path, get_model_name_from_path(model_path), None, device=device,
            )
        # Accelerate may own placement when loading leaves meta parameters.
        self.model = (
            model if any(str(parameter.device) == "meta" for parameter in model.parameters())
            else model.to(device)
        ).eval()

    def process_request(self, images: list[str], query: str, num_video_frames: int | None = None) -> str:
        import torch
        from PIL import Image
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import KeywordsStoppingCriteria, process_image, tokenizer_image_token

        total_frames = self.num_video_frames if num_video_frames is None else num_video_frames
        if total_frames < 2 or len(images) != total_frames:
            raise ValueError(f"Expected exactly {total_frames} frames, received {len(images)}")
        self.model.config.image_processor = self.image_processor
        tensors = []
        for encoded in images:
            with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
                tensors.append(process_image(image.convert("RGB"), self.model.config, None))
        if all(tensor.shape == tensors[0].shape for tensor in tensors):
            image_inputs = [torch.stack(tensors).to(self.device, dtype=torch.float16)]
        else:
            image_inputs = [tensor.to(self.device, dtype=torch.float16) for tensor in tensors]

        conversation = conv_templates[self.conv_mode].copy()
        conversation.append_message(conversation.roles[0], build_query(query, total_frames))
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        stop = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
        stopping = KeywordsStoppingCriteria([stop], self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device),
                images=image_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=[stopping],
            )
        return _decode_generation(self.tokenizer, input_ids, output_ids)

    def serve(self, host: str = "127.0.0.1", port: int = 54321) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(4)
            print(f"NaVILA listening on {host}:{port}", flush=True)
            while True:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(300.0)
                    try:
                        request = read_message(connection)
                        response = self.process_request(
                            request["images"], request["query"], request.get("num_video_frames"),
                        )
                        write_message(connection, response)
                    except Exception as error:
                        print(f"NaVILA request failed: {error}", flush=True)
                        try:
                            write_message(connection, f"server_error: {error}")
                        except OSError:
                            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54321)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--conv-mode", default="llama_3")
    args = parser.parse_args()
    backend = NaVILAServer(args.model_path, args.device, args.num_video_frames, args.conv_mode)
    backend.serve(args.host, args.port)


if __name__ == "__main__":
    main()
