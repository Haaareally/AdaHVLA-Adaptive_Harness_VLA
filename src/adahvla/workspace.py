"""Isolated harness revisions and verifiable environment evidence."""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


SOURCE_FILES = ("__init__.py", "harness.py", "vla.py")

_CREDENTIAL_NAME = re.compile(r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)(?:_|$)", re.I)


def _credential_values() -> list[str]:
    return sorted({value for name, value in os.environ.items()
                   if value and _CREDENTIAL_NAME.search(name)}, key=len, reverse=True)


def redact_secrets(text: str) -> str:
    """Remove current credential values before persisting diagnostic text."""
    for value in _credential_values():
        text = text.replace(value, "[redacted]")
    return text


def _copy_redacted(source, destination, values: list[str]) -> None:
    """Stream bytes with bounded memory, including secrets split across reads."""
    secrets = [value.encode("utf-8") for value in values]
    expression = re.compile(b"|".join(re.escape(value) for value in secrets)) if secrets else None
    retained = max((len(value) for value in secrets), default=1) - 1
    pending = b""
    while chunk := source.read1(65536):
        pending += chunk
        boundary = max(0, len(pending) - retained)
        position = 0
        if expression:
            for match in expression.finditer(pending):
                if match.start() >= boundary:
                    break
                destination.write(pending[position:match.start()] + b"[redacted]")
                position = match.end()
        boundary = max(position, boundary)
        destination.write(pending[position:boundary])
        destination.flush()
        pending = pending[boundary:]
    destination.write(expression.sub(b"[redacted]", pending) if expression else pending)
    destination.flush()


def _safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError("Identifiers require 1–96 letters, digits, underscores, or hyphens")
    return value


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    data = _encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class Case:
    id: str
    task: str
    environment: str
    split: str = "train"
    seed: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _safe_id(self.id)
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("A case requires a task instruction")
        if not isinstance(self.environment, str) or not self.environment.strip():
            raise ValueError("A case requires an environment identifier")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("Unknown case split")
        if type(self.seed) is not int or not isinstance(self.metadata, dict):
            raise ValueError("Case seed must be an integer and metadata an object")
        _encoded(self.metadata)


@dataclass(frozen=True)
class Candidate:
    id: str
    parent_id: str | None
    path: str
    digest: str
    files: dict[str, str]

    @property
    def source_path(self) -> Path:
        return Path(self.path) / "adahvla" / "harness.py"


@dataclass(frozen=True)
class Rollout:
    id: str
    candidate_id: str
    case_id: str
    source_digest: str
    success: bool
    metrics: dict
    evidence_path: str
    evidence_digest: str


_SMOKE = """\
import sys
sys.path.insert(0, sys.argv[1])
from adahvla.harness import Frame, Observation, Harness, HarnessVLA
from adahvla.vla import NaVILAClient
class Reasoner:
    calls = 0
    def complete(self, messages):
        self.calls += 1
        return dict(plan=[dict(instruction='Reach goal', completion='Goal reached')] if self.calls == 1 else [],
                    progress='hold' if self.calls == 1 else 'complete', mode='execute',
                    query='Reach goal' if self.calls == 1 else '', evidence='Goal visibly reached', memory='Observed goal')
class Executor:
    calls = 0
    action = object()
    def predict(self, query, frames):
        self.calls += 1
        assert query and frames
        return self.action
executor = Executor()
runtime = HarnessVLA(Harness(Reasoner()), executor)
runtime.reset('Reach the goal')
first = runtime.step(Observation((Frame('aW1hZ2U=', 0.0),)))
assert first.action is executor.action, 'Harness replaced the VLA action'
last = runtime.step(Observation((Frame('aW1hZ2U=', 1.0),)))
assert last.handoff.completed and executor.calls == 1, 'Completion must stop VLA calls'
runtime.reset('A new task')
assert not runtime.harness.state.completed and runtime.harness.state.step == 0
assert NaVILAClient().num_frames >= 2
print('Runtime imports, native action ownership, completion, and reset passed')
"""


class Workspace:
    def __init__(self, root: Path, source_dir: Path | None = None):
        self.root = Path(root).resolve()
        self.source_dir = Path(source_dir).resolve() if source_dir else Path(__file__).parent
        self.candidates = self.root / "candidates"
        self.rollouts = self.root / "rollouts"
        self.candidates.mkdir(parents=True, exist_ok=True)
        self.rollouts.mkdir(parents=True, exist_ok=True)

    def _fingerprint(self, directory: Path) -> dict[str, str]:
        package = directory / "adahvla"
        if package.is_symlink() or not package.is_dir():
            raise ValueError("Candidate package must be a regular directory")
        if {path.name for path in package.iterdir()} != set(SOURCE_FILES):
            raise ValueError("Candidate package must contain exactly the frozen source bundle")
        files = {}
        for name in SOURCE_FILES:
            path = package / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Candidate source is not a regular file: {name}")
            files[name] = _hash(path.read_bytes())
        return files

    def _seal(self, directory: Path, candidate_id: str, parent_id: str | None) -> Candidate:
        files = self._fingerprint(directory)
        candidate = Candidate(candidate_id, parent_id, str(directory), _hash(_encoded(files)), files)
        atomic_json(directory / "candidate.json", asdict(candidate))
        return candidate

    def baseline(self) -> Candidate:
        directory = self.candidates / "C0000"
        if directory.exists():
            candidate = Candidate(**json.loads((directory / "candidate.json").read_text()))
            if candidate.id != "C0000" or candidate.parent_id is not None:
                raise ValueError("Baseline manifest must identify the initial candidate")
            self.verify(candidate)
            return candidate
        directory.mkdir()
        try:
            package = directory / "adahvla"
            package.mkdir()
            for name in SOURCE_FILES:
                source = self.source_dir / name
                if source.is_symlink() or not source.is_file():
                    raise ValueError(f"Missing regular runtime source: {source}")
                (package / name).write_bytes(source.read_bytes())
            return self._seal(directory, "C0000", None)
        except BaseException:
            shutil.rmtree(directory)
            raise

    def verify(self, candidate: Candidate) -> None:
        _safe_id(candidate.id)
        directory = self.candidates / candidate.id
        if directory.is_symlink() or Path(candidate.path).resolve() != directory.resolve():
            raise ValueError("Candidate is outside this workspace")
        files = self._fingerprint(directory)
        if files != candidate.files or _hash(_encoded(files)) != candidate.digest:
            raise ValueError(f"Candidate source changed: {candidate.id}")
        manifest = directory / "candidate.json"
        if manifest.is_symlink() or json.loads(manifest.read_text()) != asdict(candidate):
            raise ValueError("Candidate manifest does not match its identity")

    def read_source(self, candidate: Candidate, path: str = "harness.py") -> str:
        self.verify(candidate)
        if path not in SOURCE_FILES:
            raise ValueError("Source reads are limited to the candidate runtime package")
        return (Path(candidate.path) / "adahvla" / path).read_text(encoding="utf-8")

    def search_source(self, candidate: Candidate, pattern: str) -> list[dict]:
        self.verify(candidate)
        expression = re.compile(pattern)
        return [
            {"path": name, "line": index, "text": line}
            for name in SOURCE_FILES
            for index, line in enumerate(self.read_source(candidate, name).splitlines(), 1)
            if expression.search(line)
        ]

    def apply_patch(self, parent: Candidate, candidate_id: str, edits: list[dict]) -> Candidate:
        self.verify(parent)
        _safe_id(candidate_id)
        if not isinstance(edits, list) or not edits:
            raise ValueError("A patch requires at least one source edit")
        original = self.read_source(parent)
        revised = original
        for edit in edits:
            if not isinstance(edit, dict) or set(edit) != {"path", "old", "new"}:
                raise ValueError("Each edit requires path, old, and new")
            if edit["path"] != "harness.py":
                raise ValueError("Only harness.py is editable")
            old, new = edit["old"], edit["new"]
            if not isinstance(old, str) or not old or not isinstance(new, str) or old == new:
                raise ValueError("An edit requires nonempty old text and a changed replacement")
            if revised.count(old) != 1:
                raise ValueError("Old text must occur exactly once in the current source")
            revised = revised.replace(old, new, 1)
        if revised == original:
            raise ValueError("The patch has no net source change")
        compile(revised, "harness.py", "exec")
        directory = self.candidates / candidate_id
        directory.mkdir()
        try:
            shutil.copytree(Path(parent.path) / "adahvla", directory / "adahvla")
            (directory / "adahvla" / "harness.py").write_text(revised, encoding="utf-8")
            candidate = self._seal(directory, candidate_id, parent.id)
            self.verify(parent)
            return candidate
        except BaseException:
            shutil.rmtree(directory)
            raise

    def diff(self, parent: Candidate, child: Candidate) -> str:
        return "".join(difflib.unified_diff(
            self.read_source(parent).splitlines(keepends=True),
            self.read_source(child).splitlines(keepends=True),
            fromfile=f"{parent.id}/harness.py", tofile=f"{child.id}/harness.py",
        ))

    def run_checks(self, candidate: Candidate) -> dict:
        self.verify(candidate)
        try:
            for name in SOURCE_FILES:
                compile(self.read_source(candidate, name), name, "exec")
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", _SMOKE, candidate.path],
                cwd=candidate.path, text=True, capture_output=True, timeout=30,
                # These offline checks need no user configuration or API credentials.
                env={"PATH": os.defpath, "PYTHONIOENCODING": "utf-8"},
            )
            checks = {"passed": result.returncode == 0, "output": redact_secrets(result.stdout + result.stderr)}
        except (SyntaxError, subprocess.TimeoutExpired) as error:
            checks = {"passed": False, "output": redact_secrets(str(error))}
        finally:
            self.verify(candidate)
        atomic_json(self.root / "checks" / f"{candidate.id}.json", checks)
        return checks

    def evaluate(
        self, candidate: Candidate, case: Case,
        evaluator: Callable[[Candidate, Case, Path], dict], run_id: str,
    ) -> Rollout:
        self.verify(candidate)
        _safe_id(run_id)
        directory = self.rollouts / run_id
        directory.mkdir()
        output = directory / "output"
        output.mkdir()
        case_record = json.loads(_encoded(asdict(case)))
        run_case = Case(**json.loads(_encoded(case_record)))
        try:
            result = evaluator(candidate, run_case, output)
        finally:
            self.verify(candidate)
            if _encoded(asdict(run_case)) != _encoded(case_record):
                raise ValueError("Evaluator changed the case definition during execution")
        if not isinstance(result, dict) or type(result.get("success")) is not bool:
            raise ValueError("Evaluator must return a boolean success")
        metrics, events = result.get("metrics"), result.get("events")
        if not isinstance(metrics, dict) or any(
            not isinstance(key, str) or type(value) not in {int, float} or not math.isfinite(value)
            for key, value in metrics.items()
        ):
            raise ValueError("Evaluator metrics must be finite numeric values")
        if not isinstance(events, list) or not events or any(not isinstance(event, dict) for event in events):
            raise ValueError("Evaluator must provide nonempty event evidence")
        images = result.get("images", [])
        if not isinstance(images, list):
            raise ValueError("Evaluator images must be an array")
        archived = []
        mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        for index, image in enumerate(images):
            if not isinstance(image, dict) or not {"path", "timestamp", "view"} <= image.keys():
                raise ValueError("An evidence image requires path, timestamp, and view")
            if not isinstance(image["path"], str) or not image["path"]:
                raise ValueError("An evidence image requires a relative file path")
            relative = Path(image["path"])
            timestamp, view = image["timestamp"], image["view"]
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Evidence image paths must stay within evaluator output")
            source = output / relative
            if (source.is_symlink() or not source.resolve().is_relative_to(output.resolve())
                    or not source.is_file() or output.is_symlink()
                    or any((output.joinpath(*relative.parts[:length])).is_symlink()
                           for length in range(1, len(relative.parts)))):
                raise ValueError("Evidence image must be a regular file inside evaluator output")
            if type(timestamp) not in {int, float} or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("Evidence image timestamps must be finite nonnegative seconds")
            if not isinstance(view, str) or not view.strip() or source.suffix.lower() not in mime_types:
                raise ValueError("Evidence images require a view and PNG, JPEG, or WebP format")
            data = source.read_bytes()
            if not data:
                raise ValueError("Evidence images cannot be empty")
            target = directory / "images" / f"{index:05d}{source.suffix.lower()}"
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(data)
            archived.append({"path": str(target.relative_to(directory)), "timestamp": timestamp,
                             "view": view, "sha256": _hash(data), "mime_type": mime_types[source.suffix.lower()]})
        evidence = {"id": run_id, "candidate": asdict(candidate), "case": case_record,
                    "success": result["success"], "metrics": metrics, "events": events, "images": archived}
        evidence_path = directory / "evidence.json"
        atomic_json(evidence_path, evidence)
        return Rollout(run_id, candidate.id, case.id, candidate.digest, result["success"],
                       dict(metrics), str(evidence_path), _hash(evidence_path.read_bytes()))

    def read_evidence(self, rollout: Rollout) -> dict:
        _safe_id(rollout.id)
        expected = self.rollouts / rollout.id / "evidence.json"
        path = Path(rollout.evidence_path)
        if path.is_symlink() or path.parent.is_symlink() or path.resolve() != expected.resolve():
            raise ValueError("Rollout evidence is outside this workspace")
        data = path.read_bytes()
        if _hash(data) != rollout.evidence_digest:
            raise ValueError("Rollout evidence changed after evaluation")
        evidence = json.loads(data)
        if (evidence["id"] != rollout.id or evidence["candidate"]["id"] != rollout.candidate_id
                or evidence["candidate"]["digest"] != rollout.source_digest or evidence["case"]["id"] != rollout.case_id
                or evidence["success"] != rollout.success or evidence["metrics"] != rollout.metrics):
            raise ValueError("Rollout identity does not match its recorded evidence")
        for image in evidence["images"]:
            image_path = path.parent / image["path"]
            if (image_path.is_symlink() or image_path.parent.is_symlink()
                    or not image_path.resolve().is_relative_to(path.parent.resolve())):
                raise ValueError("Archived image is outside the evidence directory")
            image_data = image_path.read_bytes()
            if _hash(image_data) != image["sha256"]:
                raise ValueError("Archived image changed after evaluation")
        return evidence

    def rollout_content(self, rollout: Rollout) -> list[dict]:
        evidence = self.read_evidence(rollout)
        content = [{"type": "text", "text": "Complete recorded event evidence; supplied images are labeled observations, "
                    "not a claim of complete video coverage.\n" + _encoded(evidence).decode()}]
        for image in evidence["images"]:
            image_data = (Path(rollout.evidence_path).parent / image["path"]).read_bytes()
            content.extend([
                {"type": "text", "text": f"Observation at {image['timestamp']} seconds, view={image['view']}"},
                {"type": "image_url", "image_url": {"url": "data:" + image["mime_type"] + ";base64," + base64.b64encode(image_data).decode()}},
            ])
        return content


@dataclass
class CommandEvaluator:
    command: Sequence[str]
    timeout: float = 600
    _protocol: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.command, str) or not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise ValueError("An evaluator command must be a nonempty argument sequence")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Evaluator timeout must be positive and finite")
        self.command = tuple(
            str(Path(part).resolve()) if Path(part).suffix.lower() == ".py" and Path(part).is_file() else part
            for part in self.command
        )
        self._protocol = self._current_fingerprint()

    def _current_fingerprint(self) -> dict:
        scripts = {
            str(Path(part).resolve()): _hash(Path(part).read_bytes())
            for part in self.command
            if Path(part).suffix.lower() == ".py" and Path(part).is_file()
        }
        return {"command": list(self.command), "timeout": self.timeout, "script_sha256": scripts}

    def fingerprint(self) -> dict:
        if self._current_fingerprint() != self._protocol:
            raise ValueError("Evaluator command, timeout, or Python source changed")
        return json.loads(_encoded(self._protocol))

    def __call__(self, candidate: Candidate, case: Case, output_dir: Path) -> dict:
        self.fingerprint()
        output_dir = Path(output_dir).resolve()
        request_path, result_path = output_dir / "request.json", output_dir / "result.json"
        atomic_json(request_path, {"candidate": asdict(candidate), "case": asdict(case)})
        # Only the online harness requires a credential; unrelated provider/cloud
        # credentials should not reach generated candidate code or the simulator.
        env = {name: value for name, value in os.environ.items()
               if not _CREDENTIAL_NAME.search(name) or name == "ADAHVLA_API_KEY"}
        env["PYTHONPATH"] = candidate.path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with (output_dir / "process.log").open("wb") as log:
            try:
                process = subprocess.Popen(
                    [*self.command, "--request", str(request_path), "--output", str(result_path)],
                    cwd=candidate.path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix",
                )
                stream_errors = []

                def copy_output():
                    try:
                        _copy_redacted(process.stdout, log, _credential_values())
                    except Exception as error:
                        stream_errors.append(error)

                def kill_process():
                    try:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                    except ProcessLookupError:
                        pass

                reader = threading.Thread(target=copy_output, daemon=True)
                reader.start()
                try:
                    returncode = process.wait(timeout=self.timeout)
                except BaseException:
                    kill_process()
                    process.wait()
                    raise
                finally:
                    reader.join(timeout=5)
                    if reader.is_alive():
                        # A subprocess descendant may still own the output pipe.
                        kill_process()
                        reader.join(timeout=5)
                    if reader.is_alive():
                        raise RuntimeError("Evaluator output stream did not close")
                    process.stdout.close()
                if stream_errors:
                    raise RuntimeError("Cannot save evaluator process output") from stream_errors[0]
            finally:
                self.fingerprint()
        if returncode != 0:
            raise RuntimeError(f"Evaluator exited with {returncode}; see {output_dir / 'process.log'}")
        if result_path.is_symlink() or not result_path.is_file():
            raise ValueError("Evaluator did not write a regular result JSON file")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("source_digest") != candidate.digest:
            raise ValueError("Evaluator must acknowledge the exact candidate source_digest")
        return {key: value for key, value in payload.items() if key != "source_digest"}
