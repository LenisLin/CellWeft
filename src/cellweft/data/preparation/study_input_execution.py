"""Recoverable Codex CLI orchestration for independently reviewed study inputs."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterator

from cellweft.data.preparation.io_coordination import default_io_root


MODEL = "gpt-6-sol"
REASONING_EFFORT = "high"
MAX_WORKERS = 15
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_PHASES = {"prepare", "repair", "review"}
_JOB_NAME = re.compile(r"^(\d+)_(prepare|review|repair)_(\d+)$")
_REPORT_FIELDS = {
    "study": {"type": "string"},
    "status": {"type": "string", "enum": ["prepared", "pass", "partial", "needs_repair", "blocked", "not_localized", "waiting_io"]},
    "script": {"type": "string"},
    **{name: {"type": "integer"} for name in (
        "expected_images", "exported_images", "sample_rows", "sample_linked_images", "patient_linked_samples")},
    "summary": {"type": "string"},
    **{name: {"type": "array", "items": {"type": "string"}} for name in (
        "issues", "controller_requests", "evidence_paths")},
}
REPORT_SCHEMA = {"type": "object", "additionalProperties": False,
                 "properties": _REPORT_FIELDS, "required": list(_REPORT_FIELDS)}


def _validate_phase_models(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("phase_models must be a dictionary")
    unknown = set(value) - _PHASES
    if unknown:
        raise ValueError(f"phase_models contains unknown phase(s): {sorted(unknown)}")
    for phase, selected in value.items():
        if not isinstance(selected, dict) or set(selected) != {"model", "reasoning_effort"}:
            raise ValueError(f"phase_models[{phase}] must contain exactly model and reasoning_effort")
        if not isinstance(selected["model"], str) or not selected["model"].strip():
            raise ValueError(f"phase_models[{phase}].model must be a nonempty string")
        if not isinstance(selected["reasoning_effort"], str) or selected["reasoning_effort"] not in _EFFORTS:
            raise ValueError(f"phase_models[{phase}].reasoning_effort is invalid")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _close_job_logs(job: Path, handles: tuple[Any, ...]) -> list[dict[str, str]]:
    failures = []
    for path, handle in zip((job / "events.jsonl", job / "stderr.log"), handles):
        try:
            handle.close()
        except OSError as error:
            failure = {"path": str(path), "error": repr(error)}
            failures.append(failure)
            print(f"CLI log close failed: {path}: {error!r}", file=sys.stderr, flush=True)
    return failures


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            exc.add_note(f"Failed to remove private state file {temporary}: {cleanup_error}")
        raise


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _proc_stat(pid: int) -> tuple[str, int, int] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return fields[0], int(fields[2]), int(fields[19])


def process_is_alive(pid: int, job: Path, start_ticks: int | None = None) -> bool:
    """Match the CLI identity and wait for any children in its process group.

    Legacy jobs lack start ticks, so their command must contain their unique
    report path. A persisted start tick protects new jobs from PID reuse.
    """
    stat = _proc_stat(pid)
    if stat is not None and stat[0] != "Z":
        if start_ticks is not None:
            return stat[2] == start_ticks
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            command = []
        return os.fsencode(str(job / "report.json")) in command
    # The CLI can exit before an already spawned writer. Its process group
    # remains owned until the last child closes, including during recovery.
    for candidate in Path("/proc").iterdir():
        if candidate.name.isdigit():
            child = _proc_stat(int(candidate.name))
            if child is not None and child[0] != "Z" and child[1] == pid:
                return True
    return False


def _discover_jobs(run_root: Path) -> dict[str, list[dict[str, int]]]:
    """Find CLIs started just before a controller crashed while saving their PID."""
    result: dict[str, list[dict[str, int]]] = {}
    prefix = os.fsencode(str(run_root / "jobs")) + b"/"
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            command = (candidate / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for index, argument in enumerate(command[:-1]):
            if argument != b"--output-last-message":
                continue
            output = command[index + 1]
            if not output.startswith(prefix) or not output.endswith(b"/report.json"):
                continue
            stat = _proc_stat(int(candidate.name))
            if stat and stat[0] != "Z":
                job = str(Path(os.fsdecode(output)).parent)
                # The Node launcher and native CLI carry the same arguments;
                # both belong to the one session created by Popen.
                pid = stat[1]
                owners = result.setdefault(job, [])
                if any(owner["pid"] == pid for owner in owners):
                    continue
                leader = _proc_stat(pid)
                owner = {"pid": pid}
                if leader:
                    owner["start_ticks"] = leader[2]
                owners.append(owner)
    return result


@dataclass(frozen=True)
class ControllerConfig:
    """Runtime locations and bounded CLI concurrency; scientific rules live in docs."""

    run_root: Path
    workers: int = 10
    model: str = MODEL
    reasoning_effort: str = REASONING_EFFORT
    codex: str = "codex"
    nas_lock_root: Path | None = None
    poll_interval: float = 2.0
    io_retry_delay: float = 30.0

    def __post_init__(self) -> None:
        if type(self.workers) is not int or not 1 <= self.workers <= MAX_WORKERS:
            raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty string")
        if self.reasoning_effort not in _EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {sorted(_EFFORTS)}")
        if self.poll_interval < 0 or self.io_retry_delay < 0:
            raise ValueError("poll and retry intervals must be nonnegative")


def load_controller_status(run_root: Path) -> dict[str, Any]:
    """Read the persisted status without acquiring a writer lock or changing files."""
    return _read_json(Path(run_root) / "status.json")


class StudyInputController:
    """Own one finite queue, retaining independent review and two repair rounds."""

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.root = Path(config.run_root).resolve()
        self.plan = _read_json(self.root / "plan.json")
        self.repo = Path(self.plan["repo"])
        self.nas = Path(self.plan["nas"])
        self.lock_root = Path(config.nas_lock_root or default_io_root(self.nas))
        records = self.plan["studies"]
        self.records = {record["study"]: record for record in records}
        if len(self.records) != len(records):
            raise ValueError("Plan contains duplicate study owners")
        for study in self.records:
            if study in {"", ".", ".."} or Path(study).name != study:
                raise ValueError("Study identifiers must be single path components")
        self.studies: dict[str, dict[str, Any]] = {}
        self.pending: dict[str, dict[str, Any]] = {}
        self.active: dict[str, dict[str, Any]] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self.handles: dict[str, tuple[Any, Any]] = {}
        self.serial = 0
        self.stop_requested = False
        self.admission_stopped = False
        self.runtime = {"workers": config.workers, "model": config.model,
                        "reasoning_effort": config.reasoning_effort,
                        "phase_models": {},
                        "pause_admission": False, "pause_dispatch": False,
                        "held_studies": []}
        self.control_error: str | None = None
        self.resource_paused = False

    def _job_model(self, job: Path, assignment: dict[str, Any],
                   saved_worker: dict[str, Any], saved: dict[str, Any]) -> dict[str, str]:
        receipt = _read_json(job / "receipt.json") if (job / "receipt.json").exists() else {}
        return {
            "model": next((source["model"] for source in (assignment, receipt, saved_worker, self.plan, saved)
                           if source.get("model")), "unknown"),
            "reasoning_effort": next((source["reasoning_effort"] for source in
                                      (assignment, receipt, saved_worker, self.plan, saved)
                                      if source.get("reasoning_effort")), "unknown"),
        }

    def _phase_model(self, phase: str) -> tuple[str, str]:
        selected = self.runtime["phase_models"].get(phase)
        if selected is None:
            return self.runtime["model"], self.runtime["reasoning_effort"]
        return selected["model"], selected["reasoning_effort"]

    @contextmanager
    def controller_lock(self) -> Iterator[None]:
        """Reject a second upstream controller sharing the NAS coordination root."""
        self.lock_root.mkdir(parents=True, exist_ok=True)
        with (self.lock_root / "study_input_controller.lock").open("a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another study-input controller owns this lock root") from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _enqueue(self, study: str, phase: str, repair: int, *, reason: str = "phase", retry_at: float = 0) -> None:
        self.pending[study] = {"phase": phase, "repair": repair, "reason": reason, "retry_at": retry_at}
        state = self.studies[study]
        state.update(next_phase=phase, repair_count=repair, retry_at=retry_at, continuation_reason=reason, terminal=False)
        state["status"] = "waiting_io" if reason == "io" else f"{phase}_queued"

    def _report(self, path: Path, study: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            report = _read_json(path)
        except json.JSONDecodeError:
            return None  # An interrupted final-message write is not a completed report.
        if not isinstance(report, dict) or report.get("study") != study:
            raise ValueError(f"Report study identity mismatch: {path}")
        for name, contract in _REPORT_FIELDS.items():
            value = report.get(name)
            kind = contract["type"]
            valid = (isinstance(value, str) if kind == "string" else
                     type(value) is int if kind == "integer" else
                     isinstance(value, list) and all(isinstance(item, str) for item in value))
            if not valid:
                raise ValueError(f"Invalid report field {name}: {path}")
        if report["status"] not in _REPORT_FIELDS["status"]["enum"]:
            raise ValueError(f"Unknown report status: {path}")
        return report

    def _consume(self, study: str, phase: str, repair: int, job: Path, report: dict[str, Any]) -> None:
        state = self.studies[study]
        state.update(status=report["status"], last_report=str(job / "report.json"),
                     last_processed_job=str(job), exported_images=report["exported_images"],
                     sample_rows=report["sample_rows"], controller_requests=report["controller_requests"],
                     repair_count=repair, terminal=False)
        state.pop("error", None)
        if report["status"] == "waiting_io":
            self._enqueue(study, phase, repair, reason="io", retry_at=time.time() + self.config.io_retry_delay)
        elif phase in {"prepare", "repair"} and report["status"] not in {"blocked", "not_localized"}:
            self._enqueue(study, "review", repair)
        elif phase == "review" and report["status"] == "needs_repair" and repair < 2:
            self._enqueue(study, "repair", repair + 1)
        else:
            state.update(terminal=True, independently_reviewed=phase == "review")
            state.pop("next_phase", None)
            self.pending.pop(study, None)

    def recover(self) -> None:
        """Adopt live jobs, consume closed reports, and re-enter interrupted phases."""
        saved = load_controller_status(self.root) if (self.root / "status.json").exists() else {}
        old_studies = saved.get("studies", {})
        old_workers = {item["job"]: item for item in saved.get("workers", {}).values()}
        discovered = _discover_jobs(self.root)
        for study, record in self.records.items():
            state = dict(old_studies.get(study, {}))
            state.setdefault("status", "queued")
            state.setdefault("script", record["script"])
            state.setdefault("repair_count", 0)
            state.setdefault("io_continuations", 0)
            state.setdefault("execution_continuations", 0)
            state.setdefault("admitted", False)
            self.studies[study] = state
            jobs = []
            for job in (self.root / "jobs" / study).glob("*"):
                match = _JOB_NAME.match(job.name)
                if match and job.is_dir():
                    serial, phase, repair = int(match[1]), match[2], int(match[3])
                    self.serial = max(self.serial, serial + 1)
                    jobs.append((serial, phase, repair, job))
            if not jobs:
                # A terminal legacy record is retained even if its job was archived.
                if state["status"] in {"pass", "partial", "blocked", "not_localized", "needs_repair"}:
                    state.setdefault("terminal", True)
                    continue
                self._enqueue(study, state.get("next_phase", "prepare"), state["repair_count"])
                continue
            state["admitted"] = True
            live = []
            for _, phase, repair, job in jobs:
                assignment = _read_json(job / "assignment.json") if (job / "assignment.json").exists() else {}
                saved_worker = old_workers.get(str(job), {})
                worker = {**assignment, **saved_worker,
                          **self._job_model(job, assignment, saved_worker, saved)}
                owners = discovered.get(str(job), [])
                if not owners and worker.get("pid"):
                    owners = [worker]
                for owner in owners:
                    if process_is_alive(int(owner["pid"]), job, owner.get("start_ticks")):
                        live.append((phase, repair, job, {**worker, **owner}))
            if len(live) > 1:
                raise RuntimeError(f"Multiple live owners for study {study}; refusing admission")
            if live:
                phase, repair, job, worker = live[0]
                slot = str(len(self.active) + 1)
                stat = _proc_stat(int(worker["pid"]))
                self.active[slot] = {**worker, "study": study, "phase": phase, "repair": repair,
                                     "job": str(job), "adopted": True,
                                     "start_ticks": stat[2] if stat else worker.get("start_ticks")}
                state.update(status=phase, job=str(job), worker=slot, repair_count=repair)
                continue
            _, phase, repair, job = max(jobs, key=lambda item: item[0])
            assignment = _read_json(job / "assignment.json") if (job / "assignment.json").exists() else {}
            saved_worker = old_workers.get(str(job), {})
            if state.get("last_processed_job") == str(job) and state.get("terminal"):
                continue
            report = self._report(job / "report.json", study)
            if report is not None:
                if state.get("last_processed_job") == str(job):
                    if not state.get("terminal"):
                        self._enqueue(study, state.get("next_phase", phase), state["repair_count"],
                                      reason=state.get("continuation_reason", "phase"), retry_at=state.get("retry_at", 0))
                else:
                    self._consume(study, phase, repair, job, report)
                if not (job / "receipt.json").exists():
                    _write_json(job / "receipt.json", {"exit_code": None, "finished_at": _now(),
                                **self._job_model(job, assignment, saved_worker, saved),
                                "recovered_report": True})
            else:
                self._enqueue(study, phase, repair, reason="interrupted")
        self._save()

    def _save(self) -> None:
        _write_json(self.root / "status.json", {
            "updated_at": _now(), "model": self.runtime["model"],
            "reasoning_effort": self.runtime["reasoning_effort"],
            "workers": self.active, "studies": self.studies, "queued": len(self.pending),
            "stop_requested": self.stop_requested, "stop_admission": self.admission_stopped,
            "pause_admission": self.runtime["pause_admission"],
            "pause_dispatch": self.runtime["pause_dispatch"],
            "phase_models": self.runtime["phase_models"],
            "held_studies": self.runtime["held_studies"], "control_error": self.control_error,
            "local_resource_paused": self.resource_paused,
            "worker_limit": self.runtime["workers"], "nas_lock_root": str(self.lock_root),
            "controller_pid": os.getpid(), "states": dict(Counter(item["status"] for item in self.studies.values())),
        })

    def _write_job_receipt(self, job: Path, worker: dict[str, Any], code: int | None,
                           close_errors: list[dict[str, str]]) -> None:
        path = job / "receipt.json"
        if path.exists():
            if not close_errors:
                return
            receipt = _read_json(path)
            receipt.setdefault("log_close_errors", []).extend(close_errors)
        else:
            receipt = {"exit_code": code, "finished_at": _now(), "pid": worker.get("pid"),
                       "model": worker["model"], "reasoning_effort": worker["reasoning_effort"],
                       "adopted": worker.get("adopted", False),
                       "log_close_errors": close_errors}
            if "launch_error" in worker:
                receipt["launch_error"] = worker["launch_error"]
        _write_json(path, receipt)

    def _prompt(self, study: str, phase: str, repair: int, job: Path) -> str:
        previous = self.studies[study].get("last_report")
        interrupted = self.studies[study].get("job")
        return f"""你是CellWeft单study输入任务。主控已授权实际执行本地完整来源整理和独立复核。用中文报告。
先读AGENTS.md、docs/project/status.md相关记录、docs/data/study_input.md、study_input_review.md及templates/study_input，再检查本study现有脚本、来源和产物。以现行文档为科学规则，终点为原生强度input.tiff及panel/image/sample/field_mapping/review_results。
只编辑TASK指定study脚本及本study NAS派生文件；共有src、docs、S1、全局catalog或其他study问题写controller_requests，交主控协调。不要另启动agent或Codex CLI。使用项目.venv-data/bin/python -B和PYTHONPATH=src。私有身份仅在NAS核对和保存，不将患者名单写代码或报告；公开检索只用公开study信息；不联系作者。
逐study主动追查sample和patient连接，按现行文档保留缺口、临床来源和覆盖。仅处理已完整合格的本地来源，复用已有核验证据、已确认图像和正确索引，新增批次合并现有表；保持在途和原始来源不变。每图或合理小批关闭文件并保存可接续索引和JOB/progress.json。
资源及I/O：CPU辅助线程最多4。共享锁根由CELLWEFT_NAS_LOCK_ROOT={self.lock_root}传入；逐image使用src/cellweft/data/preparation/io_coordination.py的正式写入lease，获取NAS名额后再获取study/input锁。只在正式发布时占写槽，来源读取/局部暂存/计算在槽外执行；重读使用共享read lease。禁止用旧RUN/io_slot.py包装整study脚本，禁止在外层重复获取helper已持有的锁。检查脚本调用共享helper，已有正确行为复用。NAS预留100GiB、本地临时盘预留64GiB，先估算空间。
本次phase={phase}，科学repair轮={repair}。waiting_io及基础设施中断续跑不增加科学repair轮；若共享I/O或空间暂时不可用，在关闭当前文件、保存具体checkpoint后用status=waiting_io返回，issues/evidence_paths说明等待原因、完成范围与续跑位置。I/O续跑只补未完成范围，不把等待当科学问题。所有spawned命令等待结束，不遗留无主文件写。
ROLE：{'独立复核。你未参与本study整理；只读来源、script和实际产物，按study_input_review.md检查。只允许更新本study review_results.csv及JOB报告，不修改script或图像。确认范围后由你按文档独占锁写入D01绑定；主控不替你写科学pass。差异返回needs_repair，必要证据缺口返回partial。' if phase == 'review' else '实际整理' + ('修补。回读上次独立review，仅修补受影响范围及直接依赖。' if phase == 'repair' else '。实际执行study脚本完成本地合格范围。') + '整理者不冒充独立reviewer；独立CLI将在你关闭产物后复核。'}
JOB={job}
上次报告={previous or '无'}
上次执行目录={interrupted or '无'}
按提供schema返回实际事实；代码写成或语法通过不代表已导出或验收。记录实际image、sample和patient覆盖、证据及具体请求。
TASK\n{json.dumps(self.records[study], ensure_ascii=False, indent=2)}\n"""

    def _launch(self, study: str, task: dict[str, Any], slot: str) -> None:
        phase, repair = task["phase"], task["repair"]
        job = self.root / "jobs" / study / f"{self.serial:04d}_{phase}_{repair}"
        self.serial += 1
        job.mkdir(parents=True)
        (job / "prompt.txt").write_text(self._prompt(study, phase, repair, job))
        model, reasoning_effort = self._phase_model(phase)
        assignment = {"study": study, "phase": phase, "repair": repair, "job": str(job),
                      "started_at": _now(), "continuation_reason": task["reason"],
                      "model": model, "reasoning_effort": reasoning_effort}
        _write_json(job / "assignment.json", assignment)
        command = [self.config.codex, "exec", "--model", assignment["model"], "-c",
                   f'model_reasoning_effort="{assignment["reasoning_effort"]}"',
                   "--approve-for-me", "--add-dir", str(self.nas), "-c", "sandbox_workspace_write.network_access=true",
                   "--ephemeral", "--json", "--output-schema", str(self.root / "report_schema.json"),
                   "--output-last-message", str(job / "report.json"), "-C", str(self.repo), "-"]
        environment = os.environ.copy()
        environment.update(CELLWEFT_NAS_LOCK_ROOT=str(self.lock_root), CELLWEFT_DATA_ROOT=str(self.nas),
                           CELLWEFT_STUDY_INPUT_JOB=str(job), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4",
                           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        executable = shutil.which(self.config.codex)
        if executable:
            environment["PATH"] = str(Path(executable).parent) + os.pathsep + environment.get("PATH", "")
        events = None
        errors = None
        try:
            events = (job / "events.jsonl").open("w")
            errors = (job / "stderr.log").open("w")
            with (job / "prompt.txt").open() as stdin:
                process = subprocess.Popen(command, stdin=stdin, stdout=events, stderr=errors,
                                           env=environment, start_new_session=True, cwd=self.repo)
        except BaseException as error:
            close_errors = _close_job_logs(job, tuple(handle for handle in (events, errors) if handle is not None))
            if close_errors:
                try:
                    self._write_job_receipt(job, {**assignment, "launch_error": repr(error)}, None,
                                            close_errors)
                except OSError as receipt_error:
                    error.add_note(f"Could not record CLI log close failures in {job / 'receipt.json'}: {receipt_error!r}")
                    print(f"CLI receipt write failed: {job / 'receipt.json'}: {receipt_error!r}",
                          file=sys.stderr, flush=True)
            raise
        self.processes[slot] = process
        self.handles[slot] = (events, errors)
        stat = _proc_stat(process.pid)
        assignment.update(pid=process.pid, start_ticks=stat[2] if stat else None)
        self.active[slot] = assignment
        _write_json(job / "assignment.json", assignment)
        self.pending.pop(study)
        state = self.studies[study]
        if task["reason"] == "io":
            state["io_continuations"] += 1
        elif task["reason"] == "interrupted":
            state["execution_continuations"] += 1
        state.update(status=phase, job=str(job), worker=slot, admitted=True, repair_count=repair)
        self._save()

    def _collect(self) -> None:
        for slot, worker in list(self.active.items()):
            job, study = Path(worker["job"]), worker["study"]
            process = self.processes.get(slot)
            code = process.poll() if process is not None else None
            if process is not None and code is None:
                continue
            if process_is_alive(worker["pid"], job, worker.get("start_ticks")):
                continue
            close_errors = _close_job_logs(job, self.handles.pop(slot, ()))
            self.processes.pop(slot, None)
            self.active.pop(slot)
            self._write_job_receipt(job, worker, code, close_errors)
            try:
                report = self._report(job / "report.json", study)
                if report is None:
                    raise RuntimeError(f"Codex CLI exit={code}; no closed report in {job}")
                self._consume(study, worker["phase"], worker["repair"], job, report)
            except (ValueError, RuntimeError) as error:
                _write_json(job / "failure.json", {"error": str(error), "time": _now()})
                self.studies[study]["error"] = str(error)
                self._enqueue(study, worker["phase"], worker["repair"], reason="interrupted",
                              retry_at=time.time() + self.config.io_retry_delay)
                if code == 2:
                    self.stop_requested = True
            self._save()

    def _controls(self) -> None:
        from cellweft.data.preparation.local_resources import admission_paused
        self.resource_paused = admission_paused(self.plan)
        self.stop_requested |= (self.root / "STOP").exists()
        self.admission_stopped |= (self.root / "STOP_ADMISSION").exists()
        path = self.root / "control.json"
        if not path.exists():
            return
        try:
            control = _read_json(path)
            if not isinstance(control, dict) or set(control) - set(self.runtime):
                raise ValueError("control.json must contain only supported runtime keys")
            updated = {**self.runtime, **control}
            if type(updated["workers"]) is not int or not 1 <= updated["workers"] <= MAX_WORKERS:
                raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
            if not isinstance(updated["model"], str) or not updated["model"].strip():
                raise ValueError("model must be a nonempty string")
            if updated["reasoning_effort"] not in _EFFORTS:
                raise ValueError("invalid reasoning_effort")
            _validate_phase_models(updated["phase_models"])
            if any(type(updated[key]) is not bool for key in ("pause_admission", "pause_dispatch")):
                raise ValueError("pause controls must be boolean")
            held = updated["held_studies"]
            if (not isinstance(held, list) or
                    any(not isinstance(study, str) or study not in self.records for study in held) or
                    len(set(held)) != len(held)):
                raise ValueError("held_studies must list distinct studies in plan.json")
        except (OSError, ValueError, TypeError) as error:
            self.control_error = str(error)
            return
        self.runtime = updated
        self.control_error = None

    def run(self) -> dict[str, Any]:
        """Recover and drain; STOP and signals wait for active writers to finish."""
        previous_handlers = {}
        def request_stop(signum: int, frame: Any) -> None:
            self.stop_requested = True
        with self.controller_lock():
            for name in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[name] = signal.signal(name, request_stop)
            try:
                self.recover()
                _write_json(self.root / "report_schema.json", REPORT_SCHEMA)
                while True:
                    self._controls()
                    self._collect()
                    candidates = [(study, task) for study, task in self.pending.items()
                                  if not (self.admission_stopped or self.runtime["pause_admission"])
                                  or self.studies[study]["admitted"]]
                    candidates.sort(key=lambda item: ({"review": 0, "repair": 1, "prepare": 2}[item[1]["phase"]],
                                                      list(self.records).index(item[0])))
                    if not self.stop_requested and not self.runtime["pause_dispatch"] and not self.resource_paused:
                        for study, task in candidates:
                            self._controls()
                            if self.stop_requested or self.runtime["pause_dispatch"] or self.resource_paused:
                                break
                            if (self.admission_stopped or self.runtime["pause_admission"]) and not self.studies[study]["admitted"]:
                                continue
                            if study in self.runtime["held_studies"]:
                                continue
                            if len(self.active) >= self.runtime["workers"]:
                                break
                            if task["retry_at"] > time.time():
                                continue
                            slot = next(str(index) for index in range(1, MAX_WORKERS + 1)
                                        if str(index) not in self.active)
                            self._launch(study, task, slot)
                    self._save()
                    if not self.active and (self.stop_requested or
                                            (self.admission_stopped and not candidates) or
                                            (not self.pending and not self.runtime["pause_dispatch"])):
                        break
                    time.sleep(self.config.poll_interval)
            except BaseException as error:
                # This path must also work when status/receipt writes failed.
                traceback.print_exc(file=sys.stderr)
                self.stop_requested = True
                for slot, worker in list(self.active.items()):
                    process = self.processes.get(slot)
                    if process is not None:
                        process.wait()
                    while process_is_alive(worker["pid"], Path(worker["job"]), worker.get("start_ticks")):
                        time.sleep(self.config.poll_interval)
                    job = Path(worker["job"])
                    close_errors = _close_job_logs(job, self.handles.pop(slot, ()))
                    if close_errors:
                        try:
                            self._write_job_receipt(job, worker, process.poll() if process is not None else None,
                                                    close_errors)
                        except OSError as receipt_error:
                            error.add_note(f"Could not record CLI log close failures in {job / 'receipt.json'}: {receipt_error!r}")
                            print(f"CLI receipt write failed: {job / 'receipt.json'}: {receipt_error!r}",
                                  file=sys.stderr, flush=True)
                    self.processes.pop(slot, None)
                    self.active.pop(slot)
                raise
            finally:
                for name, handler in previous_handlers.items():
                    signal.signal(name, handler)
            receipt = {"finished_at": _now(), "stopped": self.stop_requested,
                       "admission_stopped": self.admission_stopped, "studies": len(self.studies),
                       "states": dict(Counter(item["status"] for item in self.studies.values())),
                       "queued": len(self.pending)}
            _write_json(self.root / "controller_receipt.json", receipt)
            return receipt
