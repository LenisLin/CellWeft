"""Read-only, aggregate observation of the continuous preparation controllers."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from cellweft.data.io.writers import write_json
from cellweft.data.preparation.execution import Transport
from cellweft.data.preparation.io_coordination import io_policy_path


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _count(values: Any, key: str = "status") -> dict[str, int]:
    if not isinstance(values, list):
        return {}
    return dict(Counter(str(item.get(key, "unknown")) for item in values if isinstance(item, dict)))


def _error(value: Any) -> str | None:
    if value is None or value == "":
        return None
    # Runtime messages can contain private paths and identities.
    return "present"


def _pid_alive(pid: Any, start_ticks: Any) -> bool:
    if type(pid) is not int or type(start_ticks) is not int:
        return False
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and int(fields[19]) == start_ticks
    except (OSError, IndexError, ValueError):
        return False


def tail_lease_events(path: Path, *, limit: int = 200, max_bytes: int = 131072) -> dict:
    """Summarize a bounded tail; incomplete first/last lines are ignored."""
    if not path.exists():
        return {"events": 0, "by_slot": {}}
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        start = max(0, size - max_bytes)
        stream.seek(start)
        data = stream.read(max_bytes)
    lines = data.split(b"\n")
    if start:
        lines = lines[1:]
    if data and not data.endswith(b"\n"):
        lines = lines[:-1]
    totals: dict[str, dict[str, float | int]] = {}
    valid = 0
    for line in lines[-limit:]:
        try:
            event = json.loads(line)
            slots = event.get("slots", {})
            wait = float(event["wait_seconds"])
            hold = float(event["hold_seconds"])
            if not isinstance(slots, dict) or wait < 0 or hold < 0:
                continue
        except (ValueError, TypeError, KeyError):
            continue
        valid += 1
        for slot in slots:
            row = totals.setdefault(slot, {"events": 0, "wait_seconds": 0.0, "hold_seconds": 0.0})
            row["events"] += 1
            row["wait_seconds"] += wait
            row["hold_seconds"] += hold
    return {"events": valid, "by_slot": totals, "tail_bytes": len(data)}


def parse_mountstats(text: str, mount: str = "/mnt/NAS_21T") -> tuple[int, int] | None:
    active = False
    for line in text.splitlines():
        if line.startswith("device "):
            active = f" mounted on {mount} with fstype nfs" in line
        elif active and line.lstrip().startswith("bytes:"):
            values = line.split(":", 1)[1].split()
            if len(values) >= 6:
                return int(values[4]), int(values[5])
    return None


def parse_proc(stat: str, meminfo: str, netdev: str, mountstats: str,
               *, mount: str = "/mnt/NAS_21T") -> dict:
    cpu = next((line.split()[1:] for line in stat.splitlines() if line.startswith("cpu ")), [])
    memory = {}
    for line in meminfo.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = int(value.split()[0]) * 1024
    network = {}
    for line in netdev.splitlines():
        if ":" in line:
            name, counters = line.split(":", 1)
            values = counters.split()
            if len(values) >= 9 and name.strip() != "lo":
                network[name.strip()] = (int(values[0]), int(values[8]))
    return {"cpu": [int(value) for value in cpu], "memory": memory,
            "network": network, "nas": parse_mountstats(mountstats, mount)}


def proc_delta(previous: dict | None, current: dict, seconds: float) -> dict:
    result: dict[str, Any] = {"memory_bytes": current["memory"], "interval_seconds": seconds}
    if previous is None or seconds <= 0:
        return result
    old_cpu, cpu = previous["cpu"], current["cpu"]
    if len(old_cpu) >= 5 and len(cpu) >= 5:
        total = sum(cpu) - sum(old_cpu)
        idle = cpu[3] + cpu[4] - old_cpu[3] - old_cpu[4]
        if total > 0:
            result["cpu_utilization_percent"] = round(100 * (total - idle) / total, 2)
    network = {}
    for name, pair in current["network"].items():
        old = previous["network"].get(name)
        if old and all(new >= prior for new, prior in zip(pair, old)):
            network[name] = {"receive_bytes_per_second": round((pair[0] - old[0]) / seconds, 2),
                             "transmit_bytes_per_second": round((pair[1] - old[1]) / seconds, 2)}
    result["network"] = network
    old_nas, nas = previous["nas"], current["nas"]
    if old_nas and nas and all(new >= prior for new, prior in zip(nas, old_nas)):
        result["nas_server_bytes_per_second"] = {
            "read": round((nas[0] - old_nas[0]) / seconds, 2),
            "write": round((nas[1] - old_nas[1]) / seconds, 2)}
    return result


def _local_sample() -> dict:
    return parse_proc(Path("/proc/stat").read_text(), Path("/proc/meminfo").read_text(),
                      Path("/proc/net/dev").read_text(), Path("/proc/self/mountstats").read_text())


def _remote_node(node: dict) -> dict:
    command = [node.get("python", "python3"), "-c",
               "import json,shutil,pathlib,subprocess; "
               "m={x.split(':')[0]:int(x.split(':')[1].split()[0])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:'))}; "
               "load=float(pathlib.Path('/proc/loadavg').read_text().split()[0]); "
               "g=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8) if "
               + repr(str(node.get("id")) in {"226", "73", "72"}) + " else None; "
               "print(json.dumps({'memory_bytes':m,'cpu_load_1m':load,'disk_free_bytes':shutil.disk_usage('/').free,'disk_total_bytes':shutil.disk_usage('/').total,'gpu':g.stdout.strip().splitlines() if g and g.returncode==0 else None,'gpu_error':bool(g and g.returncode!=0)}))"]
    result = Transport.ssh(node, command, capture_output=True, text=True, timeout=20)
    data = json.loads(result.stdout)
    return {"memory_bytes": data["memory_bytes"], "cpu_load_1m": data["cpu_load_1m"],
            "disk_free_bytes": data["disk_free_bytes"], "disk_total_bytes": data["disk_total_bytes"],
            "gpu": data["gpu"], "gpu_error": data["gpu_error"]}


def remote_health(nodes: list[dict]) -> dict:
    results = {}
    with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
        futures = {pool.submit(_remote_node, node): str(node["id"]) for node in nodes}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
                results[name] = {"error_type": type(error).__name__}
    return results


def aggregate(upstream: dict, image: dict, vector: dict, discovery: dict,
              vector_plan: dict | None = None, vector_discovery: dict | None = None) -> dict:
    studies = upstream.get("studies", {})
    upstream_counts = _count(list(studies.values())) if isinstance(studies, dict) else {}
    workers = upstream.get("workers", {})
    worker_values = list(workers.values()) if isinstance(workers, dict) else []
    live = sum(_pid_alive(item.get("pid"), item.get("start_ticks")) for item in worker_values
               if isinstance(item, dict))
    scopes = image.get("accepted_scopes", [])
    scope_comparable = bool(scopes) and all(isinstance(item, dict) and
                                            isinstance(item.get("scope_key"), str)
                                            for item in scopes)
    scope_keys: dict[str, set[str]] = {}
    for item in scopes:
        if isinstance(item, dict) and isinstance(item.get("scope_key"), str):
            scope_keys.setdefault(str(item["study"]), set()).add(item["scope_key"])
    coverage = {"source_available_images": None, "input_exported_images": None,
                "accepted_scopes": len(scopes),
                "accepted_images_in_scopes": sum(int(item.get("accepted_images", 0)) for item in scopes),
                "sample_linked_images_in_scopes": sum(int(item.get("sample_linked_images", 0)) for item in scopes),
                "patient_linked_images_in_scopes": sum(int(item.get("patient_linked_images", 0)) for item in scopes)}
    image_counts = _count(image.get("outcomes", []))
    vector_counts = _count(vector.get("outcomes", []))
    historical = sum(item.get("status") == "complete" for item in vector.get("outcomes", [])
                     if isinstance(item, dict))
    current = 0
    vector_scan_complete = bool(vector_discovery and vector_discovery.get('complete') is True)
    if discovery.get("complete") is True and vector_scan_complete and vector.get("status") == "complete" and vector_plan:
        tasks = vector_plan.get("studies", [])
        expected = {task.get("study"): {scope.get("scope_key") for scope in task.get("scopes", [])}
                    for task in tasks if isinstance(task, dict)}
        current = sum(scope_comparable and item.get("status") == "complete"
                      and bool(scope_keys.get(item.get("study")))
                      and scope_keys[item["study"]] == expected.get(item.get("study"))
                      and set(item.get("scopes", [])) == expected.get(item.get("study"))
                      for item in vector.get("outcomes", []) if isinstance(item, dict))
    gaps = image.get("discovery_gaps", [])
    # Reasons may contain identities; only count dependency gaps.
    gap_counts = {"discovery_dependency_gaps": len(gaps),
                  "product_dependency_gaps": sum(len(item.get("gaps", []))
                                                 for item in image.get("outcomes", [])
                                                 if isinstance(item, dict) and isinstance(item.get("gaps"), list))}
    idle_backlog = any(image_counts.get(stage, 0) for stage in
                       ("queued", "waiting_cpu", "computing", "computed",
                        "waiting_publish", "waiting_io"))
    admission = image.get('admission', {})
    admission_counts = {'preparing': bool(admission.get('preparing_study')),
                        'pending_scopes': int(admission.get('pending_scopes', 0)),
                        'prepared_scopes': int(admission.get('prepared_scopes', 0))}
    idle_backlog |= admission_counts['preparing'] or admission_counts['pending_scopes'] > 0
    return {"upstream": {"studies": len(studies), "stages": upstream_counts,
                         "queued": upstream.get("queued", 0), "cli_live": live,
                         "cli_recorded": len(worker_values), "control_error": _error(upstream.get("control_error"))},
            "discovery": {"complete": discovery.get("complete") is True,
                          "scanned_studies": discovery.get("scanned_studies", 0),
                          "accepted_scopes": discovery.get("accepted_scopes", 0),
                          "gaps": discovery.get("gaps", 0)},
            "coverage": coverage,
            "images": {"total_current": image.get("total", 0), "stages": image_counts,
                       "admission": admission_counts,
                       "published_complete_this_run": image_counts.get("complete", 0),
                       "computed_unpublished": image_counts.get("computed", 0),
                       "gaps": gap_counts,
                       "error_count": sum(bool(item.get("error")) for item in image.get("outcomes", [])
                                          if isinstance(item, dict)),
                       "discovery_error": _error(image.get("discovery_error"))},
            "vectors": {"historical_complete_records": historical,
                        "current_scope_verified_complete": current,
                        "discovery_complete": vector_scan_complete,
                        "current_scope_comparison_available": scope_comparable,
                        "stages": vector_counts, "gap_count": len(vector.get("gaps", []))},
            "scheduling": {"has_backlog": idle_backlog,
                           "eligible_for_segmentation": None,
                           "eligible_insufficient": None,
                           "eligible_status": "unknown_from_controller_status",
                           "dependency_gap_count": len(gaps)}}


class PreparationMonitor:
    def __init__(self, config: dict):
        required = {"upstream_run", "image_run", "vector_run", "lock_root", "output_root",
                    "gpu_nodes", "cpu_nodes"}
        missing = required - config.keys()
        if missing:
            raise ValueError(f"missing monitor config keys: {', '.join(sorted(missing))}")
        self.config = config
        self.previous: dict | None = None
        self.sample_time: float | None = None
        self.remote: dict = {"enabled": bool(config.get("remote_enabled", False)), "nodes": {}}
        self.remote_time = 0.0
        self.local_memory_paused = False

    def snapshot(self) -> dict:
        config = self.config
        errors = {}
        payloads = {}
        paths = {"upstream": Path(config["upstream_run"]) / "status.json",
                 "image": Path(config["image_run"]) / "status.json",
                 "vector": Path(config["vector_run"]) / "embedding_status.json",
                 "discovery": Path(config["image_run"]) / "discovery_status.json",
                 "vector_discovery": Path(config["vector_run"]) / "discovery_status.json",
                 "vector_plan": Path(config["vector_run"]) / "plan.json",
                 "policy": io_policy_path(Path(config["lock_root"]))}
        for name, path in paths.items():
            try:
                payloads[name] = _json(path)
            except (OSError, ValueError) as error:
                errors[name] = type(error).__name__
                payloads[name] = {}
        status = aggregate(payloads["upstream"], payloads["image"], payloads["vector"],
                           payloads["discovery"], payloads["vector_plan"],payloads['vector_discovery'])
        if config.get('auxiliary_cli_manifest'):
            try:
                auxiliary = _json(Path(config['auxiliary_cli_manifest']))
                live = sum(_pid_alive(item.get('pid'),item.get('start_ticks'))
                           for item in auxiliary.values() if isinstance(item,dict))
                status['upstream']['auxiliary_cli_live'] = live
                status['upstream']['total_cli_live'] = status['upstream']['cli_live']+live
            except (OSError,ValueError) as error:
                errors['auxiliary_cli_manifest'] = type(error).__name__
        now = time.monotonic()
        if config.get('local_workload'):
            from cellweft.data.preparation.local_resources import workload_memory, pressure_state
            limits = config['local_workload']
            roots = _json(Path(limits['roots_file']))['roots']
            roots += list(payloads['upstream'].get('workers', {}).values())
            if config.get('auxiliary_cli_manifest'):
                roots += list(_json(Path(config['auxiliary_cli_manifest'])).values())
            workload = workload_memory(roots)
            budget = int(limits['memory_budget_bytes'])
            self.local_memory_paused = pressure_state(
                workload['rss_sum_bytes'], budget=budget, paused=self.local_memory_paused)
            workload.update(memory_budget_bytes=budget, admission_paused=self.local_memory_paused,
                            cpu_affinity=limits['cpu_affinity'],
                            memory_control='RSS admission budget; no delegated memory cgroup',
                            observed_at=datetime.now(timezone.utc).isoformat())
            if limits.get('gpu_memory_budget_bytes'):
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
                workload['gpu'] = gpu.stdout.strip().splitlines() if gpu.returncode == 0 else None
                workload['gpu_error'] = gpu.returncode != 0
                workload['gpu_memory_budget_bytes'] = limits['gpu_memory_budget_bytes']
            write_json(Path(limits['status_path']), workload)
            status['local_workload'] = workload
        try:
            sample = _local_sample()
            status["local_resources"] = proc_delta(self.previous, sample,
                                                    now - self.sample_time if self.sample_time else 0)
            self.previous, self.sample_time = sample, now
            if sample["nas"] is None:
                errors["nas_mountstats"] = "mount_not_found"
        except (OSError, ValueError) as error:
            errors["local_resources"] = type(error).__name__
        if config.get("remote_enabled", False) and (not self.remote_time or now - self.remote_time >= 120):
            self.remote = {"enabled": True, "sampled_at": datetime.now(timezone.utc).isoformat(),
                           "nodes": remote_health(config["gpu_nodes"] + config["cpu_nodes"])}
            self.remote_time = now
        status["remote_resources"] = self.remote
        gpu_rows = [row for node in self.remote["nodes"].values() if isinstance(node, dict)
                    for row in (node.get("gpu") or [])]
        gpu_rows += status.get('local_workload', {}).get('gpu') or []
        gpu_util = []
        for row in gpu_rows:
            try:
                gpu_util.append(float(row.split(",", 1)[0].strip()))
            except ValueError:
                continue
        gpu_idle = max(gpu_util) < 10 if gpu_util else None
        status["scheduling"]["gpu_idle_observed"] = gpu_idle
        status["scheduling"]["gpu_idle_with_backlog"] = (
            status["scheduling"]["has_backlog"] and gpu_idle if gpu_idle is not None else None)
        try:
            leases = tail_lease_events(Path(config["lock_root"]) / "io_slot_events.jsonl")
        except OSError as error:
            errors["io_events"] = type(error).__name__
            leases = {"events": 0, "by_slot": {}}
        status["io"] = {"policy": {key: payloads["policy"].get(key)
                                   for key in ("nas_read", "nas_write") if key in payloads["policy"]},
                        "completed_leases_tail": leases}
        status["observed_at"] = datetime.now(timezone.utc).isoformat()
        status["read_errors"] = errors
        write_json(Path(config["output_root"]) / "target_status.json", status)
        with (Path(config['output_root'])/'monitor_events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(status,separators=(',',':'))+'\n')
        return status
