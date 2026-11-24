"""Observe the shared local workload and stop new admission under RAM pressure."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def workload_memory(roots: list[dict], *, proc_root: Path = Path('/proc')) -> dict:
    """Count root identities and descendants once, using conservative summed RSS."""
    processes = {}
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if fields[0] != 'Z':
            processes[int(directory.name)] = (int(fields[1]), int(fields[19]),
                                               int(fields[21]) * os.sysconf('SC_PAGE_SIZE'))
    owned = {item['pid'] for item in roots if item.get('pid') in processes
             and processes[item['pid']][1] == item.get('start_ticks')}
    while True:
        expanded = owned | {pid for pid, row in processes.items() if row[0] in owned}
        if expanded == owned:
            break
        owned = expanded
    return {'processes': len(owned), 'rss_sum_bytes': sum(processes[pid][2] for pid in owned)}


def pressure_state(rss: int, *, budget: int, paused: bool = False) -> bool:
    """Reserve 8 GiB for admitted work; resume after a further 8 GiB is freed."""
    if budget < 32 * 1024**3:
        raise ValueError('The shared local memory budget must be at least 32 GiB')
    threshold = budget - (16 if paused else 8) * 1024**3
    return rss >= threshold


def admission_paused(config: Mapping[str, Any]) -> bool:
    """Read an optional monitor receipt; a configured missing receipt pauses admission."""
    value = config.get('local_resource_status')
    if not value:
        return False
    path = Path(value)
    if not path.is_file():
        return True
    status = json.loads(path.read_text())
    if type(status.get('admission_paused')) is not bool:
        raise ValueError('Local resource receipt requires boolean admission_paused')
    return status['admission_paused']
