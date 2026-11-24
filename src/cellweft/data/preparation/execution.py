"""Reviewed-image scheduling with separate compute and NAS publication stages."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import inspect
import queue
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid

import pandas as pd

from cellweft.data.io.publication import resource_lock
from cellweft.data.io.readers import open_source
from cellweft.data.contracts import SharedImageResult
from cellweft.data.io.writers import write_json, write_table, publish_image, json_value
from cellweft.data.preparation.configuration import load_settings
from cellweft.data.preparation.inputs import load_reviewed_study_inputs, assert_reviewed_binding
from cellweft.data.preparation.pipeline import prepare_image, preparation_request, file_identity, source_identity
from cellweft.data.preparation.remote_worker import limited_slots
from cellweft.data.preparation.io_coordination import formal_write_lease
from cellweft.data.preparation.segmentation import CellposeExecutor
from cellweft.data.preparation.capacity import (
    CapacityGap, CapacityReservations, image_work_bytes, source_copy_bytes, tiled_tiff_bytes,
)
from cellweft.data.preparation.deferred import DeferredStore


class CPUSlots:
    """Prefer ready postprocessing over new imports within the CPU budget."""
    def __init__(self,count: int):
        self.available=count;self.priority_waiters=0;self.condition=threading.Condition()

    def acquire(self,priority: bool = False) -> None:
        """Wait for a computation slot; mask returns take priority over new inputs."""
        with self.condition:
            if priority:self.priority_waiters+=1
            try:
                while self.available==0 or (not priority and self.priority_waiters):self.condition.wait()
                self.available-=1
            finally:
                if priority:self.priority_waiters-=1

    def release(self) -> None:
        """Release a slot while a remote task executes or local work completes."""
        with self.condition:
            self.available+=1;self.condition.notify_all()

    def __enter__(self):self.acquire();return self
    def __exit__(self,*_):self.release()


def take_pending_job(pending: list, active: set, last_study: str | None):
    """Interleave ready studies while preferring reviewed mask bindings."""
    active_counts={}
    for study,_ in active:active_counts[study]=active_counts.get(study,0)+1
    least=min(active_counts.get(job[0]['study'],0) for job in pending)
    candidates=[job for job in pending if active_counts.get(job[0]['study'],0)==least]
    known=[job for job in candidates if job[0]['mask_bindings'][job[1]]['status']!='unknown']
    candidates=known or candidates
    choice=next((job for job in candidates if job[0]['study']!=last_study),candidates[0])
    pending.remove(choice)
    return choice


class Transport:
    """SSH streams through one shared-mount gateway, with remote transfer leases."""

    def __init__(self, gateway: dict, shared_root: Path):
        self.gateway, self.root = gateway, shared_root
        self.lock_root = shared_root/'transfer_locks'
        self.relays=[];self.relay_lock=threading.Lock()

    def configure_relays(self, relays: list[dict]) -> None:
        """Configure verified shared mounts, retaining one transfer-lock domain."""
        for entry in relays:
            root=Path(entry['root'])
            if not root.is_absolute():raise ValueError('Transfer relay root must be absolute')
            relay=Transport(entry.get('gateway',self.gateway),root)
            relay.lock_root=self.lock_root
            self.relays.append({'id':entry['id'],'transport':relay,'nodes':entry.get('nodes'),
                                'bytes':0,'tasks':0})

    @contextmanager
    def relay(self,node_id: str,amount: int):
        """Balance in-flight payloads across mounts accessible on the target node."""
        with self.relay_lock:
            choices=[r for r in self.relays if r['nodes'] is None or node_id in r['nodes']]
            if self.relays and not choices:raise ValueError(f'No verified relay for node {node_id}')
            selected=min(choices,key=lambda r:(r['bytes'],r['tasks'])) if choices else None
            if selected:selected['bytes']+=amount;selected['tasks']+=1
        try:yield selected['transport'] if selected else self
        finally:
            if selected:
                with self.relay_lock:selected['bytes']-=amount;selected['tasks']-=1

    @staticmethod
    def ssh(node: dict, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Execute a quoted remote argv using established SSH credentials."""
        kwargs.setdefault('timeout',900)
        return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
            '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-p',str(node.get('port',4081)),
            f"{node.get('user','lyx')}@{node['host']}",shlex.join(command)],check=True,**kwargs)

    def command(self, direction: str, path: Path, size: int | None = None) -> list[str]:
        """Build the remote transfer wrapper command without shell interpolation."""
        g=self.gateway
        cmd=['env',f"PYTHONPATH={g['code_root']}/src",g['python'],
             f"{g['code_root']}/scripts/data/preparation/run_transfer.py",direction,
             '--path',str(path),'--node',g['id'],'--lock-root',str(self.lock_root),
             '--allowed-root',str(self.root)]
        if size is not None:cmd+=['--size',str(size)]
        return cmd

    def upload(self, source: Path, destination: Path, *, timeout: float = 900) -> None:
        """Transfer a closed asset; the receiver validates length and atomically renames."""
        with source.open('rb') as incoming:
            self.ssh(self.gateway,self.command('upload',destination,source.stat().st_size),
                     stdin=incoming,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=timeout)

    def download(self, source: Path, destination: Path, *, timeout: float = 900) -> None:
        """Accept a closed remote file only when the SSH stream completes successfully."""
        destination.parent.mkdir(parents=True,exist_ok=True)
        partial=destination.with_name(destination.name+'.partial')
        with partial.open('wb') as output:
            self.ssh(self.gateway,self.command('download',source),stdout=output,stderr=subprocess.PIPE,timeout=timeout)
        partial.replace(destination)

    def status(self, root: Path, *, timeout: float = 30) -> dict | None:
        """Read one small done/failure envelope; an absent envelope means still running."""
        code="import pathlib,json,sys;p=pathlib.Path(sys.argv[1]);f=p/'failure.json';d=p/'done.json';print(json.dumps({'failure':json.loads(f.read_text())}) if f.exists() else d.read_text() if d.exists() else 'null')"
        result=self.ssh(self.gateway,[self.gateway['python'],'-c',code,str(root)],capture_output=True,text=True,timeout=timeout)
        return json.loads(result.stdout)


class WorkerPool:
    """Start one persistent process per configured slot and retain exclusive assignments."""

    def __init__(self, nodes: list[dict], mode: str, transport: Transport, run_root: Path):
        self.mode,self.transport,self.run_root=mode,transport,run_root
        self.slots=queue.Queue()
        self.processes=[]
        self.queues=[]
        self.handles=[]
        self.assignments=[]
        self.state_lock=threading.Lock();self.usable_slots=0
        try:
            for node in nodes:
                for index in range(node['slots']):
                    launch_id=uuid.uuid4().hex
                    remote_queue=transport.root/run_root.name/mode/node['id']/str(index)
                    command=['env',f"PYTHONPATH={node['code_root']}/src",'PYTHONDONTWRITEBYTECODE=1',
                        'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
                        'HF_HUB_OFFLINE=1',f"MPLCONFIGDIR={node['cache_root']}/matplotlib",
                        node['python'],f"{node['code_root']}/scripts/data/preparation/run_remote_worker.py",
                        '--queue',str(remote_queue),'--cache',str(Path(node['cache_root'])/run_root.name/mode/str(index)),
                        '--mode',mode,'--node',node['id'],'--lock-root',str(transport.lock_root),
                        '--launch-id',launch_id]
                    if mode=='gpu':command+=['--precision',node['precision'],'--batch-size',str(node['batch_size']),
                        '--weights',node['weights'],'--device',f'cuda:{index}']
                    for relay in getattr(transport,'relays',[]):
                        if relay['nodes'] is None or node['id'] in relay['nodes']:
                            command+=['--input-root',str(relay['transport'].root),'--output-root',str(relay['transport'].root)]
                    log=(run_root/f'{mode}_{node["id"]}_{index}.log').open('ab')
                    proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30',
                        '-o','ServerAliveCountMax=3','-p',str(node.get('port',4081)),f"lyx@{node['host']}",
                        shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
                    slot={'node':node,'index':index,'queue':remote_queue,'process':proc}
                    self.assignments.append(slot)
                    self.slots.put(slot);self.usable_slots+=1;self.processes.append(proc);self.queues.append(remote_queue);self.handles.append(log)
                    deadline=time.monotonic()+60
                    recovery_deadline=deadline+840
                    while True:
                        if proc.poll() is not None:raise RuntimeError(f"{mode}/{node['id']}/{index}: worker startup failed")
                        remaining=deadline-time.monotonic()
                        if remaining<=0:
                            detail='fifteen minutes of recovery' if deadline==recovery_deadline else 'sixty seconds'
                            raise TimeoutError(f'Remote worker did not become ready within {detail}')
                        code="import pathlib,sys;p=pathlib.Path(sys.argv[1]);print(p.read_text() if p.exists() else 'null')"
                        # Query on the writing node to avoid another NFS client's
                        # negative-entry cache delaying the startup handshake.
                        status=transport.ssh(node,[node['python'],'-c',code,str(remote_queue/'worker.json')],
                                             capture_output=True,text=True,timeout=min(15,remaining))
                        worker=json.loads(status.stdout) or {}
                        if worker.get('launch_id')==launch_id and worker.get('recovering'):
                            deadline=recovery_deadline
                        if worker.get('ready') and worker.get('launch_id')==launch_id:break
                        time.sleep(.5)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def lease(self, node_id: str | tuple[str, ...] | None = None):
        """Assign a physical slot until its compute completion is verified."""
        while True:
            with self.state_lock:
                if self.usable_slots==0:raise RuntimeError('No healthy remote worker slots remain')
            try:
                slot=self.slots.get(timeout=1)
                allowed=(node_id,) if isinstance(node_id,str) else node_id
                if allowed is not None and slot['node']['id'] not in allowed:
                    self.slots.put(slot)
                    with self.state_lock:
                        if not any(s['node']['id'] in allowed and not s.get('quarantined') for s in self.assignments):
                            raise RuntimeError(f'No healthy worker remains on node {node_id}')
                    time.sleep(.05)
                    continue
                break
            except queue.Empty:continue
        healthy=not slot.get('quarantined',False)
        released=False
        physical_slot=slot
        def release_compute():
            nonlocal released
            if not released:
                released=True
                self.slots.put(physical_slot)
        # Lease state belongs to this caller; a later lessee never shares it.
        slot={**physical_slot,'_release_compute':release_compute}
        try:
            if not healthy:raise RuntimeError('Remote slot remains quarantined')
            if slot['process'].poll() is not None:
                healthy=False;raise RuntimeError('Remote worker exited before assignment')
            yield slot
        except (TimeoutError,subprocess.SubprocessError,BrokenPipeError):
            if not released:
                healthy=False
                signal=self.run_root/f'STOP_{self.mode}';signal.write_text('quarantined slot\n')
                try:self.transport.upload(signal,slot['queue']/'STOP',timeout=30)
                except subprocess.SubprocessError as error:
                    write_json(self.run_root/f'quarantined_{self.mode}_{slot["node"]["id"]}_{slot["index"]}.json',
                               {'error':str(error),'recovery':'worker controller lease expires after shutdown'})
            raise
        finally:
            if not released:
                physical_slot['quarantined']=not healthy
                if healthy:self.slots.put(physical_slot)
                else:
                    with self.state_lock:self.usable_slots-=1

    @staticmethod
    def _verified_done(candidate: dict, state: dict, task: dict, slot: dict) -> None:
        """Return a leased slot only after a complete, matching done envelope."""
        receipt=state.get('receipt')
        names=receipt.get('files') if isinstance(receipt,dict) else None
        valid=(state.get('state')=='done' and isinstance(receipt,dict) and
               receipt.get('dispatch_id')==candidate['dispatch_id'] and
               receipt.get('task_id')==task['task_id'] and isinstance(names,list) and
               'result.json' in names and all(isinstance(name,str) and name and
                   Path(name).name==name for name in names))
        if not valid:
            # An unidentified completion cannot prove that this worker is idle.
            raise BrokenPipeError('Remote completion envelope identity or files differ')
        release=slot.get('_release_compute')
        if release is not None:release()

    def _matching_attempts(self, task: dict) -> list[dict]:
        """Find local dispatch records for this task without walking payload data."""
        dispatch_root = self.run_root/'dispatch'
        if not dispatch_root.is_dir():
            return []
        records=[]
        for assignment_path in sorted(dispatch_root.glob('*/assignment.json')):
            try:
                assignment=json.loads(assignment_path.read_text())
            except (OSError,ValueError):
                # A partial controller record cannot identify a task safely.
                continue
            if assignment.get('mode') != self.mode or assignment.get('task_id') != task.get('task_id'):
                continue
            dispatch=assignment.get('dispatch_id',assignment.get('dispatchid',assignment.get('dispatch')))
            node=assignment.get('node')
            if isinstance(node,dict):
                node=node.get('id')
            slot=assignment.get('slot')
            if not isinstance(dispatch,str) or not dispatch or Path(dispatch).name != dispatch:
                raise ValueError(f'Invalid dispatch identity in {assignment_path}')
            if not isinstance(node,(str,int)) or slot is None:
                raise ValueError(f'Incomplete dispatch assignment in {assignment_path}')
            dispatch=str(dispatch)
            queue_value=assignment.get('queue')
            if queue_value is None:
                control_root=(self.transport.root/self.run_root.name/self.mode/
                              str(node)/str(slot)/dispatch)
            else:
                control_root=Path(queue_value)
                if not control_root.is_absolute():
                    raise ValueError(f'Dispatch queue must be absolute in {assignment_path}')
            data_value=assignment.get('data_root')
            payload=Path(data_value) if data_value is not None else control_root
            if not payload.is_absolute():
                raise ValueError(f'Dispatch data_root must be absolute in {assignment_path}')
            records.append({'assignment':assignment,'local':assignment_path.parent,
                            'dispatch_id':dispatch,'node':str(node),'slot':str(slot),
                            'queue':control_root,'payload':payload})
        records.sort(key=lambda record: record['local'].stat().st_mtime_ns,reverse=True)
        return records

    def _relay_for_attempt(self, payload: Path):
        """Select the configured relay owning an old attempt's declared payload."""
        candidates=[self.transport]
        candidates.extend(entry['transport'] for entry in getattr(self.transport,'relays',[]))
        ordered=sorted(candidates,key=lambda item:len(str(Path(item.root).resolve())),reverse=True)
        resolved=payload.resolve()
        for relay in ordered:
            try:
                resolved.relative_to(Path(relay.root).resolve())
            except ValueError:
                continue
            return relay
        raise ValueError(f'Existing dispatch payload is outside configured relay roots: {payload}')

    def _attempt_state(self, root: Path, *, timeout: float = 30) -> dict:
        """Read a small completion envelope, then distinguish ready/running/missing."""
        status=self.transport.status(root,timeout=timeout)
        if status is not None:
            if not isinstance(status,dict):
                raise ValueError('Remote completion envelope is not an object')
            if 'failure' in status:
                return {'state':'failure','receipt':status['failure']}
            return {'state':'done','receipt':status}
        gateway=getattr(self.transport,'gateway',{})
        python=gateway.get('python') if isinstance(gateway,dict) else None
        if not python:
            raise ValueError('Remote transport gateway has no Python executable')
        code=("import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); "
              "state='missing'; value=None; "
              "f=p/'failure.json'; d=p/'done.json'; r=p/'running.json'; q=p/'ready.json'; "
              "state='failure' if f.exists() else 'done' if d.exists() else 'running' if r.exists() else 'ready' if q.exists() else 'missing'; "
              "marker=f if state=='failure' else d if state=='done' else r if state=='running' else q if state=='ready' else None; "
              "value=json.loads(marker.read_text()) if marker is not None else None; "
              "print(json.dumps({'state':state,'receipt':value}))")
        result=self.transport.ssh(self.transport.gateway,[python,'-c',code,str(root)],
                                  capture_output=True,text=True,timeout=timeout)
        value=json.loads(result.stdout)
        if not isinstance(value,dict) or value.get('state') not in {'missing','ready','running','done','failure'}:
            raise ValueError('Remote attempt state is invalid')
        return value

    def _write_resume(self, candidate: dict, task: dict, state: dict, action: str) -> None:
        """Record controller recovery decisions beside the durable assignment."""
        receipt={'mode':self.mode,'task_id':task['task_id'],'dispatch_id':candidate['dispatch_id'],
                 'queue':str(candidate['queue']),'data_root':str(candidate['payload']),
                 'state':state['state'],'action':action,'updated_at':time.time(),
                 'attempt':state.get('receipt')}
        write_json(candidate['local']/'resume.json',receipt)

    def _download_attempt(self, candidate: dict, relay, state: dict, task: dict,
                          destination: Path, remaining) -> dict:
        """Download and validate a complete attempt without removing its source."""
        receipt=state.get('receipt')
        if not isinstance(receipt,dict):
            raise ValueError('Remote completion envelope is missing')
        if receipt.get('dispatch_id') != candidate['dispatch_id'] or receipt.get('task_id') != task['task_id']:
            raise ValueError('Received an obsolete remote attempt')
        names=receipt.get('files')
        if not isinstance(names,list) or not names:
            raise ValueError('Remote completion has no returned assets')
        destination.mkdir(parents=True,exist_ok=True)
        for name in names:
            if not isinstance(name,str) or not name or Path(name).name != name:
                raise ValueError('Unexpected returned asset name')
            relay.download(candidate['payload']/'output'/name,destination/name,timeout=remaining(900))
        result_path=destination/'result.json'
        if not result_path.is_file():
            raise ValueError('Remote result receipt is missing')
        result=json.loads(result_path.read_text())
        if result.get('task_id') != task['task_id']:
            raise ValueError('Result task identity differs')
        if self.mode=='gpu' and result.get('request') != task.get('request'):
            raise ValueError('GPU result request differs from the submitted request')
        if task.get('kind')=='normalization' and any(result.get(key)!=task.get(key) for key in
                ('kind','study','image_id','image_request')):
            raise ValueError('Normalization result identity differs from the submitted request')
        return result

    def execute(self, slot: dict, task: dict, files: dict[str,Path], destination: Path,
                phase_callback=None) -> dict:
        """Reuse a matching attempt before staging a new dispatch."""
        if slot.get('quarantined'):raise RuntimeError('Remote slot quarantined after an unresolved transfer/timeout')
        deadline=time.monotonic()+6*3600
        def remaining(limit):
            seconds=deadline-time.monotonic()
            if seconds<=0:raise TimeoutError('Remote task exceeded six hours')
            return min(limit,seconds)

        # A controller restart may leave a completed dispatch in the local
        # ledger. Reuse it before creating another attempt for the same task.
        for candidate in self._matching_attempts(task):
            relay=self._relay_for_attempt(candidate['payload'])
            state=self._attempt_state(candidate['queue'],timeout=remaining(30))
            if state['state']=='missing':
                self._write_resume(candidate,task,state,'missing')
                continue
            if state['state'] in {'ready','running'}:
                self._write_resume(candidate,task,state,'wait')
                while state['state'] in {'ready','running'}:
                    if phase_callback:
                        phase_callback('remote_compute' if state['state']=='running' else 'remote_queued',
                                       candidate['node'])
                    if slot['process'].poll() is not None:
                        raise BrokenPipeError('Remote worker exited while recovering an existing task')
                    if time.monotonic()>deadline:
                        raise TimeoutError(f'Remote task exceeded six hours: {candidate["dispatch_id"]}')
                    time.sleep(5)
                    state=self._attempt_state(candidate['queue'],timeout=remaining(30))
                if state['state']=='missing':
                    self._write_resume(candidate,task,state,'missing')
                    continue
            if state['state']=='failure':
                failure=state.get('receipt') or {}
                recovery=failure.get('recovery') if isinstance(failure,dict) else None
                recoverable=isinstance(failure,dict) and (failure.get('recoverable') is True or
                    isinstance(recovery,dict) and recovery.get('action')=='requeue')
                self._write_resume(candidate,task,state,'requeue' if recoverable else 'failed')
                if recoverable:
                    continue
                error=failure.get('error','remote task failed') if isinstance(failure,dict) else 'remote task failed'
                if self.mode=='gpu':raise BrokenPipeError(error)
                raise RuntimeError(error)
            if state['state']=='done':
                self._verified_done(candidate,state,task,slot)
                self._write_resume(candidate,task,state,'reuse_done')
                if phase_callback:phase_callback('return_transfer',candidate['node'])
                result=self._download_attempt(candidate,relay,state,task,destination,remaining)
                write_json(candidate['local']/'completion.json',state['receipt'])
                self._write_resume(candidate,task,state,'reuse_done')
                return result

        dispatch=uuid.uuid4().hex
        root=slot['queue']/dispatch
        local=self.run_root/'dispatch'/dispatch;local.mkdir(parents=True)
        write_json(local/'assignment.json',{'mode':self.mode,'node':slot['node']['id'],
                   'slot':slot['index'],'task_id':task['task_id'],'dispatch_id':dispatch})
        write_json(local/'task.json',task)
        files={**files,'task.json':local/'task.json'}
        amount=sum(path.stat().st_size for path in files.values())
        relay_context=self.transport.relay(slot['node']['id'],amount) if hasattr(self.transport,'relay') else nullcontext(self.transport)
        with relay_context as relay:
            payload=relay.root/self.run_root.name/self.mode/slot['node']['id']/str(slot['index'])/dispatch if relay is not self.transport else root
            for name,path in files.items():relay.upload(path,payload/name,timeout=remaining(900))
            ready={'dispatch_id':dispatch,'files':list(files)}
            if payload!=root:ready['data_root']=str(payload)
            write_json(local/'ready.json',ready)
            write_json(local/'assignment.json',{'mode':self.mode,'node':slot['node']['id'],
                'slot':slot['index'],'task_id':task['task_id'],'dispatch_id':dispatch,
                'queue':str(root),'data_root':str(payload)})
            self.transport.upload(local/'ready.json',root/'ready.json',timeout=remaining(30))
            if phase_callback:phase_callback('remote_queued',slot['node']['id'])
            while True:
                if slot['process'].poll() is not None:raise BrokenPipeError('Remote worker exited during task')
                status=self.transport.status(root,timeout=remaining(30))
                if status is not None:break
                if phase_callback:
                    phase=self._attempt_state(root,timeout=remaining(30))['state']
                    phase_callback('remote_compute' if phase=='running' else 'remote_queued',
                                   slot['node']['id'])
                if time.monotonic()>deadline:raise TimeoutError(f'Remote task exceeded six hours: {dispatch}')
                time.sleep(5)
            if not isinstance(status,dict):
                raise BrokenPipeError('Remote completion envelope is not an object')
            write_json(local/'completion.json',status)
            if 'failure' in status:
                failure=status['failure'] if isinstance(status['failure'],dict) else {}
                if failure.get('recoverable') is True:
                    error=failure.get('error','remote task needs an I/O retry')
                    raise OSError(error)
                if self.mode=='gpu':raise BrokenPipeError(status['failure']['error'])
                raise RuntimeError(status['failure']['error'])
            completed={'state':'done','receipt':status}
            self._verified_done({'dispatch_id':dispatch},completed,task,slot)
            if phase_callback:phase_callback('return_transfer',slot['node']['id'])
            result=self._download_attempt({'dispatch_id':dispatch,'payload':payload},relay,
                                          completed,task,destination,remaining)
            relay.ssh(relay.gateway,['rm','-rf','--',str(payload)],stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,timeout=remaining(30))
            if payload!=root:self.transport.ssh(self.transport.gateway,['rm','-rf','--',str(root)],
                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=remaining(30))
            return result

    def close(self) -> None:
        """Stop owned workers and wait; a failed SSH session cannot leave an unowned queue."""
        signal=self.run_root/f'STOP_{self.mode}';signal.write_text('controller stopped\n')
        failures=[]
        for path in self.queues:
            try:self.transport.upload(signal,path/'STOP',timeout=30)
            except subprocess.SubprocessError as error:
                failures.append({'queue':str(path),'error':str(error),'recovery':'controller lease expires within five minutes'})
        if failures:write_json(self.run_root/f'{self.mode}_stop_failures.json',{'failures':failures})
        for process in self.processes:
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.terminate();process.wait(timeout=15)
        for handle in self.handles:handle.close()


class RemoteSegmentation:
    """Match the existing executor interface while releasing the local CPU slot in flight."""
    core_size=1024
    halo=128
    match_iou=.5

    def __init__(self,pool: WorkerPool,slot: dict,cpu_slots: CPUSlots):
        self.pool,self.slot,self.cpu_slots=pool,slot,cpu_slots

    def execute(self,task: dict) -> Path:
        """Run one bound segmentation attempt on its configured persistent GPU worker."""
        self.cpu_slots.release()
        try:
            if self.slot.get('lease_on_execute'):
                with self.pool.lease(self.slot['node']['id']) as slot:
                    self.pool.execute(slot,task,{'signal.tiff':Path(task['input_path'])},Path(task['result_path']).parent)
            else:
                self.pool.execute(self.slot,task,{'signal.tiff':Path(task['input_path'])},Path(task['result_path']).parent)
        finally:self.cpu_slots.acquire(priority=True)
        return Path(task['result_path'])


class LocalSegmentation:
    """One persistent local model, serialized across threads and cooperating processes."""
    core_size=1024
    halo=128
    match_iou=.5

    def __init__(self, config: dict, cpu_slots: CPUSlots, cache: Path):
        self.config,self.cpu_slots=config,cpu_slots
        self.cache=cache/'LOCAL'
        self.lock=threading.Lock()
        self.executor=CellposeExecutor(config['device'],precision=config['precision'],
            memory_budget_bytes=config['memory_budget_bytes'])

    def execute(self, task: dict) -> Path:
        if (task['precision'] != self.config['precision'] or
                task['parameters']['batch_size'] != self.config['batch_size'] or
                Path(task['weights']).resolve() != Path(self.config['weights']).resolve()):
            raise ValueError('Local GPU task differs from its assigned slot configuration')
        self.cpu_slots.release()
        try:
            with self.lock,resource_lock(self.cache,name='gpu.lock'):
                execution_task = task
                if self.config.get('weights_cache'):
                    # The task's request/content identity stays bound to the NAS source.
                    execution_task = {**task, 'weights': self.config['weights_cache']}
                return self.executor.execute(execution_task)
        finally:
            self.cpu_slots.acquire(priority=True)


class RemoteCPU:
    """Offload cell geometry, full-instance quantification and neighbor/crop indexing."""

    def __init__(self,pool: WorkerPool,cpu_slots: CPUSlots):
        self.pool,self.cpu_slots=pool,cpu_slots

    def __call__(self,*,record,source,image_path,mask_path,mask_identity,channels,staging,settings) -> dict[str,str]:
        """Return reopened CPU products in pipeline staging; formal publishing remains local."""
        identity={'study':record.study,'image_id':record.image_id,'mask_identity':mask_identity,
                  'source':source_identity(source),
                  'channels':json_value(channels.astype(object).where(pd.notna(channels),None).to_dict('records')),
                  'block_size':settings.block_size}
        checkpoint=settings.cache_dir/'remote_cpu_task.json'
        previous=json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
        task_id=previous['task_id'] if previous.get('identity')==identity else uuid.uuid4().hex
        write_json(checkpoint,{'task_id':task_id,'identity':identity})
        channels_path=staging/'cpu_channels.parquet';write_table(channels_path,channels)
        files={'image.tiff':image_path,'mask.tiff':mask_path,'channels.parquet':channels_path}
        task={'task_id':task_id,'study':record.study,'image_id':record.image_id,'mask_identity':mask_identity,
              'image_path':'image.tiff','mask_path':'mask.tiff','channels_path':'channels.parquet',
              'valid_bounds':source.valid_bounds,'block_size':settings.block_size,'cpu_workers':4,'output_dir':'output'}
        for field in ('valid_mask','sample_region'):
            path=getattr(source,field)
            if path:files[field+'.tiff']=path;task[field]=field+'.tiff'
        self.cpu_slots.release()
        try:
            preferred=tuple(s['node']['id'] for s in getattr(self.pool,'assignments',[])
                            if s['node']['id'] in {'81','89'})
            lease=self.pool.lease(preferred) if preferred else self.pool.lease()
            with lease as slot:
                # Two slots share the 32 GiB auxiliary disk-cache budget.
                required=sum(p.stat().st_size for p in files.values())
                if required*3>16*1024**3:raise OSError('CPU task exceeds per-slot cache budget')
                receipt=self.pool.execute(slot,task,files,staging/'cpu_return')
        finally:self.cpu_slots.acquire(priority=True)
        if (receipt['mask_identity'],receipt['study'],receipt['image_id'])!=(mask_identity,record.study,record.image_id):
            raise ValueError('CPU receipt identity differs from the final mask')
        products={}
        for key,path in receipt['products'].items():
            if key not in {'cells','expression','neighbors'}:raise ValueError('Unexpected CPU product')
            name=Path(path).name
            shutil.move(staging/'cpu_return'/name,staging/name);products[key]=str(staging/name)
        channels_path.unlink();shutil.rmtree(staging/'cpu_return')
        return products


class RemoteNormalization:
    """Compute a shared image on a CPU worker, leaving formal publication to the pipeline."""

    SLOT_BYTES=16*1024**3

    def __init__(self,pool: WorkerPool,cpu_slots: CPUSlots,phase):
        self.pool,self.cpu_slots,self.phase=pool,cpu_slots,phase

    @classmethod
    def scratch_bytes(cls,source,shape: tuple[int,int],pages: int,settings,
                      panel_bytes: int = 0) -> int:
        n=shape[0]*shape[1]
        inputs=sum(path.stat().st_size for path in
                   (source.path,source.valid_mask,source.sample_region) if path is not None)
        shared=tiled_tiff_bytes(*shape,pages,4,settings.tile_size)
        # Input payload, returned TIFF, one exact float64 order statistic and
        # working headroom all occupy the worker's per-slot scratch allocation.
        return inputs+panel_bytes+shared+8*n+48*n+1024**2

    @classmethod
    def fits(cls,source,shape: tuple[int,int],pages: int,settings,panel_mapping) -> bool:
        if source.kind!='tiff' or source.paths or source.path.name not in {'input.tiff','path.tiff'}:return False
        import pyarrow as pa
        import pyarrow.parquet as pq
        sink=pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pandas(panel_mapping,preserve_index=False),sink,compression='zstd')
        return cls.scratch_bytes(source,shape,pages,settings,sink.tell())<=cls.SLOT_BYTES

    def __call__(self,*,record,source,reader,panel_mapping,staging,settings,image_request) -> SharedImageResult:
        if source.kind!='tiff' or source.paths or source.path.name!='path.tiff':
            raise ValueError('Remote normalization requires one reviewed input.tiff')
        pages=int((panel_mapping.role.ne('exclude') & panel_mapping.keep_channel.fillna(False)).sum())
        identity={'study':record.study,'image_id':record.image_id,'image_request':image_request}
        checkpoint=settings.cache_dir/'remote_normalization_task.json'
        previous=json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
        task_id=previous['task_id'] if previous.get('identity')==identity else uuid.uuid4().hex
        write_json(checkpoint,{'task_id':task_id,'identity':identity})
        panel_path=staging/'panel.parquet';write_table(panel_path,panel_mapping)
        scratch=self.scratch_bytes(source,reader.shape[-2:],pages,settings,panel_path.stat().st_size)
        if scratch>self.SLOT_BYTES:raise ValueError('Normalization exceeds remote CPU slot scratch budget')
        files={'input.tiff':source.path,'panel.parquet':panel_path}
        task={'kind':'normalization','task_id':task_id,'study':record.study,
              'image_id':record.image_id,'image_request':image_request,
              'image_path':'input.tiff','panel_path':'panel.parquet',
              'channel_count':len(source.channels),'valid_bounds':source.valid_bounds,
              'normalized_q995':source.normalized_q995,'tile_size':settings.tile_size,
              'block_size':settings.block_size,'compression':settings.compression,
              'cpu_workers':settings.cpu_workers,'scratch_bytes':scratch,'output_dir':'output'}
        for field in ('valid_mask','sample_region'):
            path=getattr(source,field)
            if path:files[field+'.tiff']=path;task[field]=field+'.tiff'
        self.phase('normalization_waiting_slot')
        self.cpu_slots.release()
        try:
            with self.pool.lease() as slot:
                node=slot['node']['id']
                self.phase('normalization_transfer',node=node)
                receipt=self.pool.execute(slot,task,files,staging/'normalization_return',
                    phase_callback=lambda phase,node_id=None:self.phase('normalization_'+phase,
                                                                        node=node_id or node))
        finally:self.cpu_slots.acquire(priority=True)
        if any(receipt.get(key)!=task[key] for key in ('task_id','kind','study','image_id','image_request')):
            raise ValueError('Normalization receipt identity differs')
        if receipt.get('shape_yx')!=list(reader.shape[-2:]):
            raise ValueError('Normalization receipt shape differs')
        products=receipt.get('products')
        if not isinstance(products,dict) or set(products) not in ({'channels'},{'channels','image'}):
            raise ValueError('Normalization receipt products differ')
        for key in products:
            expected={'image':'image.tiff','channels':'channels.parquet'}[key]
            if Path(products[key]).name!=expected:
                raise ValueError('Normalization product name differs')
            source_file=staging/'normalization_return'/expected
            if not source_file.is_file():raise ValueError('Normalization returned product is missing')
            shutil.move(source_file,staging/expected)
        channels=pd.read_parquet(staging/'channels.parquet')
        if len(channels)!=len(panel_mapping):
            raise ValueError('Normalization receipt omitted source channels')
        if ('image' in products)!=bool(channels.tiff_page.notna().any()):
            raise ValueError('Normalization image and channel pages disagree')
        shutil.rmtree(staging/'normalization_return')
        return SharedImageResult(staging/'image.tiff' if 'image' in products else None,
                                 channels,reader.shape[-2:])


def stage_source(source, destination: Path):
    """Copy reviewed assets byte-for-byte while their input lease is held by the caller."""
    destination.mkdir(parents=True,exist_ok=True)
    if source.kind!='tiff' or source.paths:
        raise ValueError('Distributed preparation requires the reviewed single input.tiff delivery')
    receipt=destination/'staged_sources.json'
    copies=json.loads(receipt.read_text()) if receipt.is_file() else {}
    def version(path):
        stat=path.stat()
        return {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    def stage(path,target):
        before={'path':str(path),**version(path)}
        saved=copies.get(target.name)
        if (saved is not None and saved['source']==before and target.is_file()
                and saved['copy']==version(target)):
            return target
        copies.pop(target.name,None);write_json(receipt,copies)
        shutil.copy2(path,target)
        if before!={'path':str(path),**version(path)} or target.stat().st_size!=before['size']:
            raise ValueError('Source changed during staging')
        copies[target.name]={'source':before,'copy':version(target)}
        write_json(receipt,copies)
        return target
    fields={}
    for key in ('path','valid_mask','sample_region'):
        path=getattr(source,key)
        if path is None:continue
        target=destination/(key+path.suffix)
        fields[key]=stage(path,target)
    if source.mask and source.mask.path.exists():
        path=source.mask.path;target=destination/('source_mask'+path.suffix)
        fields['mask']=replace(source.mask,path=stage(path,target))
    return replace(source,**fields)


def publish_staged_image(local_metadata: Path, destination: Path, attempt: str,
                         locks: Path, binding_check, input_binding: dict, mask_binding: dict,
                         *, source_directory: Path | None = None, expected_request: dict | None = None,
                         allow_new: bool = False, previous_attempt: str | None = None,
                         replace_previous: bool = False) -> dict:
    """Rebind product paths and publish under the shared NAS writer gate and current attempt."""
    metadata=json.loads(local_metadata.read_text())
    if expected_request is not None and metadata['request']!=expected_request:
        raise ValueError('Staged result does not match its frozen scientific request')
    with formal_write_lease(destination,io_lock_root=locks,priority='high'),resource_lock(destination.parents[1],shared=True),resource_lock(source_directory or destination,shared=True,name='.input.lock'),resource_lock(destination,name='.preparation.lock'):
        binding_check()
        required=sum(Path(path).stat().st_size for path in metadata['products'].values())
        if shutil.disk_usage(destination).free<100*1024**3+required:
            raise OSError('Formal NAS publication would consume the 100 GiB reserve')
        current_path=destination/'metadata.json'
        current=json.loads(current_path.read_text()) if current_path.exists() else {}
        owns_previous=replace_previous and current.get('controller_attempt')==previous_attempt
        if current.get('controller_attempt')!=attempt and not owns_previous and not (allow_new and not current_path.exists()):
            raise ValueError('Formal publication attempt is obsolete')
        # A metadata-only D01 rebind leaves valid pixel/cell products in place.
        if (metadata['products'] and all(Path(path).parent==destination for path in metadata['products'].values())
                and current.get('request')==metadata['request'] and not current.get('publication_incomplete')):
            from cellweft.data.preparation.validation import validate_products
            validate_products(metadata['products'],tuple(metadata['shape_yx']))
            metadata.update(controller_attempt=attempt,input_binding=input_binding,mask_binding=mask_binding)
            with resource_lock(destination):write_json(current_path,metadata)
            return metadata
        stage=destination.parent/('.publish_'+attempt);stage.mkdir()
        try:
            products={}
            for key,path in metadata['products'].items():
                name=Path(path).name;shutil.copy2(path,stage/name);products[key]=str(destination/name)
            metadata.update(products=products,controller_attempt=attempt,
                            input_binding=input_binding,mask_binding=mask_binding)
            with resource_lock(destination):
                write_json(destination/'metadata.json',{**current,'request':metadata['request'],
                    'controller_attempt':attempt,
                    'segmentation_task':metadata.get('segmentation_task'),'publication_incomplete':True})
            publish_image(stage,destination,metadata)
        finally:shutil.rmtree(stage)
    return metadata


def run_distributed(plan: dict, discover=None) -> dict:
    """Schedule accepted revisions, freeing remote slots before NAS publication.

    ``discover`` returns current exact accepted scopes and gaps. A continuous
    controller revisits those scopes without re-running unchanged revisions.
    STOP_ADMISSION drains admitted work; STOP also ends a discovery loop.
    Local metadata is the durable compute checkpoint until formal publication.
    """
    from cellweft.data.preparation.validation import validate_products
    from cellweft.data.preparation.local_resources import admission_paused

    root=Path(plan['run_root']);root.mkdir(parents=True,exist_ok=True)
    data_root=Path(plan['data_root']);cache=Path(plan['cache_root']);cache.mkdir(parents=True,exist_ok=True)
    locks=Path(plan['nas_lock_root'])
    local_cpu_limit=int(plan.get('local_cpu_slots',2))
    if not 1<=local_cpu_limit<=4:raise ValueError('Local CPU slots must be within 1..4')
    cpu_slots=CPUSlots(local_cpu_limit)
    local_heavy_slots=CPUSlots(max(1,local_cpu_limit-1))
    expected={'226':(2,'bf16',32),'73':(2,'bf16',16),'72':(4,'fp32',8)}
    actual={n['id']:(n['slots'],n['precision'],n['batch_size']) for n in plan['gpu_nodes']}
    if actual!=expected:raise ValueError('GPU plan differs from the approved eight-slot configuration')
    local_gpu=plan.get('local_gpu')
    if local_gpu is not None:
        if (local_gpu.get('id'),local_gpu.get('device'),local_gpu.get('precision'),
                local_gpu.get('batch_size'))!=('local','cuda:0','bf16',8):
            raise ValueError('Local GPU slot must use local/cuda:0/bf16/batch 8')
        budget=local_gpu.get('memory_budget_bytes')
        if not isinstance(budget,int) or isinstance(budget,bool) or budget<=0:
            raise ValueError('Local GPU memory budget must be a positive byte count')
        if not isinstance(local_gpu.get('weights'),str) or not Path(local_gpu['weights']).is_absolute():
            raise ValueError('Local GPU weights must be an absolute path')
    gateway=next(n for n in plan['gpu_nodes'] if n['id']=='73')
    transport=Transport(gateway,Path(plan['transfer_root']))
    if plan.get('transfer_relays'):
        transport.configure_relays(plan['transfer_relays'])
    publication_policy_path=plan.get('publication_policy_path')
    if publication_policy_path is not None:
        if not isinstance(publication_policy_path,str) or not Path(publication_policy_path).is_absolute():
            raise ValueError('publication_policy_path must be an absolute local file path')
        publication_policy_path=Path(publication_policy_path)
    publication_mode='formal'
    def refresh_publication_policy():
        nonlocal publication_mode
        mode='formal'
        if publication_policy_path is not None:
            policy=json.loads(publication_policy_path.read_text())
            if not isinstance(policy,dict) or set(policy)!={'mode'}:
                raise ValueError('Publication policy must contain only mode')
            mode=policy['mode']
        if mode not in {'formal','deferred'}:raise ValueError('Publication mode must be formal or deferred')
        changed=mode!=publication_mode
        publication_mode=mode
        return changed
    refresh_publication_policy()
    deferred=DeferredStore(transport,[transport.root]+[r['transport'].root for r in transport.relays],root.name)
    heartbeat=transport.root/root.name/'heartbeat'
    def beat():
        Transport.ssh(gateway,['mkdir','-p',str(heartbeat.parent)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        Transport.ssh(gateway,['touch',str(heartbeat)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    stopped=threading.Event()
    def heartbeat_loop():
        while not stopped.wait(30):
            try:beat()
            except subprocess.SubprocessError as error:
                write_json(root/'heartbeat_failure.json',{'error':str(error),'time':time.time()})
    heartbeat_thread=threading.Thread(target=heartbeat_loop,daemon=True)
    gpu_pool=cpu_pool=None
    capacity=CapacityReservations(cache,int(plan.get('cache_budget_bytes',256*1024**3)))
    reviewed_inputs={};header_cache={};states={};active=set();pending=[];revisions={};accepted_scopes=[]
    admission_state={'preparing_study':None,'pending_scopes':0,'prepared_scopes':0}
    def scheduler_limit(value, name):
        if isinstance(value,bool) or not isinstance(value,int) or not 1<=value<=64:
            raise ValueError(f'{name} must be an integer within 1..64')
        return value

    planned_compute=scheduler_limit(plan.get('compute_slots',8),'compute_slots')
    planned_publication=scheduler_limit(plan.get('max_pending_publications',8),'max_pending_publications')
    compute_limit=planned_compute;publication_limit=planned_publication
    policy_path=plan.get('scheduler_policy_path')
    if policy_path is not None:
        if not isinstance(policy_path,str) or not Path(policy_path).is_absolute():
            raise ValueError('scheduler_policy_path must be an absolute local file path')
        policy_path=Path(policy_path)

    def refresh_scheduler_policy():
        nonlocal compute_limit,publication_limit
        if policy_path is None:return
        try:policy=json.loads(policy_path.read_text())
        except (OSError,UnicodeError,json.JSONDecodeError) as error:
            raise ValueError(f'Invalid scheduler policy file {policy_path}: {error}') from error
        if not isinstance(policy,dict):
            raise ValueError('Scheduler policy must be a JSON object')
        unknown=set(policy)-{'compute_slots','max_pending_publications'}
        if unknown:
            raise ValueError(f'Unknown scheduler policy fields: {sorted(unknown)}')
        compute=scheduler_limit(policy.get('compute_slots',planned_compute),'compute_slots')
        publication=scheduler_limit(policy.get('max_pending_publications',planned_publication),
                                    'max_pending_publications')
        compute_limit=compute;publication_limit=publication

    refresh_scheduler_policy()
    publish_limit=int(plan.get('publish_slots',1))
    if not 1<=publish_limit<=3:
        raise ValueError('Publisher slots must be within 1..3')
    beat();heartbeat_thread.start()
    continuous=bool(plan.get('continuous',False));poll_seconds=max(5,float(plan.get('poll_seconds',30)))
    drain_requested=threading.Event();previous_handlers={}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM,signal.SIGINT):
            previous_handlers[signum]=signal.signal(signum,lambda *_:drain_requested.set())
    node_ring=[node for node in plan['gpu_nodes'] for _ in range(node['slots'])]
    if local_gpu is not None:node_ring.append(local_gpu)
    local_segmenter=LocalSegmentation(local_gpu,cpu_slots,cache) if local_gpu is not None else None
    node_counter=0;gaps=[];discovery_error=None;last_dispatched_study=None

    def set_state(key,status,**values):
        node=states.get(key,{}).get('node')
        states[key]={'study':key[0],'image_id':key[1],'status':status,'updated_at':time.time(),
                     **({'node':node} if node else {}),**values}

    def snapshot(finished=False):
        counts={}
        for state in states.values():counts[state['status']]=counts.get(state['status'],0)+1
        resource_limits={'compute_slots':compute_limit,'inflight_images':publication_limit,
                         'local_cpu_slots':local_cpu_limit,'publish_slots':publish_limit}
        if local_gpu is not None:
            resource_limits['remote_gpu_slots']=8
            resource_limits['local_gpu']={key:local_gpu[key] for key in
                ('id','device','precision','batch_size','memory_budget_bytes')}
        report={'outcomes':list(states.values()),'total':len(states),'finished':finished,
                'counts':counts,'discovery_gaps':gaps,'discovery_error':discovery_error,
                'accepted_scopes':accepted_scopes,
                'admission':dict(admission_state),
                'resource_limits':resource_limits,
                'publication_mode':publication_mode,
                'deferred_roots':[str(path) for path in deferred.roots] if publication_policy_path else [],
                'updated_at':time.time(),'continuous':continuous}
        write_json(root/'status.json',report)
        return report

    def read_gate(_path):return limited_slots(locks,'nas_read',2)

    def prepare_scope(study):
        """Read admission metadata without changing controller-owned queues or states."""
        study=dict(study)
        provided=study.pop('_reviewed_inputs',None)
        scope_key=(study['study'],tuple(study['image_ids']))
        binding=study.get('input_binding')
        cached=provided or reviewed_inputs.get(scope_key)
        if cached is None or binding is None or cached.binding!=binding:
            cached=load_reviewed_study_inputs(data_root/'processed'/study['study'],data_root=data_root,
                image_ids=study['image_ids'],header_cache=header_cache,read_gate=read_gate)
        reviewed_inputs[scope_key]=cached
        samples=cached.samples
        sample_ids=set(samples.sample_id.dropna().astype(str)) if 'sample_id' in samples else set()
        sample_links=cached.images.sample_id.notna() & cached.images.sample_id.astype(str).isin(sample_ids)
        patient_samples=set()
        if {'patient_key','sample_id'}<=set(samples):
            known=samples.patient_key.notna() & samples.patient_key.astype(str).str.strip().ne('')
            patient_samples=set(samples.loc[known,'sample_id'].astype(str))
        patient_links=sample_links & cached.images.sample_id.astype(str).isin(patient_samples)
        coverage={'study':study['study'],'scope':study.get('scope'),
            'record_id':study.get('record_id'),'scope_key':study.get('scope_key'),
            'accepted_images':len(cached.images),
            'sample_linked_images':int(sample_links.sum()),'patient_linked_images':int(patient_links.sum())}
        runtime_versions={'settings':file_identity(Path(plan['settings'])),
            'weights':file_identity(Path(plan['weights'])),
            'local_weights':file_identity(Path(local_gpu['weights'])) if local_gpu is not None else None}
        candidates=[]
        for image_id in study.get('process_image_ids',study['image_ids']):
            if image_id not in study['mask_bindings']:raise ValueError('Explicit reviewed mask binding is missing')
            key=(study['study'],image_id)
            mask_binding=study['mask_bindings'][image_id]
            mask_path=Path(mask_binding['path']) if mask_binding.get('path') else None
            if mask_path is not None and not mask_path.is_absolute():mask_path=data_root/mask_path
            revision=json.dumps({'binding':cached.binding,'mask':mask_binding,
                'mask_file':file_identity(mask_path) if mask_path and mask_path.is_file() else None,
                'mask_config':study.get('mask_bindings_revision'),
                'tissue':study.get('tissue_reviewed_panels',[]),'tissue_config':study.get('tissue_reviewed_revision'),
                **runtime_versions},sort_keys=True)
            preferred=None
            paths=[cache/key[0]/key[1]/'processed'/key[0]/'images'/key[1]/'metadata.json',
                   data_root/'processed'/key[0]/'images'/key[1]/'metadata.json']
            for path in paths:
                if path.is_file():
                    old=json.loads(path.read_text());precision=old.get('request',{}).get('segmentation',{}).get('precision')
                    batch=old.get('request',{}).get('segmentation',{}).get('parameters',{}).get('batch_size')
                    if precision is not None and batch is not None:
                        compatible=[n for n in node_ring if n['precision']==precision and n['batch_size']==batch]
                        if not compatible:raise ValueError(f'No configured GPU slot preserves {precision}/batch {batch}')
                        preferred=compatible[0]
                    break
            candidates.append((key,study,image_id,cached,revision,preferred))
        return coverage,tuple(candidates),runtime_versions

    def apply_scopes(prepared, *, authoritative=False):
        nonlocal node_counter
        candidates={}
        accepted_scopes[:]=[value[0] for value in prepared.values()]
        for _,items,_ in prepared.values():
            for key,study,image_id,inputs,revision,preferred in items:
                candidates[key]=(study,image_id,inputs,revision,preferred)
        pending[:]=[job for job in pending if
            ((job[0]['study'],job[1]) in candidates and
             job[3]==candidates[(job[0]['study'],job[1])][3]) or
            (not authoritative and (job[0]['study'],job[1]) not in candidates)]
        queued={(s['study'],i) for s,i,_,_,_ in pending}
        for key in list(states):
            if authoritative and key not in candidates and key not in active and states[key].get('status')!='withdrawn':
                previous=states[key]
                set_state(key,'withdrawn',previous_status=previous.get('status'),metadata=previous.get('metadata'))
        for key,(study,image_id,inputs,revision,preferred) in candidates.items():
            if key in active or key in queued:continue
            previous=states.get(key,{})
            if revisions.get(key)==revision:
                if previous.get('status')!='withdrawn' and (previous.get('status')!='waiting_io' or time.time()<previous.get('retry_after',0)):continue
            revisions[key]=revision
            node=preferred or node_ring[node_counter%len(node_ring)];node_counter+=1
            pending.append((study,image_id,inputs,revision,node));set_state(key,'queued',node=node['id'])
        pending.sort(key=lambda job:job[0]['mask_bindings'][job[1]]['status']=='unknown')

    def take_pending():
        nonlocal last_dispatched_study
        choice=take_pending_job(pending,active,last_dispatched_study)
        last_dispatched_study=choice[0]['study']
        return choice

    def compute(job):
        study,image_id,inputs,revision,node=job
        key=(study['study'],image_id);attempt=uuid.uuid4().hex;work=cache/key[0]/key[1]
        record,canonical,panel=inputs.source(image_id,study['mask_bindings'][image_id])
        if canonical.kind!='tiff' or canonical.paths or canonical.path.name!='input.tiff':
            raise ValueError('Only reviewed input.tiff deliveries can be staged')
        row=inputs.images.set_index('image_id').loc[image_id];shape=(int(row.height_px),int(row.width_px))
        formal=data_root/'processed'/key[0]/'images'/key[1];study_root=formal.parents[1]
        weights=Path(node['weights']) if node['id']=='local' else Path(plan['weights'])
        settings=load_settings(Path(plan['settings']),work/'cache',weights=weights,
            batch_size=node['batch_size'],precision=node['precision'],cpu_workers=4,
            allow_unknown_mask_segmentation=bool(plan.get('allow_unknown_mask_segmentation', False)),
            sampling_reviewed=row.panel_id in study.get('tissue_reviewed_panels',[]))
        desired=preparation_request(canonical,panel,settings,shape)
        settings_path=Path(plan['settings'])
        frozen_config=[(settings_path,file_identity(settings_path))]
        mask_root=plan.get('mask_bindings_root')
        for name,path in (('mask_bindings_revision',Path(mask_root)/f"{key[0]}.json" if mask_root else None),
                          ('tissue_reviewed_revision',plan.get('tissue_reviewed_panels_path'))):
            version=study.get(name)
            if version:path=version['path']
            if path:frozen_config.append((Path(path),version))
        def check():
            assert_reviewed_binding(study_root,data_root=data_root,binding=inputs.binding)
            for path,version in frozen_config:
                current=file_identity(path) if path.is_file() else None
                if current!=version:
                    raise ValueError('Reviewed runtime binding or settings changed after staging')
            if preparation_request(canonical,panel,settings,shape)!=desired:
                raise ValueError('Reviewed image/mask/weights changed after task staging')
        rebind=False
        set_state(key,'waiting_io',phase='validate_formal',attempt=attempt)
        with limited_slots(locks,'nas_read',2),resource_lock(study_root,shared=True),resource_lock(formal,shared=True):
            check()
            old=json.loads((formal/'metadata.json').read_text()) if (formal/'metadata.json').is_file() else {}
            previous_attempt=old.get('controller_attempt');new_delivery=not old
            if (old.get('request')==desired and not old.get('publication_incomplete')
                    and old.get('products') and all(Path(p).is_file() for p in old['products'].values())):
                validate_products(old['products'],shape)
                if old.get('input_binding')==inputs.binding and old.get('mask_binding')==study['mask_bindings'][image_id]:
                    return {'outcome':{'study':key[0],'image_id':key[1],'status':'reused',
                        'metadata':str(formal/'metadata.json'),'gaps':old.get('gaps',[]),
                        'sampling':old.get('sampling',{}).get('status','pending'),'node':node['id']}}
                rebind=True
        # Reusing a current formal result needs no relay SSH or mask scan.
        # Relay lookup also must not hold a scarce NAS reader while waiting.
        package=deferred.find(key[0],key[1],desired) if publication_policy_path is not None else None
        if package is not None and (package[1]['input_binding']!=inputs.binding or
                                    package[1]['mask_binding']!=study['mask_bindings'][image_id]):
            package=None
        if rebind and package is None:
            work.mkdir(parents=True,exist_ok=True)
            write_json(work/'rebind.json',old)
            return {'metadata':work/'rebind.json','formal':formal,'attempt':attempt,'check':check,
                'binding':inputs.binding,'mask_binding':study['mask_bindings'][image_id],
                'desired':desired,'previous_attempt':previous_attempt,'new_delivery':new_delivery,
                'work':work,'shape':shape,'revision':revision,'rebind':True,'node':node['id']}
        if package is not None:
            directory,manifest=package
            if (manifest['study'],manifest['image_id'])!=key or tuple(manifest['shape_yx'])!=shape:
                raise ValueError('Deferred package identity differs from reviewed image')
            if old and (old.get('request')!=desired or old.get('publication_incomplete')
                        or not all(Path(p).is_file() for p in old.get('products',{}).values())):
                with limited_slots(locks,'nas_write',1,priority='high'),resource_lock(study_root,shared=True),resource_lock(formal,name='.preparation.lock'),resource_lock(formal):
                    check()
                    current=json.loads((formal/'metadata.json').read_text())
                    if current.get('controller_attempt')!=previous_attempt:
                        raise ValueError('Preparation claim is obsolete')
                    write_json(formal/'metadata.json',{**current,'controller_attempt':attempt,'completed':[],
                        'product_status':{name:'pending' for name in current.get('products',{})},
                        'publication_incomplete':True})
                    previous_attempt=attempt
            if publication_mode=='deferred':
                if work.exists():shutil.rmtree(work)
                return {'outcome':pending_merge_outcome(manifest,directory,node['id'])}
            return {'package':package,'formal':formal,'attempt':attempt,'check':check,
                'binding':inputs.binding,'mask_binding':study['mask_bindings'][image_id],
                'desired':desired,'previous_attempt':previous_attempt,'new_delivery':new_delivery,
                'work':work,'shape':shape,'revision':revision,'rebind':False,'node':node['id'],
                'metadata':directory/'complete.json'}
        local_metadata=work/'processed'/record.study/'images'/image_id/'metadata.json'
        obsolete_package=None
        if publication_policy_path is not None and not local_metadata.is_file():
            previous=deferred.find(key[0],key[1],None)
            if previous is not None:
                previous_dir,previous_manifest=previous
                if package is None:
                    obsolete_package=str(previous_dir)
                    write_json(root/'deferred_obsolete'/key[0]/f'{key[1]}.json',
                        {'status':'obsolete','package':obsolete_package,
                         'old_revision':previous_manifest['revision'],'current_revision':revision})
                    restored=deferred.restore(previous_dir,previous_manifest,work)
                    local_metadata.parent.mkdir(parents=True,exist_ok=True)
                    write_json(local_metadata,json.loads(restored.read_text()))
        reusable=False;local={};remote_planned=False
        cpu_assistance=cpu_pool is not None and shape[0]*shape[1]*(int(row.channel_count)*12+48)<5*1024**3
        if local_metadata.is_file():
            local=json.loads(local_metadata.read_text())
            if (local.get('request')==desired and not local.get('publication_incomplete')
                    and local.get('products') and all(Path(p).is_file() for p in local['products'].values())):
                validate_products(local['products'],shape);reusable=True
        if not reusable:
            seed_products=old.get('products',{}) if old.get('products') and not local_metadata.exists() else {}
            seed_bytes=sum(Path(path).stat().st_size for path in seed_products.values() if Path(path).is_file())
            image_request=desired['image']
            existing_product=local if local_metadata.is_file() else old
            shared_products=existing_product.get('products',{})
            reuse_shared=(existing_product.get('image_request')==image_request and
                          all(name in shared_products and Path(shared_products[name]).is_file()
                              for name in ('image','channels')))
            pages=int((panel.role.ne('exclude') & panel.keep_channel.fillna(False)).sum())
            remote_planned=(bool(plan.get('remote_normalization',False)) and cpu_pool is not None
                            and not reuse_shared and RemoteNormalization.fits(canonical,shape,pages,settings,panel))
            additional=(source_copy_bytes(canonical,work/'input') + seed_bytes +
                        image_work_bytes(*shape,pages,
                            mask_unknown=canonical.mask_status=='unknown',
                            normalized=canonical.normalized_q995 is not None or remote_planned,
                            reuse_shared=reuse_shared,tile_size=settings.tile_size))
            capacity.acquire(work,additional)
        try:
            if old and (old.get('request')!=desired or old.get('publication_incomplete')
                        or not all(Path(p).is_file() for p in old.get('products',{}).values())):
                # Withdraw stale completion before a changed scientific request.
                # This short metadata write owns neither a CPU nor a GPU slot.
                with limited_slots(locks,'nas_write',1,priority='high'),resource_lock(study_root,shared=True),resource_lock(formal,name='.preparation.lock'),resource_lock(formal):
                    check()
                    current=json.loads((formal/'metadata.json').read_text())
                    if current.get('controller_attempt')!=previous_attempt:
                        raise ValueError('Preparation claim is obsolete')
                    write_json(formal/'metadata.json',{**current,'controller_attempt':attempt,'completed':[],
                        'product_status':{key:'pending' for key in current.get('products',{})},'publication_incomplete':True})
                    previous_attempt=attempt
            if not reusable:
                set_state(key,'waiting_io',phase='stage_input',attempt=attempt)
                with limited_slots(locks,'nas_read',2),resource_lock(study_root,shared=True),resource_lock(canonical.path.parent,shared=True,name='.input.lock'):
                    check();local_source=stage_source(canonical,work/'input');check()
                    # Seed a changed sampling/mask request from valid formal assets;
                    # prepare_image invalidates only its real dependencies.
                    if old.get('products') and not local_metadata.exists():
                        local_metadata.parent.mkdir(parents=True,exist_ok=True);seed={**old,'products':{}}
                        for name,path in old['products'].items():
                            target=local_metadata.parent/Path(path).name;seed['products'][name]=str(target)
                            if Path(path).is_file():
                                shutil.copy2(path,target)
                        write_json(local_metadata,seed)
                set_state(key,'waiting_cpu',phase='local_preparation',attempt=attempt)
                heavy=(bool(plan.get('remote_normalization',False)) and cpu_pool is not None
                       and not remote_planned and local_cpu_limit>1)
                # Large local jobs must leave a slot for remote submissions and returns.
                # Wait for this gate before owning any CPU computation slot.
                with (local_heavy_slots if heavy else nullcontext()),cpu_slots:
                    set_state(key,'computing',phase='local_preparation',attempt=attempt)
                    local_record=replace(record,output_path=work/'processed'/record.study)
                    with open_source(local_source) as reader:
                        remote_shared=remote_planned
                        shared_executor=(RemoteNormalization(cpu_pool,cpu_slots,
                            lambda phase,**values:set_state(key,'computing',phase=phase,attempt=attempt,**values))
                            if remote_shared else None)
                        result=prepare_image(local_record,local_source,reader,panel,settings,
                            local_segmenter if node['id']=='local' else
                            RemoteSegmentation(gpu_pool,{'node':node,'lease_on_execute':True},cpu_slots),
                            identity_source=canonical,
                            cpu_executor=RemoteCPU(cpu_pool,cpu_slots) if cpu_assistance else None,
                            shared_executor=shared_executor)
                    local_metadata=result.metadata_path
            write_json(work/'delivery.json',{'stage':'computed','study':key[0],'image_id':image_id,
                'metadata':str(local_metadata),'request':desired,'input_binding':inputs.binding,
                'mask_binding':study['mask_bindings'][image_id],'node':node['id']})
            return {'metadata':local_metadata,'formal':formal,'attempt':attempt,'check':check,
                'binding':inputs.binding,'mask_binding':study['mask_bindings'][image_id],
                'desired':desired,'previous_attempt':previous_attempt,'new_delivery':new_delivery,
                'work':work,'shape':shape,'revision':revision,'rebind':False,
                'cpu_node_assistance':cpu_assistance and not reusable,'node':node['id'],
                'obsolete_package':obsolete_package}
        finally:
            if not reusable:capacity.release(work)

    def pending_merge_outcome(manifest, directory, node):
        complete={'image','channels','mask','cells','expression','neighbors'}<=set(manifest['metadata']['completed'])
        outcome={'study':manifest['study'],'image_id':manifest['image_id'],
                'status':'computed_pending_merge','metadata':str(directory/'complete.json'),
                'products_complete':complete,'gaps':manifest['metadata'].get('gaps',[]),
                'sampling':manifest['metadata'].get('sampling',{}).get('status','pending'),
                'node':node}
        if manifest.get('obsolete_package'):outcome['obsolete_package']=manifest['obsolete_package']
        return outcome

    def publish(delivery):
        if publication_mode=='deferred':
            if 'package' in delivery:directory,manifest=delivery['package']
            else:directory,manifest=deferred.persist(delivery)
            # The verified remote package now owns the computed checkpoint.
            shutil.rmtree(delivery['work'])
            return pending_merge_outcome(manifest,directory,delivery['node'])
        if 'package' in delivery:
            directory,manifest=delivery['package']
            restore_checkpoint=manifest['pending_segmentation']
            if restore_checkpoint:
                needed=sum(size for name,size in manifest['files'].items() if name.startswith('checkpoint/')
                           and (not (delivery['work']/Path(name).relative_to('checkpoint')).is_file()
                                or (delivery['work']/Path(name).relative_to('checkpoint')).stat().st_size!=size))
                capacity.acquire(delivery['work'],needed)
            try:
                delivery={**delivery,'metadata':deferred.restore(directory,manifest,delivery['work'],
                    checkpoint=restore_checkpoint)}
            finally:
                if restore_checkpoint:capacity.release(delivery['work'])
        metadata=publish_staged_image(delivery['metadata'],delivery['formal'],delivery['attempt'],locks,
            delivery['check'],delivery['binding'],delivery['mask_binding'],source_directory=delivery['formal'],
            expected_request=delivery['desired'],allow_new=delivery['new_delivery'],
            previous_attempt=delivery['previous_attempt'],replace_previous=True)
        complete={'image','channels','mask','cells','expression','neighbors'}<=set(metadata['completed'])
        outcome={'study':delivery['formal'].parents[1].name,'image_id':delivery['formal'].name,
                 'status':'complete' if complete else 'partial','metadata':str(delivery['formal']/'metadata.json'),
                 'gaps':metadata['gaps'],'sampling':metadata['sampling']['status'],
                 'cpu_node_assistance':delivery.get('cpu_node_assistance',False),'node':delivery['node']}
        if delivery.get('obsolete_package'):outcome['obsolete_package']=delivery['obsolete_package']
        if 'package' in delivery and (complete or metadata.get('segmentation_task') is None):
            try:deferred.retire(*delivery['package'])
            except (OSError,subprocess.SubprocessError) as error:
                outcome['deferred_cleanup_pending']=str(delivery['package'][0])
                outcome['deferred_cleanup_error']=str(error)
        # Unknown-mask normalization is recoverable from its formal products;
        # retain only an unresolved segmentation handoff not yet held on NAS.
        if complete or metadata.get('segmentation_task') is None:shutil.rmtree(delivery['work'])
        else:write_json(delivery['work']/'delivery.json',{'stage':'published_partial','metadata':outcome['metadata']})
        return outcome

    def failure(key,error):
        if isinstance(error,CapacityGap):
            decision=error.decision
            set_state(key,'capacity_gap',required_bytes=decision.required_bytes,
                      budget_bytes=decision.budget_bytes,free_bytes=decision.free_bytes,
                      existing_bytes=decision.existing_bytes,reason=decision.reason)
            return
        io=isinstance(error,(OSError,subprocess.SubprocessError))
        set_state(key,'waiting_io' if io else 'failed',error=str(error),retry_after=time.time()+60)

    try:
        gpu_pool=WorkerPool(plan['gpu_nodes'],'gpu',transport,root)
        cpu_pool=WorkerPool(plan['cpu_nodes'],'cpu',transport,root) if plan.get('cpu_nodes') else None
        with ThreadPoolExecutor(max_workers=64) as computations,ThreadPoolExecutor(max_workers=publish_limit) as publications,ThreadPoolExecutor(max_workers=1) as discoveries:
            computing={};publishing={};ready=[];next_discovery=0;next_status=0
            discovering=None;scope_events=queue.Queue();current_scopes={}
            scope_versions={};prepared_scopes={};dirty_scopes={};last_checked={};serial=0
            preparing=None;authoritative_pending=False
            last_runtime_versions=None
            supports_stream=discover is not None and 'on_scope' in inspect.signature(discover).parameters
            def scope_id(entry):return (entry['study'],entry.get('scope'),entry.get('record_id'),tuple(entry['image_ids']))
            def retain_scope(entry):
                nonlocal serial
                entry={key:deepcopy(value) for key,value in entry.items() if key!='_reviewed_inputs'} | (
                    {'_reviewed_inputs':entry['_reviewed_inputs']} if '_reviewed_inputs' in entry else {})
                identity=scope_id(entry)
                previous=current_scopes.get(identity)
                if previous is not None and {k:v for k,v in previous.items() if k!='_reviewed_inputs'}=={
                        k:v for k,v in entry.items() if k!='_reviewed_inputs'}:
                    if '_reviewed_inputs' in entry:current_scopes[identity]=entry
                    return
                serial+=1
                current_scopes[identity]=entry;scope_versions[identity]=serial;dirty_scopes[identity]=serial

            def prepared_current():
                return {identity:prepared_scopes[identity][1] for identity in current_scopes
                    if identity in prepared_scopes and scope_versions[identity]==prepared_scopes[identity][0]}

            def start_preparation(identity,version,entry):
                future=Future()
                def read_metadata():
                    if not future.set_running_or_notify_cancel():return
                    try:future.set_result(prepare_scope(entry))
                    except BaseException as error:future.set_exception(error)
                threading.Thread(target=read_metadata,name='cellweft-admission',daemon=True).start()
                return identity,version,future

            while True:
                refresh_scheduler_policy()
                refresh_publication_policy()
                draining=drain_requested.is_set() or (root/'STOP_ADMISSION').exists() or (root/'STOP').exists()
                while True:
                    try:entry=scope_events.get_nowait()
                    except queue.Empty:break
                    if not draining:retain_scope(entry)
                if discovering is not None and discovering.done():
                    try:
                        studies,new_gaps=discovering.result();gaps=new_gaps
                        # Retain validated adapters delivered by scope callbacks.
                        completed={scope_id(entry):{**entry,**({'_reviewed_inputs':current_scopes[scope_id(entry)]['_reviewed_inputs']} if scope_id(entry) in current_scopes and '_reviewed_inputs' in current_scopes[scope_id(entry)] else {})} for entry in studies}
                        if not draining:
                            for identity in set(current_scopes)-set(completed):
                                current_scopes.pop(identity);scope_versions.pop(identity)
                                dirty_scopes.pop(identity,None);last_checked.pop(identity,None)
                            for entry in completed.values():retain_scope(entry)
                            current_scopes={identity:current_scopes[identity] for identity in completed}
                            if not dirty_scopes and current_scopes:
                                oldest=min(current_scopes,key=lambda key:last_checked.get(key,0))
                                if time.monotonic()-last_checked.get(oldest,0)>=poll_seconds:
                                    serial+=1;scope_versions[oldest]=serial;dirty_scopes[oldest]=serial
                            authoritative_pending=True
                        discovery_error=None
                    except (OSError,ValueError,KeyError) as error:
                        discovery_error=str(error)
                        if not continuous:raise
                    discovering=None;next_discovery=time.monotonic()+poll_seconds
                if time.monotonic()>=next_discovery and discovering is None and not draining:
                    if discover is None:
                        for entry in plan['studies']:retain_scope(entry)
                        authoritative_pending=True;next_discovery=float('inf')
                    elif supports_stream:
                        discovering=discoveries.submit(discover,on_scope=scope_events.put)
                    else:discovering=discoveries.submit(discover)
                if preparing is not None and preparing[2].done():
                    identity,version,future=preparing;preparing=None
                    try:
                        result=future.result()
                        if not draining and scope_versions.get(identity)==version:
                            if last_runtime_versions is not None and result[2]!=last_runtime_versions:
                                for other in current_scopes:
                                    if other!=identity:
                                        serial+=1;scope_versions[other]=serial;dirty_scopes[other]=serial
                            last_runtime_versions=result[2]
                            prepared_scopes[identity]=(version,result)
                            last_checked[identity]=time.monotonic()
                            dirty_scopes.pop(identity,None)
                            apply_scopes(prepared_current())
                            discovery_error=None
                    except (OSError,ValueError,KeyError) as error:
                        if not draining and scope_versions.get(identity)==version:
                            discovery_error=str(error)
                            dirty_scopes.pop(identity,None)
                            if not continuous:raise
                if authoritative_pending and not draining and all(
                        identity in prepared_scopes and prepared_scopes[identity][0]==version
                        for identity,version in scope_versions.items()):
                    prepared_scopes={identity:value for identity,value in prepared_scopes.items()
                        if identity in current_scopes}
                    apply_scopes(prepared_current(),authoritative=True)
                    authoritative_pending=False
                if preparing is None and dirty_scopes and not draining:
                    identity,version=next(iter(dirty_scopes.items()))
                    if scope_versions.get(identity)==version:
                        preparing=start_preparation(identity,version,current_scopes[identity])
                    else:dirty_scopes.pop(identity)
                if draining:pending.clear()
                # Returns/publication are collected before admitting another image.
                for future in [f for f in publishing if f.done()]:
                    key=publishing.pop(future)
                    try:states[key]=future.result()
                    except Exception as error:failure(key,error)
                    active.discard(key)
                for future in [f for f in computing if f.done()]:
                    key=computing.pop(future)
                    try:
                        delivery=future.result()
                        if 'outcome' in delivery:states[key]=delivery['outcome'];active.discard(key)
                        else:ready.append((key,delivery));set_state(key,'computed',metadata=str(delivery['metadata']))
                    except Exception as error:failure(key,error);active.discard(key)
                if publication_mode=='formal' and not draining:
                    waiting_merge=[key for key,state in states.items()
                                   if state['status']=='computed_pending_merge' and key not in active]
                    if waiting_merge:
                        for key in waiting_merge:revisions.pop(key,None)
                        apply_scopes(prepared_current())
                while ready and len(publishing)<publish_limit:
                    key,delivery=ready.pop(0);set_state(key,'waiting_publish',metadata=str(delivery['metadata']))
                    publishing[publications.submit(publish,delivery)]=key
                memory_paused=admission_paused(plan)
                while pending and not memory_paused and len(computing)<compute_limit and len(computing)+len(ready)+len(publishing)<publication_limit:
                    job=take_pending();key=(job[0]['study'],job[1]);active.add(key);set_state(key,'computing')
                    computing[computations.submit(compute,job)]=key
                admission_state.update(
                    preparing_study=preparing[0][0] if preparing is not None and not draining else None,
                    pending_scopes=len(dirty_scopes) if not draining else 0,
                    prepared_scopes=len(prepared_current()))
                accepted_scopes[:]=[value[0] for value in prepared_current().values()]
                if time.monotonic()>=next_status:snapshot();next_status=time.monotonic()+5
                preparing_current=(preparing is not None and
                    scope_versions.get(preparing[0])==preparing[1])
                if not (pending or computing or ready or publishing or discovering is not None or
                        (not draining and (preparing_current or dirty_scopes or authoritative_pending))) and (draining or not continuous):break
                waiting=list(computing)+list(publishing)
                if waiting:wait(waiting,timeout=1,return_when=FIRST_COMPLETED)
                else:stopped.wait(min(1,poll_seconds))
    finally:
        try:
            try:
                if gpu_pool:gpu_pool.close()
            finally:
                if cpu_pool:cpu_pool.close()
        finally:
            stopped.set();heartbeat_thread.join(timeout=5)
            for signum,handler in previous_handlers.items():signal.signal(signum,handler)
            snapshot()
    return snapshot(finished=True)
