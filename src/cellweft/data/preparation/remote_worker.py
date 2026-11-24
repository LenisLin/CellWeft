"""Persistent queue workers using local scratch and explicit completed envelopes."""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import time
import traceback
import fcntl
import os
import threading
import uuid
from contextlib import contextmanager
from collections.abc import Iterable

from cellweft.data.io.publication import resource_lock
from cellweft.data.io.writers import write_json
from cellweft.data.preparation.io_coordination import _slots, limited_slots


@contextmanager
def transfer_slots(root: Path, node: str):
    """Bound all transfer copies to four globally and two per node."""
    with _slots(root, ((f'node_{node}', 2), ('global', 4))):
        yield


@contextmanager
def shared_disk_reservation(lock_root: Path, destination: Path, amount: int,
                            *, owner: str | None = None):
    """Reserve a transfer write on the destination filesystem.

    Transfer gates remain shared by all roots. Space reservations are counted
    per filesystem because the configured NAS roots may have different
    capacities while sharing the same four transfer slots.
    """
    root=lock_root.parent/'shared_disk_reservations';root.mkdir(parents=True,exist_ok=True)
    record=root/(uuid.uuid4().hex+'.json')
    with resource_lock(root):
        device=os.stat(destination).st_dev
        pending=0
        for path in root.glob('*.json'):
            try:value=json.loads(path.read_text())
            except (OSError,ValueError):continue
            if value.get('device',device)==device:pending+=int(value['bytes'])
        if shutil.disk_usage(destination).free-pending-amount<100*1024**3:
            raise OSError('Shared transfer reservations would consume the 100 GiB reserve')
        write_json(record,{'bytes':amount,'pid':os.getpid(),'host':os.uname().nodename,
                           'destination':str(destination),'device':device,'owner':owner})
    try:yield
    finally:
        with resource_lock(root):record.unlink(missing_ok=True)


@contextmanager
def disk_reservation(root: Path, node: str, cache: Path, task_id: str, amount: int, reserve: int):
    """Reserve shared node-local scratch atomically across this node's worker processes."""
    directory=root.parent/'disk_reservations'/node
    directory.mkdir(parents=True,exist_ok=True)
    record=directory/(task_id+'.json')
    with resource_lock(directory):
        used=0
        for path in directory.glob('*.json'):
            value=json.loads(path.read_text())
            try:os.kill(value['pid'],0)
            except ProcessLookupError:
                path.unlink();continue
            used+=value['bytes']
        disk=shutil.disk_usage(cache)
        if disk.free-used-amount<reserve:
            raise OSError('Node scratch reservations exceed safe available disk')
        write_json(record,{'pid':os.getpid(),'bytes':amount,'task_id':task_id,
                           'cache':str(cache.resolve())})
    try:yield
    finally:
        with resource_lock(directory):record.unlink(missing_ok=True)


def _resolve_request_root(attempt: Path, value: object, allowed: Iterable[Path], label: str,
                          default: Path) -> Path:
    """Resolve a request root, allowing only the queue default or configured roots."""
    if value is None:
        return default.resolve()
    elif not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a non-empty absolute path')
    else:
        candidate = Path(value)
        if not candidate.is_absolute():
            raise ValueError(f'{label} must be an absolute path')
        candidate = candidate.resolve()
    roots = [Path(path).resolve() for path in allowed]
    if not roots:
        raise ValueError(f'{label} is outside the worker configured roots')
    for root in roots:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        return candidate
    raise ValueError(f'{label} is outside the worker configured roots')


def request_roots(attempt: Path, envelope: dict, *, input_roots: Iterable[Path] = (),
                  output_roots: Iterable[Path] = ()) -> tuple[Path, Path]:
    """Return validated request input/output roots.

    A legacy envelope is scoped to its attempt directory. New envelopes may
    name a shared-mount directory, but the worker must have configured that
    directory (or an ancestor) explicitly at startup.
    """
    input_roots=tuple(input_roots)
    output_roots=tuple(output_roots)
    data_root=envelope.get('data_root')
    if data_root is not None:
        if envelope.get('input_root') is not None and envelope.get('input_root') != data_root:
            raise ValueError('data_root and input_root disagree')
        input_value=data_root
        output_value=envelope.get('output_root')
        if output_value is None:
            if not isinstance(data_root,str) or not data_root.strip():
                raise ValueError('data_root must be a non-empty absolute path')
            output_value=str(Path(data_root)/'output')
    else:
        input_value=envelope.get('input_root')
        output_value=envelope.get('output_root')
    input_root = _resolve_request_root(attempt, input_value, input_roots,
                                       'input_root', attempt)
    output_allowlist=tuple(output_roots) or tuple(input_roots)
    output_root = _resolve_request_root(attempt, output_value, output_allowlist,
                                        'output_root', attempt/'output')
    if input_root == output_root:
        raise ValueError('input_root and output_root must be different directories')
    return input_root, output_root


def validate_transport_path(path: Path, allowed: Iterable[Path] = ()) -> Path:
    """Resolve a transfer path and enforce the explicitly configured root set."""
    candidate=Path(path).resolve()
    roots=[Path(root).resolve() for root in allowed]
    if roots:
        for root in roots:
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            return candidate
        raise ValueError('Transfer path is outside the configured relay roots')
    return candidate


def _plain_names(names: object, *, label: str) -> list[str]:
    """Validate transport asset names before joining them to a declared root."""
    if not isinstance(names, list) or not names:
        raise ValueError(f'{label} must be a non-empty list')
    result=[]
    for name in names:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError(f'{label} must contain plain filenames')
        if name in result:
            raise ValueError(f'{label} contains a duplicate filename')
        result.append(name)
    return result


def _nonempty_files(directory: Path, names: Iterable[str]) -> bool:
    """Check that a candidate result set contains closed, non-empty files."""
    return all((directory/name).is_file() and not (directory/name).is_symlink()
               and (directory/name).stat().st_size > 0 for name in names)


def _result_names(mode: str, task: dict, directory: Path, *, require_identity: bool = True) -> list[str]:
    """Validate a complete task-local result and return its portable filenames."""
    receipt_path=directory/'result.json'
    if not receipt_path.is_file() or receipt_path.stat().st_size == 0:
        raise ValueError('result.json is missing or empty')
    receipt=json.loads(receipt_path.read_text())
    result_task=receipt.get('task_id')
    if result_task is not None and result_task != task.get('task_id'):
        raise ValueError('result task identity differs from the running attempt')
    if require_identity and result_task != task.get('task_id'):
        raise ValueError('result task identity is missing from the running attempt')
    if mode == 'cpu':
        kind=task.get('kind','quantification')
        if kind not in {'quantification','normalization'}:
            raise ValueError(f'Unsupported CPU task kind: {kind}')
        for key in ('study','image_id'):
            if receipt.get(key) != task.get(key):
                raise ValueError(f'CPU result {key} identity differs from the running attempt')
        if kind == 'quantification':
            if receipt.get('mask_identity') != task.get('mask_identity'):
                raise ValueError('CPU result mask identity differs from the running attempt')
        else:
            if receipt.get('kind') != 'normalization' or receipt.get('image_request') != task.get('image_request'):
                raise ValueError('Normalization result request identity differs from the running attempt')
    if mode == 'gpu':
        mask=receipt.get('mask','mask.tiff')
        if not isinstance(mask,str) or Path(mask).name != mask:
            raise ValueError('GPU result mask is not a plain filename')
        names=[mask,'result.json']
    else:
        products=receipt.get('products')
        if not isinstance(products,dict) or not products:
            raise ValueError('CPU result has no products')
        if kind == 'normalization':
            if set(products) not in ({'channels'}, {'image','channels'}):
                raise ValueError('Normalization result has unexpected products')
            if {key:Path(value).name for key,value in products.items() if isinstance(value,str)} != {
                    key:('image.tiff' if key == 'image' else 'channels.parquet') for key in products}:
                raise ValueError('Normalization product filenames differ from the task contract')
        names=[]
        for value in products.values():
            if not isinstance(value,str) or not value.strip():
                raise ValueError('CPU result contains an invalid product path')
            name=Path(value).name
            if not name or name in {'.','..'}:
                raise ValueError('CPU result contains an invalid product filename')
            if name not in names:names.append(name)
        names.append('result.json')
    _plain_names(names,label='result files')
    if not _nonempty_files(directory,names):
        raise ValueError('result files are incomplete')
    if mode == 'cpu' and kind == 'normalization':
        from cellweft.data.io.writers import json_value
        from cellweft.data.preparation.validation import validate_products
        import pandas as pd
        shape=receipt.get('shape_yx')
        if (not isinstance(shape,list) or len(shape) != 2
                or any(type(value) is not int or value < 1 for value in shape)):
            raise ValueError('Normalization result has invalid shape_yx')
        channels=pd.read_parquet(directory/'channels.parquet')
        scientific=task.get('image_request',{}).get('panel')
        if not isinstance(scientific,list) or len(channels) != len(scientific):
            raise ValueError('Normalization result omits source channels')
        if channels.source_channel_index.tolist() != list(range(len(channels))):
            raise ValueError('Normalization source channel ordering is invalid')
        fields=list(scientific[0]) if scientific else []
        if (any(key not in channels for key in fields)
                or json_value(channels[fields].astype(object).where(pd.notna(channels),None)
                              .to_dict('records')) != scientific):
            raise ValueError('Normalization channel mapping differs from image_request panel')
        active=int(channels.tiff_page.notna().sum())
        if ('image' in products) != (active > 0):
            raise ValueError('Normalization image presence differs from channel pages')
        validate_products({key:str(directory/Path(value).name) for key,value in products.items()},tuple(shape))
    return names


def _scratch_plan(mode: str, task: dict, transfer_bytes: int) -> tuple[int,int]:
    """Return node reservation and free-space reserve for this submitted task."""
    if mode == 'cpu' and task.get('kind','quantification') == 'normalization':
        amount=task.get('scratch_bytes')
        if type(amount) is not int or not 0 < amount <= 16*1024**3:
            raise ValueError('normalization scratch_bytes must be within 1..16 GiB')
        if amount < transfer_bytes:
            raise ValueError('normalization scratch_bytes cannot be smaller than staged inputs')
        return amount,20*1024**3
    if mode == 'cpu' and task.get('kind','quantification') != 'quantification':
        raise ValueError(f"Unsupported CPU task kind: {task.get('kind')}")
    if mode == 'gpu':
        height,width=task['signal']['shape'][1:]
        transfer_bytes+=height*width*20
    return transfer_bytes*3,0


def _cleanup_stale_reservations(lock_root: Path, node: str, dispatch_id: str) -> None:
    """Withdraw reservations owned by a dead attempt after its physical slot is reclaimed."""
    directory=lock_root.parent/'disk_reservations'/node
    with resource_lock(directory):
        (directory/(dispatch_id+'.json')).unlink(missing_ok=True)
    shared=lock_root.parent/'shared_disk_reservations'
    if shared.exists():
        with resource_lock(shared):
            for path in shared.glob('*.json'):
                try:value=json.loads(path.read_text())
                except (OSError,ValueError):continue
                if value.get('owner') == dispatch_id:path.unlink(missing_ok=True)


def _copy_results(source: Path, destination: Path, names: list[str], lock_root: Path,
                  node: str, dispatch_id: str) -> None:
    """Copy a validated result set with one shared transfer lease and atomic files."""
    total=sum((source/name).stat().st_size for name in names)
    with transfer_slots(lock_root,node):
        destination.mkdir(parents=True,exist_ok=True)
        with shared_disk_reservation(lock_root,destination,total,owner=dispatch_id):
            for name in names:
                temporary=destination/(name+'.partial')
                shutil.copy2(source/name,temporary)
                temporary.replace(destination/name)


def _recover_running_attempt(root: Path, cache: Path, *, mode: str, lock_root: Path,
                             node: str, input_roots: Iterable[Path],
                             output_roots: Iterable[Path]) -> dict:
    """Recover one attempt left in ``running`` after a worker process died.

    A complete output is adopted and marked done. Incomplete output is
    reported as a recoverable failure; it is never silently replayed under the
    old dispatch identity. Both cases release the old scratch reservation.
    """
    running=root/'running.json'
    envelope=json.loads(running.read_text())
    dispatch_id=envelope.get('dispatch_id')
    if not isinstance(dispatch_id,str) or not dispatch_id or root.name != dispatch_id:
        raise ValueError('running attempt dispatch identity differs from its directory')
    input_root,output_root=request_roots(root,envelope,input_roots=input_roots,output_roots=output_roots)
    task=json.loads((input_root/'task.json').read_text())
    task_id=task.get('task_id')
    if not isinstance(task_id,str) or not task_id:
        raise ValueError('running task has no task_id')
    names=_plain_names(envelope.get('files'),label='request files')
    if 'task.json' not in names:
        raise ValueError('request files do not contain task.json')
    local=cache/dispatch_id
    try:
        candidates=[]
        for candidate in (output_root,local/'output'):
            if candidate.exists():
                try:candidate_names=_result_names(mode,task,candidate)
                except (OSError,ValueError):continue
                candidates.append((candidate,candidate_names))
        if candidates:
            source,names=candidates[0]
            result_receipt=json.loads((source/'result.json').read_text())
            if source.resolve() != output_root.resolve():
                _copy_results(source,output_root,names,lock_root,node,dispatch_id)
            write_json(root/'done.json',{'dispatch_id':dispatch_id,'task_id':task_id,
                'files':names,'recovered':True,'seconds':0.0,
                'recovery':{'status':'done','action':'adopt_output',
                            'source':str(source),'running':envelope,
                            'result':result_receipt}})
            return {'dispatch_id':dispatch_id,'status':'done','task_id':task_id,'files':names}
        reason='worker restarted with incomplete running attempt; controller must requeue a new dispatch'
        write_json(root/'failure.json',{'dispatch_id':dispatch_id,'task_id':task_id,
            'error':reason,'error_type':'RecoveryIncomplete','recoverable':True,
            'recovery':{'status':'failed','action':'requeue',
            'previous_state':'running','scratch':str(local),'running':envelope}})
        return {'dispatch_id':dispatch_id,'status':'failure','task_id':task_id,'recoverable':True}
    finally:
        _cleanup_stale_reservations(lock_root,node,dispatch_id)
        if local.exists():shutil.rmtree(local)


def recover_running_attempts(queue: Path, cache: Path, *, mode: str, lock_root: Path,
                             node: str, input_roots: Iterable[Path] = (),
                             output_roots: Iterable[Path] = ()) -> list[dict]:
    """Recover every uncompleted attempt while its physical queue slot is held."""
    input_roots=tuple(input_roots)
    output_roots=tuple(output_roots)
    outcomes=[]
    for running in sorted(queue.glob('*/running.json')):
        root=running.parent
        if (root/'done.json').exists() or (root/'failure.json').exists():
            continue
        try:
            outcomes.append(_recover_running_attempt(root,cache,mode=mode,lock_root=lock_root,
                node=node,input_roots=input_roots,output_roots=output_roots))
        except Exception as error:
            dispatch_id=root.name
            _cleanup_stale_reservations(lock_root,node,dispatch_id)
            try:running_envelope=json.loads((root/'running.json').read_text())
            except (OSError,ValueError):running_envelope=None
            recovery={'status':'failed','action':'requeue','previous_state':'running',
                      'reason':'recovery_validation_failed'}
            if running_envelope is not None:recovery['running']=running_envelope
            write_json(root/'failure.json',{'dispatch_id':dispatch_id,
                'error':str(error),'error_type':type(error).__name__,'recoverable':True,
                'recovery':recovery})
            scratch=cache/dispatch_id
            if scratch.exists():shutil.rmtree(scratch)
            outcomes.append({'dispatch_id':dispatch_id,'status':'failure','recoverable':True,
                             'error':str(error)})
    # A prior worker may have written the completion envelope and died before
    # its finally block removed scratch. These directories are scoped by
    # dispatch and are safe to reclaim once done/failure is durable.
    completed_roots={marker.parent.name for marker in
                     (*queue.glob('*/done.json'),*queue.glob('*/failure.json'))}
    for dispatch_id in completed_roots:
        scratch=cache/dispatch_id
        if scratch.exists():shutil.rmtree(scratch)
    return outcomes


def run_queue(queue: Path, *, mode: str, cache: Path, weights: Path | None = None,
              precision: str = 'bf16', batch_size: int = 16, device: str = 'cuda:0',
              reserve_bytes: int = 20 * 1024**3, lock_root: Path, node: str,
              input_roots: Iterable[Path] = (), output_roots: Iterable[Path] = (),
              launch_id: str | None = None) -> None:
    """Consume one assigned queue; publish completion only after all returned files close.

    The controller owns dispatch and one queue per GPU/CPU slot. A STOP file ends
    idle polling. Failed tasks remain failed and are never retried invisibly.
    """
    queue.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    input_roots=tuple(input_roots)
    output_roots=tuple(output_roots)
    launch_id=launch_id or uuid.uuid4().hex
    heartbeat=queue.parents[2]/'heartbeat'
    ended=threading.Event()
    def enforce_lease():
        while not ended.wait(10):
            if (queue/'STOP').exists() or not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime>300:
                os._exit(75)
    threading.Thread(target=enforce_lease,daemon=True).start()
    executor = None
    physical=lock_root.parent/'worker_slots'
    physical.mkdir(parents=True,exist_ok=True)
    handle=(physical/f'{node}_{mode}_{queue.name}.lock').open('a+b')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close();ended.set();raise RuntimeError('Physical worker slot is already owned')
    with handle:
        # A previous controller may have left STOP behind after its process
        # exited. The slot lock makes this removal safe: the old worker cannot
        # still be observing the queue while a new instance owns the slot.
        (queue/'STOP').unlink(missing_ok=True)
        # Publish the new owner before model loading or recovery. A large
        # adopted result may take longer than the normal worker startup window.
        write_json(queue/'worker.json', {'mode':mode,'device':device,'precision':precision,
                                        'batch_size':batch_size,'ready':False,'recovering':True,
                                        'pid':os.getpid(),'launch_id':launch_id})
        if mode == 'gpu':
            import torch
            from cellweft.data.preparation.segmentation import CellposeExecutor
            torch.set_num_threads(4)
            executor = CellposeExecutor(device, precision=precision)
        # The physical lock proves that the previous owner of this queue slot
        # is gone. Recover its attempts before advertising readiness; a second
        # worker must never execute against the same running dispatch.
        recover_running_attempts(queue,cache,mode=mode,lock_root=lock_root,node=node,
                                 input_roots=input_roots,output_roots=output_roots)
        write_json(queue/'worker.json', {'mode':mode,'device':device,'precision':precision,
                                        'batch_size':batch_size,'ready':True,'pid':os.getpid(),
                                        'launch_id':launch_id})
        try:
            while not (queue/'STOP').exists():
                requests = sorted(queue.glob('*/ready.json'))
                for ready in requests:
                    root = ready.parent
                    if (root/'done.json').exists() or (root/'failure.json').exists():
                        continue
                    envelope = json.loads(ready.read_text())
                    dispatch_id=envelope.get('dispatch_id')
                    if not isinstance(dispatch_id,str) or not dispatch_id:
                        dispatch_id=root.name
                    # Claim before execution so controller cleanup cannot expose
                    # a stale ready marker after removing the completion envelope.
                    ready.rename(root/'running.json')
                    started = time.time()
                    local = cache/dispatch_id
                    reservation=None
                    output_root=None
                    task=None
                    try:
                        if dispatch_id != root.name:
                            raise ValueError('Request dispatch identity differs from its directory')
                        input_root,output_root=request_roots(root,envelope,input_roots=input_roots,
                                                             output_roots=output_roots)
                        names=_plain_names(envelope.get('files'),label='request files')
                        if 'task.json' not in names:
                            raise ValueError('request files do not contain task.json')
                        required = sum((input_root/name).stat().st_size for name in names)
                        submitted=json.loads((input_root/'task.json').read_text())
                        amount,normalization_reserve=_scratch_plan(mode,submitted,required)
                        safe_reserve=(normalization_reserve if normalization_reserve else
                                      max(reserve_bytes,int(shutil.disk_usage(cache).total*.2))
                                      if mode=='cpu' else reserve_bytes)
                        reservation=disk_reservation(lock_root,node,cache,dispatch_id,amount,safe_reserve)
                        reservation.__enter__()
                        local.mkdir()
                        with transfer_slots(lock_root,node):
                            for name in names:shutil.copy2(input_root/name,local/name)
                        task = json.loads((local/'task.json').read_text())
                        if task.get('task_id') in (None,''):
                            raise ValueError('Task has no task_id')
                        if mode == 'gpu':
                            height,width=task['signal']['shape'][1:]
                            if shutil.disk_usage(cache).free < reserve_bytes+height*width*20:
                                raise OSError('GPU mask scratch exceeds available local disk')
                            if task['precision'] != precision or task['parameters']['batch_size'] != batch_size:
                                raise ValueError('Task precision/batch differs from configured worker')
                            task['weights'] = str(weights)
                            receipt_path = executor.execute(task,input_path=local/'signal.tiff',result_path=local/'output'/'result.json')
                            output = receipt_path.parent
                            names = _result_names(mode,task,output,require_identity=False)
                        else:
                            from cellweft.data.preparation.cpu_tasks import execute_cpu_task
                            for key in ('image_path','mask_path','channels_path','panel_path','valid_mask','sample_region'):
                                if task.get(key):task[key]=str(local/Path(task[key]).name)
                            task['output_dir'] = str(local/'output')
                            execute_cpu_task(task)
                            output = local/'output'
                            names = _result_names(mode,task,output,require_identity=False)
                        _copy_results(output,output_root,names,lock_root,node,dispatch_id)
                        write_json(root/'done.json',{'dispatch_id':dispatch_id,
                            'task_id':task['task_id'],'files':names,'seconds':time.time()-started})
                    except Exception as error:
                        write_json(root/'failure.json',{'dispatch_id':dispatch_id,
                            'task_id':task.get('task_id') if isinstance(task,dict) else None,
                            'error':str(error),'traceback':traceback.format_exc(),
                            'error_type':type(error).__name__,
                            'seconds':time.time()-started,
                            'recoverable':isinstance(error,OSError)})
                    finally:
                        if local.exists():shutil.rmtree(local)
                        if reservation is not None:reservation.__exit__(None,None,None)
                time.sleep(2)
        finally:
            ended.set()
            if executor is not None:executor.__exit__(None,None,None)
            write_json(queue/'worker.json',{'mode':mode,'ready':False,
                                            'pid':os.getpid(),'launch_id':launch_id})
