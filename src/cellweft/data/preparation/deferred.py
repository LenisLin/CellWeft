"""Durable, separately validated preparation checkpoints on shared transfer mounts."""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid

from cellweft.data.io.writers import write_json
from cellweft.data.preparation.validation import validate_products


def inspect_package(directory: Path) -> dict | None:
    """A manifest is authoritative only after every declared byte and product reopens."""
    marker=directory/'complete.json'
    if not marker.is_file():return None
    manifest=json.loads(marker.read_text())
    if manifest.get('version')!=1 or not isinstance(manifest.get('files'),dict):
        raise ValueError('Invalid deferred package manifest')
    for name,size in manifest['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or str(relative)=='.':
            raise ValueError('Invalid deferred package path')
        path=directory/relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size!=size:
            raise ValueError(f'Deferred package has an incomplete file: {name}')
    products={key:str(directory/name) for key,name in manifest['products'].items()}
    if not products or any(name not in manifest['files'] for name in manifest['products'].values()):
        raise ValueError('Deferred package product list is incomplete')
    validate_products(products,tuple(manifest['shape_yx']))
    return manifest


def remote_action(action: str, path: Path) -> dict | list | None:
    if action=='usage':
        usage=shutil.disk_usage(path)
        return {'free':usage.free}
    if action=='list':
        if not path.is_dir():return []
        result=[]
        for marker in path.glob('*/complete.json'):
            manifest=inspect_package(marker.parent)
            if manifest is not None:result.append({'path':str(marker.parent),'manifest':manifest})
        return result
    if action=='inspect':return inspect_package(path)
    if action=='retire':
        if (len(path.name)!=32 or any(c not in '0123456789abcdef' for c in path.name)
                or path.parents[3].name!='outbox' or path.is_symlink()):
            raise ValueError('Invalid deferred package retirement path')
        if path.exists():
            if not (path/'complete.json').is_file():
                raise ValueError('Cannot retire an unsealed package')
            shutil.rmtree(path)
        return {'retired':str(path)}
    raise ValueError(f'Unknown deferred action: {action}')


class DeferredStore:
    """Own one outbox namespace; transfer gates remain owned by Transport."""

    def __init__(self, transport, roots: list[Path], namespace: str, reserve_bytes: int = 100*1024**3):
        self.transport=transport
        self.roots=list(dict.fromkeys(Path(root) for root in roots))
        self.namespace=namespace
        self.reserve_bytes=reserve_bytes
        self.lock=threading.Lock()
        self.inflight={str(root):0 for root in self.roots}
        self.persisted={str(root):0 for root in self.roots}
        if not self.roots:raise ValueError('Deferred publication needs a transfer root')

    def _transport(self, root: Path):
        if root==self.transport.root:return self.transport
        for relay in self.transport.relays:
            candidate=relay['transport']
            if candidate.root==root:return candidate
        raise ValueError(f'No configured transport for deferred root {root}')

    def _remote(self, root: Path, action: str, path: Path):
        transport=self._transport(root)
        gateway=transport.gateway
        command=['env',f"PYTHONPATH={gateway['code_root']}/src",gateway['python'],
                 '-m','cellweft.data.preparation.deferred',action,str(path)]
        result=transport.ssh(gateway,command,capture_output=True,text=True,timeout=900)
        return json.loads(result.stdout)

    def _image_dir(self, root: Path, study: str, image_id: str) -> Path:
        for value in (self.namespace,study,image_id):
            if not value or Path(value).name!=value or value in {'.','..'}:
                raise ValueError('Invalid deferred package identity')
        return root/'outbox'/self.namespace/study/image_id

    def find(self, study: str, image_id: str, request: dict | None) -> tuple[Path,dict] | None:
        candidates=[]
        unavailable=[]
        for root in self.roots:
            try:entries=self._remote(root,'list',self._image_dir(root,study,image_id))
            except (OSError,subprocess.SubprocessError) as error:
                unavailable.append(error)
                continue
            for entry in entries:
                manifest=entry['manifest']
                if request is None or manifest['request']==request:
                    candidates.append((Path(entry['path']),manifest))
        if candidates:return max(candidates,key=lambda item:item[1].get('created_at',0))
        if unavailable:raise OSError('Cannot determine whether a deferred package exists on an unavailable relay') from unavailable[0]
        return None

    def _choose(self, amount: int) -> Path:
        with self.lock:
            choices=[]
            for root in self.roots:
                try:free=self._remote(root,'usage',root)['free']-self.inflight[str(root)]
                except (OSError,subprocess.SubprocessError):continue
                if free-amount>=self.reserve_bytes:
                    choices.append((self.inflight[str(root)]+self.persisted[str(root)],-free,root))
            if not choices:raise OSError('Deferred transfer roots cannot preserve the 100 GiB reserve')
            root=min(choices,key=lambda choice:choice[:2])[2]
            self.inflight[str(root)]+=amount
            return root

    def persist(self, delivery: dict) -> tuple[Path,dict]:
        metadata=json.loads(Path(delivery['metadata']).read_text())
        if metadata['request']!=delivery['desired']:
            raise ValueError('Computed metadata differs from the frozen request')
        shape=tuple(metadata['shape_yx'])
        if shape!=tuple(delivery['shape']):raise ValueError('Computed shape differs from reviewed image')
        products={key:Path(value) for key,value in metadata['products'].items()}
        validate_products({key:str(path) for key,path in products.items()},shape)
        pending_task=metadata.get('segmentation_task') is not None and 'mask' not in metadata.get('completed',[])
        files={f'products/{key}{path.suffix}':path for key,path in products.items()}
        names={key:f'products/{key}{path.suffix}' for key,path in products.items()}
        if pending_task:
            work=Path(delivery['work'])
            for path in work.rglob('*'):
                if path.is_file():
                    if path.is_symlink():raise ValueError('Deferred checkpoint contains a symlink')
                    files[f'checkpoint/{path.relative_to(work)}']=path
        sizes={name:path.stat().st_size for name,path in files.items()}
        manifest={'version':1,'study':delivery['formal'].parents[1].name,
                  'image_id':delivery['formal'].name,'attempt':delivery['attempt'],
                  'created_at':time.time(),
                  'revision':delivery['revision'],'request':delivery['desired'],
                  'input_binding':delivery['binding'],'mask_binding':delivery['mask_binding'],
                  'shape_yx':list(shape),'metadata':metadata,'products':names,'files':sizes,
                  'pending_segmentation':pending_task,'node':delivery['node'],
                  'obsolete_package':delivery.get('obsolete_package')}
        amount=sum(sizes.values())+len(json.dumps(manifest).encode())
        root=self._choose(amount)
        directory=self._image_dir(root,manifest['study'],manifest['image_id'])/uuid.uuid4().hex
        try:
            transport=self._transport(root)
            for name,path in files.items():transport.upload(path,directory/name)
            marker=Path(delivery['work'])/f'deferred_{directory.name}.json'
            write_json(marker,manifest)
            try:transport.upload(marker,directory/'complete.json')
            finally:marker.unlink(missing_ok=True)
            verified=self._remote(root,'inspect',directory)
            if verified!=manifest:raise ValueError('Deferred package verification differs from upload')
            with self.lock:self.persisted[str(root)]+=amount
            return directory,manifest
        finally:
            with self.lock:self.inflight[str(root)]-=amount

    def restore(self, directory: Path, manifest: dict, work: Path, *, checkpoint: bool = False) -> Path:
        root=next(root for root in self.roots if directory.is_relative_to(root))
        if self._remote(root,'inspect',directory)!=manifest:
            raise ValueError('Deferred package changed before restoration')
        destination=work/'deferred_restore'/directory.name
        destination.mkdir(parents=True,exist_ok=True)
        # These verified NFS mounts are readable on the controller. Publish
        # directly from them rather than duplicating unreserved large products
        # in local scratch during final merging.
        products={key:str(directory/name) for key,name in manifest['products'].items()}
        validate_products(products,tuple(manifest['shape_yx']))
        metadata={**manifest['metadata'],'products':products}
        path=destination/'metadata.json'
        write_json(path,metadata)
        if checkpoint:
            transport=self._transport(root)
            for name,size in manifest['files'].items():
                if name.startswith('checkpoint/'):
                    target=work/Path(name).relative_to('checkpoint')
                    if not target.is_file() or target.stat().st_size!=size:
                        transport.download(directory/name,target)
                    if target.stat().st_size!=size:raise ValueError('Deferred checkpoint length differs')
        return path

    def retire(self, directory: Path, manifest: dict) -> None:
        """Call only after current formal publication has reopened successfully."""
        root=next(root for root in self.roots if directory.is_relative_to(root))
        if directory.parent!=self._image_dir(root,manifest['study'],manifest['image_id']):
            raise ValueError('Deferred retirement identity differs')
        self._remote(root,'retire',directory)


if __name__=='__main__':
    print(json.dumps(remote_action(sys.argv[1],Path(sys.argv[2]))))
