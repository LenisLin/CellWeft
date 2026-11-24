"""Stream one controller-owned asset over SSH within shared transfer limits."""
import argparse
import shutil
import sys
from pathlib import Path
from cellweft.data.preparation.remote_worker import (
    shared_disk_reservation,
    transfer_slots,
    validate_transport_path,
)


def main() -> None:
    """Atomically receive a complete stream, or send one closed file."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('direction',choices=['upload','download'])
    p.add_argument('--path',type=Path,required=True)
    p.add_argument('--node',required=True)
    p.add_argument('--lock-root',type=Path,required=True)
    p.add_argument('--size',type=int)
    p.add_argument('--allowed-root','--root','--relay-root','--data-root',dest='allowed_roots',
                   action='append',type=Path,default=[],
                   help='Configured relay root; repeat for multiple mounted roots.')
    a=p.parse_args()
    path=validate_transport_path(a.path,a.allowed_roots)
    with transfer_slots(a.lock_root,a.node):
        if a.direction=='upload':
            path.parent.mkdir(parents=True,exist_ok=True)
            if a.size is None or shutil.disk_usage(path.parent).free < 100*1024**3+a.size:
                raise OSError('Transfer storage cannot preserve its 100 GiB reserve')
            temporary=path.with_name(path.name+'.partial')
            with shared_disk_reservation(a.lock_root,path.parent,a.size):
                with temporary.open('wb') as output:shutil.copyfileobj(sys.stdin.buffer,output,1024*1024)
                if temporary.stat().st_size!=a.size:
                    temporary.unlink(missing_ok=True)
                    raise ValueError('Transferred input length differs from declared file size')
                temporary.replace(path)
        else:
            with path.open('rb') as source:shutil.copyfileobj(source,sys.stdout.buffer,1024*1024)


if __name__=='__main__':main()
