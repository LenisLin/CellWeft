"""Run one persistent GPU or CPU queue under the local controller's supervision.

Inputs: completed queue envelopes, node-local environment, explicit GPU configuration.
Outputs: closed result files and done/failure receipts in the shared transfer directory.
Side effects: uses bounded local scratch; writes no formal NAS products.
Prerequisites: verified node configuration and controller-owned queue.
"""
import argparse
import os
from pathlib import Path


def main() -> None:
    """Load one slot configuration and execute until its controller writes STOP."""
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1' if name != 'OMP_NUM_THREADS' else '4'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--mode',choices=['gpu','cpu'],required=True)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--lock-root',type=Path,required=True)
    parser.add_argument('--node',required=True)
    parser.add_argument('--launch-id')
    parser.add_argument('--input-root','--allowed-input-root','--data-root',dest='input_roots',
                        action='append',type=Path,default=[],
                        help='Allowed shared-mount root for request data_root/input_root (repeatable).')
    parser.add_argument('--output-root','--allowed-output-root',dest='output_roots',
                        action='append',type=Path,default=[],
                        help='Allowed shared-mount root for request output_root (repeatable).')
    args=parser.parse_args()
    from cellweft.data.preparation.remote_worker import run_queue
    run_queue(args.queue,mode=args.mode,cache=args.cache,weights=args.weights,
              precision=args.precision,batch_size=args.batch_size,device=args.device,
              lock_root=args.lock_root,node=args.node,input_roots=args.input_roots,
              output_roots=args.output_roots,launch_id=args.launch_id)


if __name__=='__main__':main()
