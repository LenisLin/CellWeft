"""Freeze, verify, or execute independent panel-only CPU vector tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--study', required=True)
    prepare.add_argument('--node-id', required=True, choices=('72', '73'))
    prepare.add_argument('--data-root', required=True, type=Path)
    prepare.add_argument('--s1', required=True, type=Path)
    prepare.add_argument('--bank', required=True, type=Path)
    prepare.add_argument('--packet-root', required=True, type=Path)
    prepare.add_argument('--io-lock-root', type=Path)
    prepare.add_argument('--review-scope-receipt', type=Path)
    verify = sub.add_parser('verify-source')
    verify.add_argument('--packet-root', required=True, type=Path)
    verify.add_argument('--data-root', required=True, type=Path)
    verify.add_argument('--io-lock-root', type=Path)
    stage = sub.add_parser('stage-owner')
    stage.add_argument('--packet-root', required=True, type=Path)
    stage.add_argument('--reused-resources', required=True, type=Path)
    stage.add_argument('--node-id', required=True, choices=('72', '73'))
    execute = sub.add_parser('execute')
    execute.add_argument('--packet-root', required=True, type=Path)
    execute.add_argument('--node-id', required=True, choices=('72', '73'))
    execute.add_argument('--node-root', required=True, type=Path)
    execute.add_argument('--io-lock-root', required=True, type=Path)
    args = parser.parse_args()
    from cellweft.data.preparation.panel_embeddings import (
        build_panel_embedding_packet, execute_panel_embedding_packet,
        stage_owner_panel_packet, verify_panel_embedding_source,
    )
    if args.action == 'prepare':
        result = build_panel_embedding_packet(
            args.study, data_root=args.data_root, s1_path=args.s1,
            bank_dir=args.bank, node_id=args.node_id,
            packet_root=args.packet_root, io_lock_root=args.io_lock_root,
            review_scope_receipt=args.review_scope_receipt)
    elif args.action == 'verify-source':
        result = verify_panel_embedding_source(
            args.packet_root, data_root=args.data_root,
            io_lock_root=args.io_lock_root)
    elif args.action == 'stage-owner':
        result = stage_owner_panel_packet(
            args.packet_root, reused_resources=args.reused_resources,
            node_id=args.node_id)
    else:
        result = execute_panel_embedding_packet(
            args.packet_root, node_id=args.node_id,
            node_root=args.node_root, io_lock_root=args.io_lock_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
