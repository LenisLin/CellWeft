"""Export fixed MAP knowledge-aligned gene representations on CPU.

Inputs: official MAP-KG checkpoint, ESM2 gene table, MAP-KG gene metadata.
Outputs: embeddings.pt, gene_symbols.json and manifest.json in a new NAS directory.
Upstream: MAGIC-AI4Med/MAP, MAP-KG/model/model.py, ResidualProjector.
The functional forward matches the upstream projector in eval mode (dropout off).
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--esm-table', type=Path, required=True)
parser.add_argument('--gene-metadata', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--threads', type=int, default=4)
args = parser.parse_args()
torch.set_num_threads(args.threads)
args.output.mkdir(parents=True, exist_ok=False)

# The official checkpoint includes NumPy scalar optimizer metadata.
# Allow only its observed types while retaining PyTorch's weights-only loader.
with torch.serialization.safe_globals([
    (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
    np.dtype,
    type(np.dtype('float64')),
]):
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
weights = {
    key.removeprefix('gene_projector.'): value
    for key, value in checkpoint['model_state_dict'].items()
    if key.startswith('gene_projector.')
}
del checkpoint
esm = torch.load(args.esm_table, map_location='cpu', weights_only=True)
gene_symbols = sorted(esm)
with args.gene_metadata.open() as handle:
    kg_genes = {row['ESM'] for row in csv.DictReader(handle)}

# Export the raw gene branch before the contrastive loss's L2 normalization.
# Widths come from the trained tensors; no new projection is trained here.
outputs = []
with torch.inference_mode():
    for start in range(0, len(gene_symbols), args.batch_size):
        x = torch.stack([esm[key] for key in gene_symbols[start:start + args.batch_size]])
        hidden = F.linear(x, weights['mlp.0.weight'], weights['mlp.0.bias'])
        hidden = F.layer_norm(hidden, (hidden.shape[-1],), weights['mlp.1.weight'], weights['mlp.1.bias'])
        hidden = F.gelu(hidden)
        transformed = F.linear(hidden, weights['mlp.4.weight'], weights['mlp.4.bias'])
        residual = F.linear(x, weights['residual_proj.weight'], weights['residual_proj.bias'])
        gate = torch.sigmoid(F.linear(x, weights['gate.0.weight'], weights['gate.0.bias']))
        outputs.append(gate * transformed + (1 - gate) * residual)
embeddings = torch.cat(outputs)
if not torch.isfinite(embeddings).all():
    raise ValueError('Non-finite protein representations; bank was not saved')

# Keep row identity explicit. Gene symbols still require panel/UniProt mapping.
torch.save(embeddings, args.output / 'embeddings.pt')
(args.output / 'gene_symbols.json').write_text(json.dumps(gene_symbols, indent=2) + '\n')
manifest = {
    'bank_version': args.output.name,
    'provider': 'MAP gene_projector',
    'upstream_repository': 'https://github.com/MAGIC-AI4Med/MAP',
    'upstream_revision': '629ebdc1617eaf89825185a1512d772eccc3dc7e',
    'key_type': 'gene_symbol',
    'rows': len(gene_symbols),
    'input_dimension': weights['mlp.0.weight'].shape[1],
    'dimension': embeddings.shape[1],
    'dtype': str(embeddings.dtype),
    'normalization': 'raw projector output; no L2 normalization',
    'genes_in_map_kg_metadata': len(kg_genes),
    'bank_genes_in_map_kg_metadata': len(set(gene_symbols) & kg_genes),
    'map_kg_genes_without_esm': sorted(kg_genes - set(gene_symbols)),
    'marker_registry_mapping': 'pending panel-specific canonical protein/measured-form mapping',
    'inputs': {},
    'outputs': {},
    'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__, 'device': 'cpu'},
}
for name, path in [('checkpoint', args.checkpoint), ('esm_table', args.esm_table), ('gene_metadata', args.gene_metadata)]:
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    manifest['inputs'][name] = {'path': str(path.resolve()), 'sha256': digest}
for name in ['embeddings.pt', 'gene_symbols.json']:
    with (args.output / name).open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    manifest['outputs'][name] = {'sha256': digest}
(args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'output': str(args.output), 'shape': list(embeddings.shape), 'finite': True}))
