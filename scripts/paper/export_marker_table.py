"""Export the five-column manuscript marker table from the accepted CSV mappings."""
import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lookup', type=Path, required=True)
parser.add_argument('--alignment', type=Path, required=True)
parser.add_argument('--recipes', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True, help='New directory for the dated export')
args = parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


lookup_fields, lookup = read_csv(args.lookup)
alignment_fields, alignment = read_csv(args.alignment)
assert lookup_fields == ['aliases', 'marker', 'n_studies', 'map_kg', 'uniprot_id']
assert alignment_fields == ['study', 'panel', 'alias', 'marker']
by_marker = {row['marker']: row for row in lookup}
assert len(by_marker) == len(lookup)
alias_map = {}
for row in lookup:
    for alias in row['aliases'].split(' | '):
        assert alias not in alias_map, alias
        alias_map[alias] = row['marker']

# The display status belongs in the unified-name column; encoding fields stay empty.
display = {}
marker_status = {}
mean_prefixes = {'均值', '成员均值', '家族均值', '家族参考均值', '分组代表均值'}
single_prefixes = {'基因代表', '基础蛋白代表', '指定代表', '代表', '替代'}
for marker, row in by_marker.items():
    mapping = row['map_kg']
    if marker == '排除':
        display[marker] = ('Excluded', '', '')
        marker_status[marker] = 'excluded'
    elif mapping.startswith('仅分割'):
        display[marker] = (marker + ' (segmentation only)', '', '')
        marker_status[marker] = 'segmentation_only'
    elif mapping.startswith('待确认'):
        display[marker] = (marker + ' (mapping pending)', '', '')
        marker_status[marker] = 'mapping_pending'
    else:
        if '(' in mapping:
            match = re.fullmatch(r'([^()]+)\(([^()]+)\)', mapping)
            assert match is not None, (marker, mapping)
            operation, content = match.groups()
            members = content.split(',')
            assert len(members) == len(set(members)), marker
            if operation in mean_prefixes:
                encoded = 'mean(' + ', '.join(members) + ')'
            else:
                assert operation in single_prefixes and len(members) == 1, (marker, mapping)
                encoded = 'proxy(' + members[0] + ')' if operation == '替代' else members[0]
        else:
            assert re.fullmatch(r'[A-Za-z0-9-]+', mapping), (marker, mapping)
            encoded = mapping
        assert row['uniprot_id'], marker
        display[marker] = (marker, encoded, row['uniprot_id'])
        marker_status[marker] = 'mapped'

unique = {}
source_counts = Counter()
studies_by_marker = defaultdict(set)
for row in alignment:
    assert alias_map[row['alias']] == row['marker'], row
    original = row['alias'].removesuffix(' [' + row['study'] + ']')
    key = (row['study'], original)
    values = (row['study'], original, *display[row['marker']])
    if key in unique:
        assert unique[key] == values, f'Panel-dependent mapping requires review: {key}'
    unique[key] = values
    source_counts[marker_status[row['marker']]] += 1
    studies_by_marker[row['marker']].add(row['study'])
assert all(int(r['n_studies']) == len(studies_by_marker[r['marker']]) for r in lookup)

fields = ['Study', 'Original marker', 'Unified marker', 'MAP-KG mapping', 'UniProt accession']
table = [dict(zip(fields, unique[key])) for key in sorted(unique, key=lambda k: (k[0].casefold(), k[1].casefold(), k))]
args.output.mkdir(parents=True, exist_ok=False)
source_dir = args.output / 'source_tables'
source_dir.mkdir()
for path in (args.lookup, args.alignment, args.recipes):
    shutil.copyfile(path, source_dir / path.name)
table_path = args.output / 'supplementary_table_s1_marker_mapping.csv'
with table_path.open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(table)
assert read_csv(table_path) == (fields, table)
output_counts = Counter()
for row in table:
    unified = row['Unified marker']
    status = ('excluded' if unified == 'Excluded' else 'segmentation_only' if unified.endswith(' (segmentation only)')
              else 'mapping_pending' if unified.endswith(' (mapping pending)') else 'mapped')
    output_counts[status] += 1
    if status != 'mapped':
        assert row['MAP-KG mapping'] == row['UniProt accession'] == ''
summary = {
    'table': table_path.name, 'columns': fields, 'studies': len({r['Study'] for r in table}),
    'source_alignment_rows': len(alignment), 'source_lookup_rows': len(lookup), 'table_rows': len(table),
    'source_rows_by_status': dict(source_counts), 'table_rows_by_status': dict(output_counts),
    'marker_categories_by_status': dict(Counter(marker_status.values())),
    'deduplication': 'Identical mappings for the same study and original marker are listed once across panels.',
    'study_scope_suffix_removed': True, 'within_study_mapping_conflicts': 0,
    'inputs': {name: str(getattr(args, name).resolve()) for name in ['lookup', 'alignment', 'recipes']},
}
(args.output / 'export_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
