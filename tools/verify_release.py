import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLITS = {
    'KSDD2': {'train': 2098, 'val': 234, 'test': 1004},
    'MTD_binary': {'train': 856, 'val': 214, 'test': 270},
    'NEU_Seg': {'train': 2904, 'val': 726, 'test': 840},
}


def read_ids(path):
    return [line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def main():
    errors = []
    for dataset, split_counts in EXPECTED_SPLITS.items():
        split_sets = {}
        for split, expected_count in split_counts.items():
            path = ROOT / 'data' / dataset / 'splits' / f'{split}.txt'
            if not path.is_file():
                errors.append(f'Missing split file: {path.relative_to(ROOT)}')
                continue
            ids = read_ids(path)
            split_sets[split] = set(ids)
            if len(ids) != expected_count:
                errors.append(
                    f'{dataset}/{split}: expected {expected_count}, found {len(ids)}')
            if len(ids) != len(split_sets[split]):
                errors.append(f'{dataset}/{split}: duplicate identifiers found')
        if 'train' in split_sets and 'val' in split_sets:
            overlap = split_sets['train'] & split_sets['val']
            if overlap:
                errors.append(f'{dataset}: train/val overlap contains {len(overlap)} identifiers')

    metrics_path = ROOT / 'results' / 'expected_metrics.csv'
    with metrics_path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected_columns = {
        'dataset', 'variant', 'config_relative_path', 'metric', 'reported_value'}
    if not rows:
        errors.append('results/expected_metrics.csv is empty')
    elif set(rows[0]) != expected_columns:
        errors.append('results/expected_metrics.csv has unexpected columns')
    seen = set()
    for row in rows:
        key = (row['dataset'], row['variant'])
        if key in seen:
            errors.append(f'Duplicate result mapping: {key[0]} / {key[1]}')
        seen.add(key)
        config = ROOT / Path(row['config_relative_path'])
        if not config.is_file():
            errors.append(f'Missing configuration: {row["config_relative_path"]}')

    binaries = list(ROOT.rglob('*.pth'))
    if binaries:
        errors.append(f'Found {len(binaries)} trained checkpoint file(s) in the release tree')

    if errors:
        print('Release verification failed:')
        for error in errors:
            print(f'- {error}')
        raise SystemExit(1)

    print('Release verification passed.')
    print(f'- Result mappings: {len(rows)}')
    print('- Fixed split manifests: 9')
    print('- Trained checkpoint files: 0')


if __name__ == '__main__':
    main()
