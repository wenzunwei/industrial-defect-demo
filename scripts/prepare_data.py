"""Prepare audited KSDD2 splits and bind their checksum to training configs."""
import argparse
import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jointcv.preparation import prepare_data


def bind_configs(manifest, paths):
    manifest = manifest.resolve()
    relative = manifest.relative_to(ROOT).as_posix()
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for path in paths:
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Config must stay inside this checkout')
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        config['data'] = {'manifest':relative, 'manifest_sha256':digest}
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=ROOT/'data/raw/KolektorSDD2')
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/ksdd2')
    args = parser.parse_args()
    raw, output = args.raw_dir.resolve(), args.output.resolve()
    if not raw.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError('Use raw/output paths inside this checkout')
    config = {'data':{'raw_dir':raw.relative_to(ROOT).as_posix(),
                      'prepared_dir':output.relative_to(ROOT).as_posix(),
                      'seed':42, 'val_fraction':.2, 'connectivity':8, 'preview_count':8,
                      'expected':{'train':{'images':2331,'positive':246},
                                  'test':{'images':1004,'positive':110}}}}
    prepare_data(config, ROOT, output)
    digest = bind_configs(output/'manifest.json', [ROOT/'configs/m0.yaml',ROOT/'configs/m1.yaml'])
    print(f'Dataset prepared; configs/m0.yaml and m1.yaml bound to SHA256 {digest}')


if __name__=='__main__':
    main()
