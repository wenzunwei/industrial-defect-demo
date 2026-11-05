"""Check that prepared targets retain foreground and configs bind the new manifest."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jointcv.data import instances_from_mask
from jointcv.preparation import prepare_data
from jointcv.preparation_io import read_image,write_image


class PreparationTests(unittest.TestCase):
    def test_small_real_file_pipeline_and_hash_binding(self):
        cache = ROOT/'.cache/tests'
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            root = Path(temporary).resolve()
            self.assertTrue(root.is_relative_to(cache.resolve()))
            for split,count in [('train',4),('test',2)]:
                for i in range(count):
                    image = np.full((12,9,3),i+1+(10 if split=='test' else 0),np.uint8)
                    mask = np.zeros((12,9),np.uint8)
                    if i%2:
                        mask[3:7,2:4]=255
                    write_image(root/f'raw/{split}/{i}.png',image)
                    write_image(root/f'raw/{split}/{i}_GT.png',mask)
            config = {'data':{'raw_dir':'raw','prepared_dir':'prepared','seed':42,
                              'val_fraction':.5,'connectivity':8,'preview_count':0,
                              'expected':{'train':{'images':4,'positive':2},'test':{'images':2,'positive':1}}}}
            prepare_data(config,root)
            manifest = root/'prepared/manifest.json'
            records = json.loads(manifest.read_text(encoding='utf-8'))['samples']
            self.assertEqual([sum(r['split']==s for r in records) for s in ('train','val','test')],[2,2,2])
            for record in records:
                mask = read_image(root/'prepared'/record['mask'],0)
                target = instances_from_mask(mask)
                np.testing.assert_array_equal(target['masks'].any(0).numpy(),mask>0)
            script = ROOT/'scripts/prepare_data.py'
            spec = importlib.util.spec_from_file_location('prepare_cli',script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config_file = root/'m1.yaml'
            config_file.write_text('data: {}\n',encoding='utf-8')
            digest = module.bind_configs(manifest,[config_file])
            bound = yaml.safe_load(config_file.read_text())['data']
            self.assertEqual(bound['manifest_sha256'],digest)
            self.assertEqual((ROOT/bound['manifest']).resolve(),manifest)


if __name__=='__main__':
    unittest.main()
