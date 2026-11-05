"""Compare an exported model to its source checkpoint using supplied images."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.runtime import MaskRCNNOnnx, read_image
from jointcv.commands import load_trained
from jointcv.common import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'exports/maskrcnn')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--input', type=Path, default=ROOT/'examples/inputs')
    args = parser.parse_args()
    folder = args.directory.resolve()
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    weights = args.weights or Path(manifest['source_checkpoint'])
    if not weights.is_absolute():
        weights = ROOT/weights
    if sha256(weights)!=manifest['source_sha256']:
        raise ValueError('Checkpoint checksum does not match this ONNX')
    torch.set_num_threads(2)
    model, _, _ = load_trained(weights, torch.device('cpu'))
    runtime = MaskRCNNOnnx(folder/manifest['file'], allow_unverified=True)
    paths = [args.input] if args.input.is_file() else sorted(args.input.rglob('*.png'))
    if not paths:
        raise ValueError('No input images')
    cases = [(p.name,read_image(p),'input') for p in paths]
    base = cases[-1][1]
    center = base.shape[0]//2
    crop = base[max(0,center-150):min(base.shape[0],center+150)]
    cases += [('black',np.zeros((480,192,3),np.uint8),'probe'),
              ('landscape',np.ascontiguousarray(np.rot90(base)),'probe'),
              ('square',cv2.resize(base,(512,512)),'probe'),
              ('stacked',np.ascontiguousarray(np.vstack((crop,crop))),'probe')]
    rows = []
    for name, image, purpose in cases:
        tensor = torch.from_numpy(np.ascontiguousarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).transpose(2,0,1))).float()/255
        with torch.inference_mode():
            pt = {k:v.cpu() for k,v in model([tensor])[0].items()}
        keep = pt['scores']>=manifest['embedded_score_floor']
        pt = {k:v[keep] for k,v in pt.items()}
        actual = runtime.raw(image)
        n = len(pt['scores'])
        if n!=len(actual['scores']):
            raise AssertionError(f'{name}: detection count mismatch')
        matches, used = [], set()
        if n:
            overlap = box_iou(pt['boxes'],torch.from_numpy(actual['boxes']))
            for i in range(n):
                value,j = max((float(overlap[i,j]),j) for j in range(n) if j not in used)
                if value<.999:
                    raise AssertionError(f'{name}: box overlap={value}')
                used.add(j); matches.append(j)
        order = np.asarray(matches,dtype=int)
        if not np.array_equal(pt['labels'].numpy(),actual['labels'][order]):
            raise AssertionError(f'{name}: class mismatch')
        box_error = float(np.abs(pt['boxes'].numpy()-actual['boxes'][order]).max()) if n else 0.
        score_error = float(np.abs(pt['scores'].numpy()-actual['scores'][order]).max()) if n else 0.
        a = pt['masks'][:,0].numpy().max(0)>=.5 if n else np.zeros(image.shape[:2],bool)
        b = actual['masks'][:,0].max(0)>=.5 if n else np.zeros(image.shape[:2],bool)
        union = np.count_nonzero(a|b)
        iou = float(np.count_nonzero(a&b)/union) if union else 1.
        agreement = float(np.mean(a==b))
        if box_error>.05 or score_error>1e-4 or iou<.995 or agreement<.9995:
            raise AssertionError(f'{name}: box={box_error}, score={score_error}, mask_iou={iou}')
        row = {'name':name,'purpose':purpose,'detections':n,'box_max_abs_error_px':box_error,
               'score_max_abs_error':score_error,'mask_binary_iou':iou,'mask_pixel_agreement':agreement,'status':'passed'}
        rows.append(row)
        print(row,flush=True)
    onnx.checker.check_model(str(folder/manifest['file']),full_check=True)
    report = {'status':'passed','provider':'CPUExecutionProvider','case_count':len(rows),'cases':rows,
              'onnx_sha256':manifest['onnx_sha256'],'source_sha256':manifest['source_sha256'],
              'empty_output_cases':sum(r['detections']==0 for r in rows),
              'multi_detection_cases':sum(r['detections']>1 for r in rows),'test_evaluated':False}
    write_json(folder/'verification.json',report)
    manifest.update(status='verified_cpu',verification='verification.json')
    write_json(folder/'manifest.json',manifest)
    (folder/'INCOMPLETE').unlink(missing_ok=True)
    print({'status':'passed','cases':len(rows)},flush=True)


if __name__=='__main__':
    main()
