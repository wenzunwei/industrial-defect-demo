"""Audit exported resize geometry over train/val sizes without model inference."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision.models.detection.transform import _resize_image_and_masks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointcv.common import WORKSPACE, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=WORKSPACE/'exports/maskrcnn')
    args = parser.parse_args()
    folder = args.directory.resolve()
    metadata = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    source = json.loads((WORKSPACE/'data/processed/ksdd2/manifest.json').read_text(encoding='utf-8'))
    dimensions = sorted({(r['height'], r['width']) for r in source['samples'] if r['split'] in ('train', 'val')})
    dimensions.extend([(232, 645), (512, 512), (480, 192)])
    graph = onnx.load(str(folder/metadata['file']))
    resize = next(n for n in graph.graph.node if n.op_type == 'Resize')
    graph.graph.node.append(onnx.helper.make_node('Shape', [resize.output[0]], ['audit_resize_shape']))
    del graph.graph.output[:]
    graph.graph.output.append(onnx.helper.make_tensor_value_info('audit_resize_shape', onnx.TensorProto.INT64, [4]))
    # Explicitly prune unused detection branches. ORT otherwise may schedule
    # disconnected legacy nodes even though the only requested output is shape.
    needed = {'audit_resize_shape'}
    keep_nodes = []
    for node in reversed(graph.graph.node):
        if needed.intersection(node.output):
            keep_nodes.append(node)
            needed.update(node.input)
    del graph.graph.node[:]
    graph.graph.node.extend(reversed(keep_nodes))
    keep_weights = [value for value in graph.graph.initializer if value.name in needed]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(keep_weights)
    torch.set_num_threads(2)
    rows = []
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.log_severity_level = 3
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=opts, providers=['CPUExecutionProvider'])
    for h, w in dimensions:
        x = torch.zeros(3, h, w)
        native, _ = _resize_image_and_masks(x, metadata['internal_resize']['min_size'], metadata['internal_resize']['max_size'])
        observed = session.run(None, {'images': np.zeros((1, 3, h, w), np.float32)})[0][2:].tolist()
        expected = list(native.shape[-2:])
        if observed != expected:
            raise AssertionError(f'{h}x{w}: ONNX {observed} != PyTorch {expected}')
        rows.append({'input_hw': [h, w], 'resized_hw': observed, 'status': 'passed'})
    report = {'status': 'passed', 'onnx_sha256': sha256(folder/metadata['file']),
              'geometry_count': len(rows), 'scope': 'All distinct train/val H/W plus landscape/square/black probe sizes; no test images or annotations read.',
              'note': 'Shape-only check of the actual exported first Resize, not full model predictions.', 'cases': rows}
    write_json(folder/'resize_geometry.json', report)
    print({'status': 'passed', 'geometries': len(rows)}, flush=True)


if __name__ == '__main__':
    main()
