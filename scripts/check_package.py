"""Read-only audit of the upload file layout and portable model artifacts."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [p for p in ROOT.rglob('*') if p.is_file()
             and not {'.git','.cache','__pycache__'}.intersection(p.relative_to(ROOT).parts)]
    forbidden = [p for p in files if p.suffix.lower() in {'.pt','.pth','.ckpt','.zip','.whl'}]
    if forbidden:
        raise ValueError(f'Unwanted binary files: {forbidden}')
    if any(p for p in (ROOT/'data').rglob('*') if p.is_file() and p.name!='README.md'):
        raise ValueError('Upload data folder must contain documentation only')
    raw = sorted((ROOT/'examples/inputs').glob('*'))
    if [p.name for p in raw]!=['example_1.png','example_2.png','example_3.png']:
        raise ValueError('Keep exactly three input images')
    weights = [p for p in files if p.suffix=='.onnx']
    if weights!=[ROOT/'models/model.onnx']:
        raise ValueError('Only the final single ONNX weight belongs in the upload')
    manifest = json.loads((ROOT/'models/manifest.json').read_text(encoding='utf-8'))
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(weights[0])==manifest['onnx_sha256']
    assert digest(ROOT/'inference/runtime.py')==manifest['runtime_sha256']
    selection = json.loads((ROOT/'examples/selection.json').read_text(encoding='utf-8'))
    for entry in selection['examples']:
        assert digest(ROOT/'examples'/entry['file'])==entry['original_sha256']
    for path in files:
        if path.suffix=='.py':
            ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    page = (ROOT/'docs/index.html').read_text(encoding='utf-8')
    assert len(re.findall(r'<figure',page))==5
    assert all((ROOT/'docs'/name).is_file() for name in re.findall(r'src="([^"]+)"',page))
    for term in ('训练损失','联合最佳','验证集','样例','误差分析','五列'):
        assert term not in page
    for match in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',(ROOT/'README.md').read_text(encoding='utf-8')):
        assert (ROOT/match).is_file(),match
    print(json.dumps({'status':'passed','files':len(files),'input_images':len(raw),
                      'onnx_weights':len(weights),'onnx_mib':round(weights[0].stat().st_size/2**20,2)},ensure_ascii=False))


if __name__=='__main__':
    main()
