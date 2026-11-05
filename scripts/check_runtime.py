"""Check inference in a child process that cannot import training libraries."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/demo')
    parser.add_argument('--report',type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    child = r'''
import importlib.abc,runpy,sys
blocked={'torch','torchvision','onnx','jointcv','ultralytics','segmentation_models_pytorch'}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in blocked:
            raise ImportError('Training/export dependency forbidden: '+fullname)
sys.meta_path.insert(0,Guard())
script,inputs,output=sys.argv[1:]
sys.path.insert(0,str(__import__('pathlib').Path(script).parent))
sys.argv=[script,'--input',inputs,'--output',output]
runpy.run_path(script,run_name='__main__')
assert not blocked.intersection(sys.modules)
'''
    result = subprocess.run([sys.executable,'-X','utf8','-s','-B','-c',child,
                             str(ROOT/'infer.py'),str(ROOT/'examples/inputs'),str(output)],
                            check=True,capture_output=True,text=True,encoding='utf-8')
    selection = json.loads((ROOT/'examples/selection.json').read_text(encoding='utf-8'))
    rows = []
    for entry in selection['examples']:
        name = Path(entry['file']).name
        mask = output/f'{name}.mask.png'
        observed = hashlib.sha256(mask.read_bytes()).hexdigest()
        if observed!=entry['expected_prediction']['mask_sha256']:
            raise AssertionError(f'{name}: mask differs from verified CPU prediction')
        rows.append({'input':entry['file'],'status':'passed','mask_sha256':observed})
    report = {'status':'passed','provider':'CPUExecutionProvider','training_imports_blocked':True,
              'images':len(rows),'cases':rows,'stdout':result.stdout.strip()}
    target = args.report or output/'runtime_acceptance.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','images':len(rows),'report':str(target)},ensure_ascii=False))


if __name__=='__main__':
    main()
