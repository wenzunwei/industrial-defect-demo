"""Build a minimal presentation preview from the three verified ONNX outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE = WORKSPACE
FONT = Path('C:/Windows/Fonts/msyh.ttc')
BOLD = Path('C:/Windows/Fonts/msyhbd.ttc')
INK = '#172b32'
TEAL = '#168878'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def display_crop(original, boxes, mask):
    h, w = original.shape[:2]
    ys = np.flatnonzero(mask.any(axis=1))
    bounds = [float(y) for box in boxes for y in (box[1], box[3])]
    if len(ys):
        bounds.extend((float(ys[0]), float(ys[-1]+1)))
    low, high = (min(bounds), max(bounds)) if bounds else (h/2, h/2)
    crop_height = min(h, max(360, int(np.ceil(high-low))+96))
    top = int(np.clip(round((low+high-crop_height)/2), 0, h-crop_height))
    bottom = top+crop_height
    if int(mask[top:bottom].sum()) != int(mask.sum()):
        raise ValueError('Display crop would remove predicted foreground')
    if any(box[1] < top or box[3] > bottom for box in boxes):
        raise ValueError('Display crop would truncate a detection')
    return (0, top, w, bottom)


def panel_image(original, boxes, mask, destination, crop_box):
    detected = Image.fromarray(original.copy())
    pen = ImageDraw.Draw(detected)
    for x1, y1, x2, y2 in boxes:
        pen.rectangle((round(x1), round(y1), round(x2)-1, round(y2)-1),
                      outline=(55, 222, 178), width=2)
    segmented = original.copy()
    color = np.array([43, 207, 185], dtype=np.float32)
    segmented[mask] = np.rint(.58*original[mask]+.42*color).astype(np.uint8)
    panels = [image.crop(crop_box) for image in (Image.fromarray(original), detected, Image.fromarray(segmented))]
    width, height, margin, gap = 1080, 600, 24, 18
    cell = (width-margin*2-gap*2)//3
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (title, image) in enumerate(zip(('原图', '目标检测', '缺陷分割'), panels)):
        left = margin+i*(cell+gap)
        draw.text((left+cell/2, 29), title, font=font(23, True), fill=INK, anchor='mm')
        draw.rounded_rectangle((left, 65, left+cell, height-24), radius=11, fill='#f5f7f8')
        resized = ImageOps.contain(image, (cell-24, height-117), Image.Resampling.LANCZOS)
        x = left+(cell-resized.width)//2
        y = 65+(height-89-resized.height)//2
        canvas.paste(resized, (x, y))
    canvas.save(destination, optimize=True)
    return canvas


def ema(values, alpha=.35):
    values = np.asarray(values, dtype=float)
    output = values.copy()
    for i in range(1, len(values)):
        output[i] = alpha*values[i]+(1-alpha)*output[i-1]
    return output


def style_axis(ax, epochs, title, ylabel):
    ax.set_title(title, loc='left', fontsize=19, fontweight='bold', color=INK, pad=24)
    ax.set_xlabel('Epoch', labelpad=12, color='#667780')
    ax.set_ylabel(ylabel, labelpad=12, color='#667780')
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_xticks([1,5,10,15,21] if epochs[-1]==21 else np.unique(np.linspace(epochs[0],epochs[-1],min(5,len(epochs)),dtype=int)))
    ax.grid(axis='y', color='#e9edef', linewidth=.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#dfe6e9')
    ax.tick_params(axis='both', colors='#70818a', length=0, pad=10, labelsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(5))


def make_charts(history, output):
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(FONT)).get_name(),
                         'axes.unicode_minus': False, 'svg.fonttype': 'path',
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    epochs = np.array([row['epoch'] for row in history])
    if epochs.tolist() != list(range(1, len(history)+1)) or len(history)<2:
        raise ValueError('Use consecutive training history starting at epoch 1')
    total = np.array([row['train_loss'] for row in history])
    smoothed = ema(total)
    rates = np.array([row['lr'] for row in history])
    for stem, values, title, ylabel, color in (
        ('training_loss', smoothed, '损失', 'Loss', TEAL),
        ('learning_rate', rates, '学习率', 'Learning rate', '#596c7d'),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 4.7))
        fig.subplots_adjust(left=.13, right=.96, top=.79, bottom=.17)
        style_axis(ax, epochs, title, ylabel)
        ax.plot(epochs, values, color=color, linewidth=2.8,
                solid_joinstyle='round', solid_capstyle='round', antialiased=True,
                label='总损失' if stem=='training_loss' else None)
        if stem=='training_loss':
            ax.set_ylim(bottom=0, top=float(max(values))*1.12)
            ax.fill_between(epochs, values, 0, color=color, alpha=.045)
            ax.legend(loc='upper right', frameon=False, fontsize=10, labelcolor='#667780')
        else:
            ax.set_ylim(bottom=0, top=float(max(values))*1.18)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
        for extension in ('png', 'svg'):
            fig.savefig(output/f'{stem}.{extension}', dpi=190, facecolor='white')
        plt.close(fig)
    charts = Image.new('RGB', (1615*2+24, 893), 'white')
    for i, stem in enumerate(('training_loss', 'learning_rate')):
        charts.paste(Image.open(output/f'{stem}.png').convert('RGB'), (i*(1615+24), 0))
    charts.save(output/'training_curves.png', optimize=True)
    with (output/'curve_values.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'total_loss_raw', 'total_loss_display_ema_alpha_0.35', 'learning_rate'])
        writer.writerows(zip(epochs, total, smoothed, rates))


def make_page(output):
    content = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>工业表面缺陷检测与分割</title>
<style>
*{box-sizing:border-box}html{color-scheme:light}body{margin:0;background:#f5f7f8;color:#172b32;font-family:"Microsoft YaHei",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
main{width:min(1120px,calc(100% - 48px));margin:0 auto;padding:62px 0 72px}header{display:flex;align-items:center;gap:17px;margin-bottom:54px}
.mark{width:47px;height:47px;display:grid;place-items:center;border:1px solid #dce9e6;border-radius:14px;background:#ecf5f2;color:#168878;flex:none}
h1{font-size:clamp(22px,3vw,34px);font-weight:650;letter-spacing:-.6px;line-height:1.4;margin:0}
section+section{margin-top:54px}h2{display:flex;align-items:center;gap:18px;font-size:19px;font-weight:600;margin:0 0 23px}h2::after{content:"";height:1px;flex:1;background:#dfe6e8}
.results{display:grid;gap:26px}.card{margin:0;padding:12px;background:white;border:1px solid #e1e7e9;border-radius:18px;overflow:hidden;box-shadow:0 7px 22px #253d4a04}
.card img{display:block;width:100%;height:auto;border-radius:8px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:22px}.chart{margin:0;padding:0;border:0;border-radius:0}.chart img{display:block;width:100%;height:auto}
@media(max-width:700px){main{width:calc(100% - 24px);padding-top:30px}header{gap:12px;margin-bottom:32px}.mark{width:38px;height:38px;border-radius:11px}.card{padding:3px;border-radius:12px}.results{gap:16px}.charts{grid-template-columns:1fr;gap:15px}section+section{margin-top:34px}h2{font-size:17px;margin-bottom:17px}}
</style></head><body><main><header><div class="mark" aria-hidden="true"><svg width="25" height="25" viewBox="0 0 24 24" fill="none"><path d="M8 3H3v5M16 3h5v5M21 16v5h-5M8 21H3v-5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="m8 13 3-4 3 2 2-1v6H8v-3Z" fill="currentColor" opacity=".75"/></svg></div><h1>工业表面缺陷检测与分割</h1></header>
<section><h2>效果展示</h2><div class="results">
<figure class="card"><img src="assets/result_a.png" alt="原图、目标检测与缺陷分割" width="1080" height="600"></figure>
<figure class="card"><img src="assets/result_b.png" alt="原图、目标检测与缺陷分割" width="1080" height="600" loading="lazy"></figure>
<figure class="card"><img src="assets/result_c.png" alt="原图、目标检测与缺陷分割" width="1080" height="600" loading="lazy"></figure>
</div></section><section><h2>训练过程</h2><div class="charts">
<figure class="chart"><img src="assets/training_loss.png" alt="总损失" width="1615" height="893" loading="lazy"></figure>
<figure class="chart"><img src="assets/learning_rate.png" alt="学习率" width="1615" height="893" loading="lazy"></figure>
</div></section></main></body></html>'''
    (output/'index.html').write_text(content, encoding='utf-8')
    (output/'README_展示片段.md').write_text('''## 效果展示

![原图、目标检测与缺陷分割](assets/result_a.png)

![原图、目标检测与缺陷分割](assets/result_b.png)

![原图、目标检测与缺陷分割](assets/result_c.png)

## 训练过程

<p>
  <img src="assets/training_loss.png" width="49%" alt="损失">
  <img src="assets/learning_rate.png" width="49%" alt="学习率">
</p>
''', encoding='utf-8')


def overview(output, cards):
    width = 1200
    height = 230+sum(card.height+26 for card in cards)+460
    image = Image.new('RGB', (width, height), '#f5f7f8')
    draw = ImageDraw.Draw(image)
    draw.text((60, 48), '工业表面缺陷检测与分割', font=font(35, True), fill=INK)
    draw.text((60, 135), '效果展示', font=font(24, True), fill=INK)
    y = 190
    for card in cards:
        image.paste(card, ((width-card.width)//2, y))
        y += card.height+26
    draw.text((60, y+25), '训练过程', font=font(24, True), fill=INK)
    for i, name in enumerate(('training_loss', 'learning_rate')):
        chart = Image.open(output/f'assets/{name}.png').convert('RGB')
        chart = ImageOps.contain(chart, (530, 340), Image.Resampling.LANCZOS)
        image.paste(chart, (60+i*550, y+85))
    image.save(output/'overview.jpg', quality=93, subsampling=0)


def load_history(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [{'epoch':int(r['epoch']), 'train_loss':float(r['train_loss']), 'lr':float(r['lr'])} for r in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORKSPACE/'outputs/presentation')
    parser.add_argument('--results', type=Path, default=WORKSPACE/'outputs/demo')
    parser.add_argument('--history', type=Path, default=WORKSPACE/'experiments/m1/history.csv')
    args = parser.parse_args()
    output = args.output.resolve()
    if WORKSPACE not in output.parents or output.exists():
        raise ValueError('Choose a new output folder within this checkout')
    selection = read_json(PACKAGE/'examples/selection.json')
    model = read_json(WORKSPACE/'models/manifest.json')
    results = args.results.resolve()
    runtime = read_json(results/'summary.json')
    history = load_history(args.history)
    if model['status']!='verified_cpu' or runtime['model_sha256']!=model['onnx_sha256']:
        raise ValueError('Use the verified model outputs')
    if [r['sample_id'] for r in selection['examples']]!=['10457','11258','10401']:
        raise ValueError('Keep the three previously chosen inputs')
    assets = output/'assets'
    assets.mkdir(parents=True)
    cards, sources = [], []
    for entry, stem, prediction in zip(selection['examples'], ('result_a','result_b','result_c'), runtime['results']):
        path = PACKAGE/'examples'/entry['file']
        if digest(path)!=entry['original_sha256'] or Path(prediction['input']).name!=path.name:
            raise ValueError('Input identity mismatch')
        original = read_rgb(path)
        mask_path = results/f'{path.name}.mask.png'
        mask = np.array(Image.open(mask_path).convert('L'))>0
        if list(original.shape[:2])!=prediction['shape_hw'] or mask.shape!=original.shape[:2]:
            raise ValueError('Unexpected output geometry')
        if int(mask.sum())!=prediction['foreground_pixels']:
            raise ValueError('Stored predicted mask changed')
        destination = assets/f'{stem}.png'
        crop_box = display_crop(original, prediction['boxes_xyxy'], mask)
        cards.append(panel_image(original, prediction['boxes_xyxy'], mask, destination, crop_box))
        sources.append({'sample_id':entry['sample_id'], 'original_sha256':digest(path),
                        'mask_sha256':digest(mask_path), 'display_crop_xyxy':list(crop_box),
                        'asset':destination.relative_to(output).as_posix()})
    make_charts(history, assets)
    make_page(output)
    overview(output, cards)
    # Presentation settings and provenance live in metadata, outside the page.
    evidence = {'source_onnx_sha256':model['onnx_sha256'], 'cases':sources,
                'history_sha256':digest(args.history),
                'loss_display':{'method':'EMA','alpha':.35,'epochs':len(history)},
                'learning_rate_display':'Original epoch-end recorded values, no smoothing',
                'curve_values':'assets/curve_values.csv', 'model_predictions_modified':False,
                'asset_sha256':{p.relative_to(output).as_posix():digest(p) for p in sorted(assets.iterdir())}}
    (output/'render_provenance.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'completed','output':str(output),'cases':len(cards),'charts':2},ensure_ascii=False))


if __name__=='__main__':
    main()
