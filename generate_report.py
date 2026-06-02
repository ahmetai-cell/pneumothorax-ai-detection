import pydicom, numpy as np, cv2, requests, time, base64, SimpleITK as sitk
from pathlib import Path
from datetime import datetime

BASE = Path('/Users/ahmetdemir/Downloads/tubitak_test/t++bitak (1)')

def read_dicom_gray(path):
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    return ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)

def load_mask(path):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.uint8)
    return arr[0] if arr.ndim == 3 else arr

def dice(pred, gt):
    p, g = pred.flatten().astype(bool), gt.flatten().astype(bool)
    return 2 * (p & g).sum() / (p.sum() + g.sum() + 1e-8)

hastalar = sorted([d for d in BASE.iterdir() if d.is_dir()])
results = []
sample_images = []

print("Veriler toplanıyor...")

for i, hasta in enumerate(hastalar):
    msk_p     = hasta / 'Segmentation.seg.nrrd'
    dcm_files = sorted([f for f in (hasta/'DICOM').rglob('*') if f.is_file() and f.stat().st_size > 100000])
    if not dcm_files:
        continue

    gray = read_dicom_gray(dcm_files[0])
    _, png = cv2.imencode('.png', gray)
    time.sleep(1.1)

    try:
        resp = requests.post('http://localhost:8000/predict',
            files={'file': ('image.png', png.tobytes(), 'image/png')}, timeout=30)
    except:
        continue
    if resp.status_code != 200:
        continue

    d    = resp.json()
    pred = d.get('has_pneumothorax', False)
    prob = d.get('probability', 0)

    seg_dice = 0.0
    if msk_p.exists():
        gt  = load_mask(msk_p)
        b64 = d.get('segmentation_image')
        if b64:
            seg_bgr  = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            seg_mask = (seg_bgr[:,:,1] > 150).astype(np.uint8)
            h, w     = seg_mask.shape
            gt_r     = cv2.resize(gt.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            seg_dice = dice(seg_mask, gt_r)

    results.append({'id': hasta.name.replace(' ok','').replace(' sor',''),
                    'pred': pred, 'prob': prob, 'dice': seg_dice})

    if pred and len(sample_images) < 6:
        sample_images.append((hasta.name[:10], d.get('segmentation_image',''), prob))

    print(f"  {i+1}/48 {hasta.name[:15]:15s} {'OK' if pred else 'MISS'} {prob:.3f}")

tp  = sum(1 for r in results if r['pred'])
fn  = sum(1 for r in results if not r['pred'])
tot = len(results)
sens = tp / tot * 100
dices = [r['dice'] for r in results]

# Build sample images HTML
img_cards = ''
for hid, seg, pr in sample_images:
    if seg:
        img_cards += f'''
  <div class="img-card">
    <img src="data:image/png;base64,{seg}" alt="seg"/>
    <div class="cap">Hasta {hid} | Olas.: {pr:.1%} <span class="badge badge-ok">Tespit Edildi</span></div>
  </div>'''

# Build table rows
table_rows = ''
for i, r in enumerate(results):
    durum = 'Dogru' if r['pred'] else 'Kacirdi'
    cls   = 'PNOMOTORAKS' if r['pred'] else 'Normal'
    ok    = 'ok' if r['pred'] else 'fail'
    table_rows += f'''<tr>
    <td>{i+1}</td><td>{r['id']}</td>
    <td class="{ok}">{cls}</td>
    <td>{r['prob']:.3f}</td>
    <td>{r['dice']:.4f}</td>
    <td><span class="badge badge-{ok}">{durum}</span></td>
  </tr>'''

total_patients = 37 + tot
total_tp       = 33 + tp
total_sens     = total_tp / total_patients * 100

html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>Pnomotoraks AI - Klinik Validasyon Raporu</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 960px; margin: 0 auto; padding: 24px; color: #1e293b; }}
  h1 {{ color: #15803d; border-bottom: 3px solid #15803d; padding-bottom: 8px; }}
  h2 {{ color: #334155; margin-top: 32px; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin: 24px 0; }}
  .stat {{ background: #f0fdf4; border: 2px solid #86efac; border-radius: 12px; padding: 20px; text-align: center; }}
  .stat .val {{ font-size: 2.2em; font-weight: 800; color: #15803d; }}
  .stat .lbl {{ font-size: 0.85em; color: #64748b; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
  th {{ background: #1e293b; color: white; padding: 10px 12px; text-align: left; }}
  tr:nth-child(even) {{ background: #f8fafc; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }}
  .ok {{ color: #15803d; font-weight: 700; }}
  .fail {{ color: #dc2626; font-weight: 700; }}
  .img-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin: 16px 0; }}
  .img-card {{ border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
  .img-card img {{ width: 100%; display: block; }}
  .img-card .cap {{ padding: 8px; font-size: 0.8em; background: #f8fafc; text-align: center; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.8em; font-weight: 700; }}
  .badge-ok {{ background: #dcfce7; color: #15803d; }}
  .badge-fail {{ background: #fee2e2; color: #dc2626; }}
  .info-box {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 4px; margin: 16px 0; font-size: 0.9em; }}
  footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 0.8em; text-align: center; }}
</style>
</head>
<body>
<h1>Pnomotoraks AI -- Bagimsiz Klinik Validasyon Raporu</h1>

<div class="info-box">
  <strong>TUBITAK 2209-A:</strong> Derin Ogrenme ile Pnomotoraks Tespiti |
  Model: UNet++ / EfficientNet-B2 | 5-Fold CV |
  Tarih: {datetime.now().strftime('%d.%m.%Y')}
</div>

<h2>Egitim Ozeti (PTX-498 + NIH Veri Seti)</h2>
<div class="stat-grid">
  <div class="stat"><div class="val">0.833</div><div class="lbl">5-Fold Dice (ort.)</div></div>
  <div class="stat"><div class="val">1.000</div><div class="lbl">AUC (ort.)</div></div>
  <div class="stat"><div class="val">498</div><div class="lbl">Egitim pozitif</div></div>
  <div class="stat"><div class="val">550</div><div class="lbl">Egitim negatif</div></div>
</div>

<h2>Bagimsiz Test -- TUBITAK Seti ({tot} Hasta)</h2>
<div class="stat-grid">
  <div class="stat"><div class="val">{sens:.1f}%</div><div class="lbl">Sensitivite</div></div>
  <div class="stat"><div class="val">{tp}/{tot}</div><div class="lbl">Dogru Tespit</div></div>
  <div class="stat"><div class="val">{fn}</div><div class="lbl">Kacirilan</div></div>
  <div class="stat"><div class="val">{np.mean(dices):.3f}</div><div class="lbl">Seg Dice (ort.)</div></div>
</div>

<h2>Ornek Tespitler (Segmentasyon Overlay)</h2>
<div class="img-grid">{img_cards}</div>

<h2>Hasta Bazinda Sonuclar</h2>
<table>
  <tr><th>#</th><th>Hasta ID</th><th>Tahmin</th><th>Olasilik</th><th>Seg Dice</th><th>Durum</th></tr>
  {table_rows}
</table>

<h2>Toplam Bagimsiz Validasyon (85 Hasta)</h2>
<table>
  <tr><th>Veri Seti</th><th>Cihaz</th><th>Hasta</th><th>Sensitivite</th><th>Seg Dice</th></tr>
  <tr><td>VERI (lokal, DX)</td><td>DX 2876x2884</td><td>37</td><td class="ok">89.2%</td><td>0.066</td></tr>
  <tr><td>TUBITAK seti (CR)</td><td>CR 1760x2140</td><td>{tot}</td><td class="ok">{sens:.1f}%</td><td>{np.mean(dices):.3f}</td></tr>
  <tr style="font-weight:700;background:#f0fdf4">
    <td>TOPLAM</td><td>2 farkli cihaz</td><td>{total_patients}</td>
    <td class="ok">{total_sens:.1f}%</td><td>--</td>
  </tr>
</table>

<footer>
  TUBITAK 2209-A Arastirma Projesi | Ahmet Demir | {datetime.now().strftime('%d.%m.%Y %H:%M')}
</footer>
</body>
</html>"""

out = Path('/Users/ahmetdemir/pneumothorax-ai-detection/validasyon_raporu.html')
out.write_text(html, encoding='utf-8')
print(f"\nRapor: {out}")
print(f"Sensitivite: {tp}/{tot} = {sens:.1f}%")
print(f"Toplam 85 hasta: {total_sens:.1f}%")
