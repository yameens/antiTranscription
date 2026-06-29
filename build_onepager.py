"""
build_onepager.py
assemble the self-contained HTML one-pager for 'final accuracy/'.
embeds the chart PNGs as base64 data-URIs so the file is fully portable
(no external assets -> works offline and under the Artifact CSP).

writes:
  final accuracy/index.html          standalone document (double-clickable)
  final accuracy/_artifact_body.html content-only (for the Artifact tool)
"""

import base64
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "final accuracy")


def data_uri(path):
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


PITCH = data_uri(os.path.join(OUT, "pitch_comparison.png"))
ABLATION = data_uri(os.path.join(OUT, "domain_gap_ablation.png"))
CAMERA = data_uri(os.path.join(OUT, "camera_augmentation.png"))

CSS = """
:root{
  --paper:#fbfcfd; --ink:#16202b; --muted:#5c6b7a; --hair:#e2e8ee;
  --old:#c0392b; --geo:#c98a0e; --new:#1f8b4c; --new-soft:#eef6f1;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
  --mono:"SF Mono","JetBrains Mono",Menlo,Consolas,ui-monospace,monospace;
}
*{box-sizing:border-box;}
.page{
  max-width:880px;margin:0 auto;padding:46px 40px 30px;
  background:var(--paper);color:var(--ink);
  font-family:var(--sans);font-size:14px;line-height:1.55;
}
/* ---- header with faint staff-line motif ---- */
.head{position:relative;padding-bottom:22px;margin-bottom:26px;
  border-bottom:1px solid var(--hair);}
.staff{position:absolute;inset:6px 0 auto 0;height:46px;z-index:0;opacity:.5;
  background:repeating-linear-gradient(to bottom,
    transparent 0 9px,var(--hair) 9px 10px);}
.head>*{position:relative;z-index:1;}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--new);margin:0 0 8px;}
h1{font-family:var(--serif);font-weight:600;font-size:34px;line-height:1.08;
  margin:0 0 8px;text-wrap:balance;letter-spacing:-.01em;}
.sub{color:var(--muted);font-size:15px;margin:0;max-width:62ch;}
.byline{font-family:var(--mono);font-size:11px;color:var(--muted);
  margin-top:12px;letter-spacing:.03em;}
/* ---- stat strip ---- */
.strip{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;
  gap:6px;margin:0 0 30px;}
.stat{text-align:center;padding:14px 8px;border-radius:12px;}
.stat .n{font-family:var(--serif);font-weight:600;font-size:30px;line-height:1;
  font-variant-numeric:tabular-nums;}
.stat .l{font-family:var(--mono);font-size:10.5px;letter-spacing:.04em;
  color:var(--muted);margin-top:8px;text-transform:uppercase;line-height:1.4;}
.stat.old{background:#fbeeec;} .stat.old .n{color:var(--old);}
.stat.geo{background:#fbf3e1;} .stat.geo .n{color:var(--geo);}
.stat.new{background:var(--new-soft);} .stat.new .n{color:var(--new);}
.arrow{font-size:20px;color:#b6c2cd;}
.strip-cap{font-family:var(--mono);font-size:10.5px;color:var(--muted);
  text-align:center;margin:-22px 0 30px;letter-spacing:.03em;}
/* ---- section headers ---- */
h2{font-family:var(--serif);font-weight:600;font-size:19px;margin:0 0 10px;
  display:flex;align-items:baseline;gap:10px;}
h2 .tag{font-family:var(--mono);font-size:10px;letter-spacing:.14em;
  text-transform:uppercase;color:#aab6c1;font-weight:400;}
/* ---- two-column methods ---- */
.cols{display:grid;grid-template-columns:1fr 1fr;gap:26px;margin-bottom:30px;}
.col p{margin:0;color:#2b3947;}
.col .kicker{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;margin:0 0 8px;}
.col.paper .kicker{color:var(--muted);}
.col.up .kicker{color:var(--new);}
.col.up{background:var(--new-soft);border-radius:12px;padding:18px 20px;
  margin:-18px -2px;}
/* ---- figures ---- */
figure{margin:0 0 26px;}
figure img{width:100%;height:auto;display:block;border:1px solid var(--hair);
  border-radius:10px;background:#fff;}
figcaption{font-family:var(--mono);font-size:11px;color:var(--muted);
  margin-top:8px;line-height:1.5;}
.figrow{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-bottom:28px;}
.figrow figure{margin:0;}
/* ---- tables ---- */
table{width:100%;border-collapse:collapse;font-size:13px;margin:0 0 8px;}
caption{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);text-align:left;
  padding-bottom:8px;}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--hair);
  font-variant-numeric:tabular-nums;}
th:first-child,td:first-child{text-align:left;}
thead th{font-family:var(--mono);font-size:10.5px;letter-spacing:.04em;
  text-transform:uppercase;color:var(--muted);font-weight:600;
  border-bottom:1.5px solid #cdd7df;}
tbody tr:last-child td{border-bottom:1.5px solid #cdd7df;}
.win{color:var(--new);font-weight:700;}
.lose{color:var(--old);}
.tables{display:grid;grid-template-columns:1.25fr 1fr;gap:30px;margin-bottom:28px;}
/* ---- takeaway ---- */
.takeaway{background:var(--ink);color:#eaf1f4;border-radius:14px;
  padding:22px 26px;margin-bottom:22px;}
.takeaway .kicker{font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;
  text-transform:uppercase;color:#7fd0a3;margin:0 0 8px;}
.takeaway p{margin:0;font-family:var(--serif);font-size:18px;line-height:1.45;}
.takeaway b{color:#8be0ad;font-weight:600;}
/* ---- footer ---- */
.foot{display:flex;justify-content:space-between;align-items:baseline;
  font-family:var(--mono);font-size:11px;color:var(--muted);
  border-top:1px solid var(--hair);padding-top:14px;flex-wrap:wrap;gap:8px;}
.note{font-size:12px;color:var(--muted);font-style:italic;margin:0 0 26px;}
@media (max-width:680px){
  .cols,.figrow,.tables{grid-template-columns:1fr;}
  .strip{grid-template-columns:1fr;} .arrow{display:none;}
  .strip-cap{margin-top:8px;}
}
@media print{
  .page{padding:0;} body{background:#fff;}
  figure img{border-color:#ddd;}
}
"""

BODY = f"""
<div class="page">
  <header class="head">
    <div class="staff"></div>
    <p class="eyebrow">antiTranscription · accuracy upgrade</p>
    <h1>From a Phone Photo of Sheet Music to MIDI</h1>
    <p class="sub">An end-to-end CRNN+CTC reader replaces the original
      segment-and-classify pipeline — lifting exact pitch on real phone photos
      from <b>8–14%</b> to <b>90–100%</b>, adding rhythm, and closing the
      clean-to-real domain gap.</p>
    <p class="byline">CS 131 Computer Vision · Yameen Sekandari · final results brief</p>
  </header>

  <section class="strip">
    <div class="stat old"><div class="n">8–14%</div>
      <div class="l">old CNN<br>segment + classify</div></div>
    <div class="arrow">&rarr;</div>
    <div class="stat geo"><div class="n">60–86%</div>
      <div class="l">geometry-only<br>notehead template</div></div>
    <div class="arrow">&rarr;</div>
    <div class="stat new"><div class="n">90–100%</div>
      <div class="l">CRNN + CTC<br>this work</div></div>
  </section>
  <p class="strip-cap">exact pitch accuracy, measured on four real phone photos</p>

  <section class="cols">
    <div class="col paper">
      <p class="kicker">The original method</p>
      <p>A five-stage pipeline: rectify the page, detect staves with the Hough
        transform, remove the staff lines and segment symbols by connected
        components, classify each crop's duration with a small CNN, and read
        pitch from notehead geometry. Its central finding was a
        <b>clean-to-real domain gap</b> — the CNN scored 58.9% on clean glyphs
        but only 22.7% on real photo crops, because segmentation merges destroy
        notes and clean-trained features don't survive blur, shadow and ink-bleed.</p>
    </div>
    <div class="col up">
      <p class="kicker">What we upgraded</p>
      <p>We removed segmentation entirely and dropped in an <b>end-to-end
        CRNN+CTC</b> reader (CNN&nbsp;&rarr;&nbsp;BiLSTM&nbsp;&rarr;&nbsp;CTC) that
        maps a whole staff image straight to a token sequence — recovering
        <b>pitch and duration jointly</b>. We trained it on 15,354 PrIMuS treble
        staves with <b>synthetic phone-photo augmentation</b> (blur, uneven light,
        JPEG, ink-bleed), on the laptop's Apple GPU.</p>
    </div>
  </section>

  <div class="figrow">
    <figure>
      <img src="{PITCH}" alt="three-way pitch and duration comparison">
      <figcaption>The CRNN beats both the old CNN and the geometry reader on
        every piece, and uniquely reads note durations.</figcaption>
    </figure>
    <figure>
      <img src="{ABLATION}" alt="domain gap ablation">
      <figcaption>Camera augmentation shrinks the clean-to-real Symbol Error Rate
        gap from +0.721 to +0.004 — a 99.4% reduction.</figcaption>
    </figure>
  </div>

  <div class="tables">
    <table>
      <caption>Exact pitch on the four phone photos</caption>
      <thead><tr><th>Piece</th><th>Old&nbsp;CNN</th><th>Geometry</th>
        <th>CRNN</th><th>Dur.</th></tr></thead>
      <tbody>
        <tr><td>Yankee Doodle</td><td class="lose">14%</td><td>86%</td>
          <td class="win">100%</td><td>97%</td></tr>
        <tr><td>Twinkle Twinkle</td><td class="lose">10%</td><td>86%</td>
          <td class="win">90%</td><td>89%</td></tr>
        <tr><td>Mary Had a Little Lamb</td><td class="lose">8%</td><td>69%</td>
          <td class="win">100%</td><td>54%</td></tr>
        <tr><td>CS 131</td><td class="lose">8%</td><td>60%</td>
          <td class="win">90%</td><td>98%</td></tr>
      </tbody>
    </table>
    <table>
      <caption>Domain-gap ablation (SER&nbsp;&darr;)</caption>
      <thead><tr><th>Model</th><th>Clean</th><th>Phone</th></tr></thead>
      <tbody>
        <tr><td>Clean-trained</td><td>0.030</td><td class="lose">0.750</td></tr>
        <tr><td>Camera-aug.</td><td>0.024</td><td class="win">0.028</td></tr>
      </tbody>
    </table>
  </div>

  <div class="takeaway">
    <p class="kicker">The takeaway</p>
    <p>Reading the staff <b>end-to-end</b> — instead of cutting it into crops and
      labelling each — turns the learned model from the pipeline's weakest link
      into its strongest, recovering <b>pitch and rhythm together</b> and
      generalizing from clean training data to real phone photos.</p>
  </div>

  <p class="note">Honest limitation: Mary Had a Little Lamb's durations (54%) lag —
    pitch is perfect but some note lengths are misread. Training was stopped early
    (epoch 6) to spare the laptop; a longer run should close this.</p>

  <div class="foot">
    <span>model: CNN&rarr;BiLSTM&rarr;CTC · 3.1M params · trained on Apple-GPU (MPS)</span>
    <span>reproduce: python test_crnn.py · eval_ablation.py</span>
  </div>
</div>
"""

HTML = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    '<title>antiTranscription — Final Accuracy</title>'
    f'<style>{CSS}</style></head><body>{BODY}</body></html>'
)

with open(os.path.join(OUT, "index.html"), "w") as fh:
    fh.write(HTML)
with open(os.path.join(OUT, "_artifact_body.html"), "w") as fh:
    fh.write(f"<style>{CSS}</style>{BODY}")

print("wrote final accuracy/index.html and _artifact_body.html")
print(f"index.html size: {os.path.getsize(os.path.join(OUT,'index.html'))//1024} KB")
