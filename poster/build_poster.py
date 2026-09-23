r"""
build_poster.py — UNT Research Day poster, built on the COI 2x3 template.

Structure follows the example posters Dr. Aledhari supplied: a filled header
bar on every section, sub-headers written as complete thoughts rather than
labels, each figure sitting inside the section it belongs to, and body text as
one-sentence bullets.

The visual identity is UNT's only. Every colour below is read out of the COI
template itself -- UNT_GREEN is the template's own heading colour and the pale
background is the template slide's. Nothing is taken from the example posters.

Kept from the template: the 24x36 canvas, the green header bar, the UNT logo
and Calibri. The template's seven body text boxes are removed, because the new
section list does not map onto them.

Section heights are estimated (see `body`) rather than measured, since
PowerPoint resolves autofit at render time; the script prints where each
column ends so the layout can be balanced against BOTTOM.

    python poster/build_poster.py
"""
import os
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

TEMPLATE = os.path.expanduser(r"~/Downloads/COI Research Poster Template 2x3.pptx")
OUT = "poster/FL_Poster_Chevuri_Aledhari.pptx"
FIGDIR = "poster/figures"

# ── UNT palette, read out of the template ────────────────────────────────
UNT_GREEN = RGBColor(0x00, 0x74, 0x39)   # the template's own heading colour
UNT_SUB = RGBColor(0x3F, 0x8C, 0x5C)     # lighter tint of the same hue
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x1A, 0x1A)
FONT = "Calibri"                          # the template's minor font

# ── layout grid ──────────────────────────────────────────────────────────
COL_W = 7.26
COL_X = [0.60, 8.36, 16.12]
TOP = 3.62                                # header bar ends at 3.27
BOTTOM = 35.3

BAR_H, BAR_PT = 0.78, 25
SUB_H, SUB_PT = 0.62, 19
BODY_PT = 19
REF_PT = 16
LINE21 = 0.355                            # height of one 21 pt line, in inches
GAP = 0.20

TITLE = "Robustness or Weighting? Re-Evaluating Federated Aggregation on Heterogeneous Health Data"
AUTHORS = "Kunal Chevuri and Dr. Mohammed Aledhari"
AFFIL = "University of North Texas"

# ═════════════════════════════════════════════════════════════════════════
# Content. Every bullet is one sentence, as in the example posters.
# ═════════════════════════════════════════════════════════════════════════
MOTIVATION = [
    "Hospitals and health agencies hold records that could sharpen how we "
    "predict disease, but privacy law stops them pooling the data.",
    "Federated learning trains one shared model without moving any patient "
    "record out of the institution holding it.",
    "The rule that merges each site's work is normally chosen for its "
    "robustness to noisy or unusual sites.",
    "Robust rules are always scored against FedAvg, which weights each site "
    "by how much data it holds, so the biggest site gets the loudest vote.",
    "Robust rules discard that weighting entirely, so two things change at "
    "once and nobody had separated them.",
    "We ask: when a robust rule wins, is it the robust statistic winning, or "
    "just the end of the loudest vote?",
]

BACKGROUND = [
    "Each site trains on its own data and sends only the model update, which "
    "the server merges into one shared model.",
]

DATASET = [
    "CDC BRFSS 2023: 382,709 U.S. adults, predicting self-reported "
    "depression, which 20.7% of respondents report.",
    "UCI Breast Cancer Wisconsin runs alongside as a clean clinical benchmark.",
    "Each dataset is dealt to 20 simulated sites by Dirichlet partitioning, "
    "whose dial alpha sets how lopsided the split is.",
]
DATASET_AFTER = [
    "At alpha = 0.1 one site holds 106,481 training rows, 34.8% of the total, "
    "while two sites receive nothing at all.",
]

METHODS = [
    "Seven merge rules run head to head: FedAvg, FedProx, CS-Agg, Krum, "
    "trimmed mean, coordinate-wise median, and an unweighted mean.",
    "The unweighted mean is our control: it uses no robust statistic and "
    "differs from FedAvg only in ignoring how much data each site holds.",
    "Anything the control gains over FedAvg is therefore weighting alone, "
    "which is what isolates the effect.",
    "Labels are flipped at 20% of sites, across 0 to 30% of their records.",
    "On BRFSS we flip them only the way real people err, reporting no "
    "depression when they have it.",
    "624 conditions cover every combination of rule, dataset, split, noise "
    "level and random seed.",
    "Each runs 50 rounds, scored by AUC-ROC on a clean held-out test set the "
    "sites never see.",
    "Differences are tested with Wilcoxon signed-rank, Holm-corrected within "
    "each of three comparison families.",
]

RES_A_SUB = "Most of the gain over FedAvg is weighting, not robustness"
RES_A = [
    "The control alone recovers 79 to 98% of what the three strongest robust "
    "rules gain over FedAvg.",
    "Only coordinate-wise median clearly clears it, at +0.0163 AUC "
    "(Holm p < 0.0001).",
    "Trimmed mean's +0.0026 is significant but negligible, and CS-Agg cannot "
    "be told apart from the control (p = 0.22).",
]

RES_C_SUB = "Krum is unreliable, not merely weaker"
RES_C = [
    "Krum lands below the control and collapses to 0.328 AUC on one run, "
    "worse than a coin flip.",
]

RES_G_SUB = "Lopsided data hurts far more than broken labels"
RES_G = [
    "Moving from alpha = 1.0 to 0.1 costs FedAvg 0.2348 AUC, while 0% to 30% "
    "label noise costs it 0.0016.",
    "That 149x gap narrows to about 14x when sites track twelve health "
    "variables instead of four, but it never closes.",
]

RES_B_SUB = "A clean benchmark would have hidden all of this"
RES_B = [
    "On Breast Cancer every rule but Krum reaches 0.997 AUC or better, where "
    "no difference between them is visible.",
    "Messy, self-reported population health data is where these rules "
    "actually separate.",
]

FUTURE = [
    "Test adaptive Byzantine attackers, not only the label noise studied here.",
    "Replace the simulated Dirichlet split with data partitioned across real "
    "institutions.",
    "Check whether the weighting effect persists for deeper models and larger "
    "numbers of sites.",
]

CONCLUSIONS = [
    "Most of the advantage robust aggregators hold over FedAvg comes from how "
    "they weight sites, not from the robust statistic they are named for.",
    "Any robustness evaluation run on size-skewed data should include an "
    "unweighted-mean control.",
    "The control costs one line of code, and without it a reported gain "
    "cannot be credited to robustness.",
    "A rule's reputation is not evidence: Krum is widely cited and failed "
    "here, while coordinate-wise median is simple and held.",
]

ACK = [
    "Survey data collected and published by the Centers for Disease Control "
    "and Prevention (BRFSS 2023).",
    "Carried out in the Computational Healthcare and Biotechnology Lab, "
    "University of North Texas.",
]

CODE = [
    "All code, the merged results and a script that re-checks every number on "
    "this poster are openly available.",
    "github.com/kunalchevuri/federated-learning-health-robustness",
]

REFERENCES = [
    "[1]  McMahan et al. Communication-efficient learning of deep networks "
    "from decentralized data. AISTATS, 2017.",
    "[2]  Blanchard et al. Machine learning with adversaries: Byzantine "
    "tolerant gradient descent. NIPS, 2017.",
    "[3]  Yin et al. Byzantine-robust distributed learning: towards optimal "
    "statistical rates. ICML, 2018.",
    "[4]  Karimireddy et al. Byzantine-robust learning on heterogeneous "
    "datasets via bucketing. ICLR, 2022.",
    "[5]  Hsu et al. Measuring the effects of non-identical data distribution "
    "for federated visual classification. arXiv:1909.06335, 2019.",
    "[6]  CDC. BRFSS 2023 survey data and documentation.",
]

# ═════════════════════════════════════════════════════════════════════════
prs = Presentation(TEMPLATE)
slide = prs.slides[0]
shapes = {sh.name: sh for sh in slide.shapes}


def drop(name):
    sh = shapes.get(name)
    if sh is not None:
        sh._element.getparent().remove(sh._element)


def bar(col, y, text, height=BAR_H, size=BAR_PT, fill=UNT_GREEN):
    """A filled UNT-green section header with white text."""
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(COL_X[col]),
                                Inches(y), Inches(COL_W), Inches(height))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.16)
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    r.font.name, r.font.size, r.font.bold = FONT, Pt(size), True
    r.font.color.rgb = WHITE
    return y + height + 0.12


def body(col, y, items, size=BODY_PT, bullet=True, width=None):
    """Bulleted body text; returns the estimated y just below it."""
    w = width if width else COL_W
    # 21 pt wraps at ~52 chars in a 7.26 in column; scale with size and width.
    per_line = int(52 * (21.0 / size) * (w / COL_W))
    lines = sum(max(1, -(-len(t) // per_line)) for t in items)
    h = lines * (LINE21 * size / 21.0) + 0.135 * len(items) + 0.10

    tb = slide.shapes.add_textbox(Inches(COL_X[col]), Inches(y),
                                  Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, t in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before, p.space_after = Pt(0), Pt(8)
        if bullet:
            pPr = p._p.get_or_add_pPr()
            pPr.set("indent", "-215900")
            pPr.set("marL", "215900")
            pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
            pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))
        r = p.add_run()
        r.text = t
        r.font.name, r.font.size = FONT, Pt(size)
        r.font.color.rgb = INK
    return y + h + GAP


def figure(col, y, fn, height, width=None, left=None):
    w = width if width else COL_W
    x = left if left is not None else COL_X[col]
    slide.shapes.add_picture(os.path.join(FIGDIR, fn), Inches(x), Inches(y),
                             Inches(w), Inches(height))
    return y + height + GAP


# ── header ───────────────────────────────────────────────────────────────
hdr = shapes["TextBox 4"]
hdr.width = Inches(18.4)
for para, txt, size in zip(hdr.text_frame.paragraphs, [TITLE, AUTHORS, AFFIL], [44, 30, 30]):
    runs = para.runs
    runs[0].text = txt
    runs[0].font.size = Pt(size)
    for r in runs[1:]:
        r._r.getparent().remove(r._r)

for n in ("TextBox 7", "TextBox 10", "TextBox 12", "TextBox 14",
          "TextBox 17", "TextBox 21", "TextBox 23", "Picture 5"):
    drop(n)

# ── column 1: the setup ──────────────────────────────────────────────────
y = TOP
y = bar(0, y, "Motivation & Questions")
y = body(0, y, MOTIVATION)
y = bar(0, y, "Background: how federated learning works")
y = body(0, y, BACKGROUND)
y = figure(0, y, "posterD_federated.png", 2.85)
y = bar(0, y, "Dataset")
y = body(0, y, DATASET)
y = figure(0, y, "posterF_partition.png", 4.35)
y = body(0, y, DATASET_AFTER)
y = bar(0, y, "Code & Data")
qr_top = y
y = body(0, y, CODE, bullet=False, width=COL_W - 1.80)
figure(0, qr_top + 0.02, "posterE_qr.png", 1.55, width=1.55,
       left=COL_X[0] + COL_W - 1.60)
y = bar(0, max(y, qr_top + 1.75), "References")
y = body(0, y, REFERENCES, size=REF_PT, bullet=False)
print(f"  column 1 ends at {y - GAP:5.2f} in   (limit {BOTTOM})")

# ── column 2: what we did, and the headline result ───────────────────────
y = TOP
y = bar(1, y, "Methods")
y = body(1, y, METHODS)
y = bar(1, y, "Results")
y = bar(1, y, RES_A_SUB, height=SUB_H, size=SUB_PT, fill=UNT_SUB)
y = figure(1, y, "posterA_decomposition.png", 7.40)
y = body(1, y, RES_A)
y = bar(1, y, RES_C_SUB, height=SUB_H, size=SUB_PT, fill=UNT_SUB)
y = figure(1, y, "posterC_runlevel.png", 6.20)
y = body(1, y, RES_C)
print(f"  column 2 ends at {y - GAP:5.2f} in   (limit {BOTTOM})")

# ── column 3: the rest of the results, then what it means ────────────────
y = TOP
y = bar(2, y, "Results, continued")
y = bar(2, y, RES_G_SUB, height=SUB_H, size=SUB_PT, fill=UNT_SUB)
y = figure(2, y, "posterG_stressors.png", 4.40)
y = body(2, y, RES_G)
y = bar(2, y, RES_B_SUB, height=SUB_H, size=SUB_PT, fill=UNT_SUB)
y = figure(2, y, "posterB_both_datasets.png", 7.20)
y = body(2, y, RES_B)
y = bar(2, y, "Future Directions")
y = body(2, y, FUTURE)
y = bar(2, y, "Conclusions")
y = body(2, y, CONCLUSIONS)
y = bar(2, y, "Acknowledgements")
y = body(2, y, ACK)
print(f"  column 3 ends at {y - GAP:5.2f} in   (limit {BOTTOM})")

prs.save(OUT)
print("\nwrote", OUT)
