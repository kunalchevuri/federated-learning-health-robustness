r"""
build_poster.py — UNT Research Day poster, built on the COI 2x3 template.

Structure follows the example posters Dr. Aledhari supplied: a filled header
bar on every section, sub-headers written as complete thoughts rather than
labels, each figure sitting inside the section it belongs to, and body text as
one-sentence bullets.

Two tables carry the numbers a judge would otherwise have to dig out of the
bullets: a results summary under the headline result, and the study design
under Methods.

The visual identity is UNT's only. UNT_GREEN and the pale board are read out
of the COI template itself; UNT_SUB and UNT_BAND are tints of that same hue,
mixed here rather than lifted from any file. Nothing comes from the example
posters.

Kept from the template: the 24x36 canvas, the green header bar, the UNT logo
and Calibri. The template's seven body text boxes are removed, because the new
section list does not map onto them.

The logo is the College of Information lockup, which no longer matches the
college named in the header, so it is cropped back to the primary UNT mark --
see LOGO_CROP.

Section heights are estimated (see `body`) rather than measured, since
PowerPoint resolves autofit at render time; the script prints where each
column ends so the layout can be balanced against BOTTOM.

    python poster/figures_poster.py     # charts
    python poster/render_diagram.py     # the schematic
    python poster/build_poster.py       # this file
"""
import copy
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

# ── UNT palette ──────────────────────────────────────────────────────────
UNT_GREEN = RGBColor(0x00, 0x74, 0x39)   # the template's own heading colour
UNT_SUB = RGBColor(0x3F, 0x8C, 0x5C)     # lighter tint of the same hue
UNT_BAND = RGBColor(0xCF, 0xE3, 0xC4)    # pale tint, for the control row
PAPER = RGBColor(0xE2, 0xF0, 0xD9)       # the template slide's own fill
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x1A, 0x1A)
CAP_INK = RGBColor(0x33, 0x33, 0x33)     # captions sit back from the body text
FONT = "Calibri"                          # the template's minor font

# ── layout grid ──────────────────────────────────────────────────────────
COL_W = 7.26
COL_X = [0.60, 8.36, 16.12]
TOP = 3.62                                # header bar ends at 3.27
BOTTOM = 34.2                             # the footer bar starts at 34.40

BAR_H, BAR_PT = 0.78, 25
SUB_H, SUB_PT = 0.62, 19
BODY_PT = 19
REF_PT = 17
TBL_PT = 15
CAP_PT = 15                               # figure captions
LINE21 = 0.355                            # height of one 21 pt line, in inches
GAP = 0.20

TITLE = "Robustness or Weighting? Re-Evaluating Federated Aggregation on Heterogeneous Health Data"
AUTHORS = "Kunal Chevuri and Dr. Mohammed Aledhari"
# Both names read exactly as Dr. Aledhari specified them. They share two lines
# rather than three because a fifth header line drops below the green bar.
AFFIL = ["Data Science Department",
         "College of Artificial Intelligence and Advanced Analytics"
         "  ·  University of North Texas"]

# The supplied logo is the College of Information lockup. Cropping its bottom
# off leaves the primary UNT mark, which is what the header should carry now
# that the college has a different name. Measured off the image itself: the
# wordmark ends at row 492 of 901, the college text runs 500-583.
LOGO_CROP = 0.45394
LOGO_BOX = (19.30, 0.76, 3.98, 1.76)      # left, top, width, height

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
    "Robust rules are scored against FedAvg, which weights each site by how "
    "much data it holds -- and they discard that weighting entirely, so two "
    "things change at once and nobody had separated them.",
    "We ask: when a robust rule wins, is it the robust statistic winning, or "
    "just the end of the loudest vote?",
]

BACKGROUND = [
    "Each site trains on its own data and sends only the model update -- no "
    "patient record ever leaves the institution holding it.",
]

DATASET = [
    "CDC BRFSS 2023: 382,709 U.S. adults, predicting self-reported "
    "depression, which 20.7% of respondents report.",
    "UCI Breast Cancer Wisconsin runs alongside as a clean clinical benchmark.",
    "Each dataset is dealt to 20 simulated sites by Dirichlet partitioning, "
    "whose dial alpha sets how lopsided the split is.",
]
DATASET_AFTER = []

METHODS = [
    "Seven merge rules run head to head: FedAvg, FedProx, CS-Agg, Krum, "
    "trimmed mean, coordinate-wise median, and an unweighted mean.",
    "The unweighted mean is our control: it uses no robust statistic and "
    "differs from FedAvg only in ignoring how much data each site holds.",
    "Anything the control gains over FedAvg is therefore weighting alone, "
    "which is what isolates the effect.",
    "On BRFSS we corrupt labels only the way real people err, reporting no "
    "depression when they have it.",
]

# ── Table 2: the study design ────────────────────────────────────────────
DESIGN_TABLE = [
    ["What we varied", "Levels tested"],
    ["Merge rules", "7, including the unweighted-mean control"],
    ["Datasets", "BRFSS 2023 and Breast Cancer Wisconsin"],
    ["Split skew, alpha", "0.1, 0.5, 1.0 — lower is more lopsided"],
    ["Label noise", "0, 10, 20, 30% of records, at 20% of sites"],
    ["Every run", "20 sites, 50 rounds, seeds 42 / 123 / 456"],
    ["Total", "624 completed runs"],
]
DESIGN_W = [2.20, 5.06]

RES_A_SUB = "Most of the gain over FedAvg is weighting, not robustness"
RES_A = [
    "Pooled across every condition the control recovers 78 to 96% of what the "
    "three strongest robust rules gain, and CS-Agg cannot be told apart from it.",
]

# ── Table 1: the results summary ─────────────────────────────────────────
# Every figure below reproduces from results/experiment_results_merged.csv;
# the deltas and Holm-corrected p-values use the same Wilcoxon signed-rank
# procedure as verify_paper_numbers.py.
RESULTS_TABLE = [
    ["Merge rule", "Mean\nAUC", "Worst\nrun", "vs.\nFedAvg", "vs.\ncontrol", "Holm p\nvs. control"],
    ["FedAvg", "0.7081", "0.4002", "—", "-0.0584", "0.0036"],
    ["FedProx", "0.7286", "0.4874", "+0.0205", "-0.0379", "0.0036"],
    ["Krum", "0.7439", "0.3275", "+0.0358", "-0.0226", "0.22  n.s."],
    ["Unweighted mean", "0.7665", "0.6405", "+0.0584", "—", "control"],
    ["Trimmed mean", "0.7691", "0.6143", "+0.0610", "+0.0026", "0.0036"],
    ["CS-Agg", "0.7746", "0.7025", "+0.0665", "+0.0081", "0.22  n.s."],
    ["Coord. median", "0.7828", "0.7279", "+0.0747", "+0.0163", "< 0.0001"],
]
RESULTS_W = [2.00, 0.90, 0.90, 1.05, 1.05, 1.36]
RESULTS_CAPTION = ("36 runs per rule on BRFSS, every condition pooled. Holm-corrected "
                   "Wilcoxon signed-rank against the control.")

RES_C_SUB = "Krum is unreliable, not merely weaker"
RES_C = []

RES_G_SUB = "Lopsided data hurts far more than broken labels"
RES_G = [
    "That 149x gap narrows to about 14x when sites track twelve health "
    "variables instead of four, but it never closes.",
]

RES_B_SUB = "A clean benchmark would have hidden all of this"
RES_B = [
    "Messy, self-reported population health data is where these rules "
    "actually separate.",
]

FUTURE = [
    "Test adaptive Byzantine attackers, not only the label noise studied here.",
    "Replace the simulated split with data partitioned across real "
    "institutions, and check that the weighting effect holds for deeper "
    "models and more sites.",
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
# Figure captions. Numbered in the poster's reading order. Each one says what
# the figure shows and then what it means for the result, so a judge can read
# a figure without reading the bullets around it. Every number quoted here
# re-derives from results/experiment_results_merged.csv.
# ═════════════════════════════════════════════════════════════════════════
CAPTIONS = {
    1: "Figure 1. One training round, run two ways. The same 20 sites send the "
       "same updates; only the merge rule differs. FedAvg hands the largest "
       "site 34.8% of the vote, the control gives every site 5% -- worth "
       "+0.172 AUC on its own.",
    2: "Figure 2. How the training data is dealt across 20 sites. At "
       "alpha = 0.1 one site holds 34.8% of it and two sites get none; at "
       "alpha = 1.0 the split is mild. This skew is what gives FedAvg's "
       "size-weighting something to get wrong.",
    3: "Figure 3. All seven rules under the harshest split, 12 runs each. "
       "Three robust rules clear the dashed control line, but only by 0.004 "
       "to 0.045, against the 0.172 the control alone gains over FedAvg.",
    4: "Figure 4. The individual runs behind those means. Krum's average of "
       "0.658 hides a run at 0.328, worse than a coin flip, while the control "
       "never falls below 0.64. An average is not evidence a rule is safe.",
    5: "Figure 5. Both stressors, measured on FedAvg. Lopsided data costs 149 "
       "times more AUC than broken labels do, so robustness work aimed at "
       "noisy labels is aimed at the smaller problem.",
    6: "Figure 6. Both datasets, averaged over every split, noise level and "
       "seed (252 runs each). On the clean benchmark every rule but Krum "
       "reaches 0.997, so a study run only there would have found nothing.",
}

# ═════════════════════════════════════════════════════════════════════════
prs = Presentation(TEMPLATE)
slide = prs.slides[0]
shapes = {sh.name: sh for sh in slide.shapes}


# Layout runs twice per column: once with DRY set, to measure, and once for
# real with the section gap solved so the column lands just above the footer.
DRY = False
G = GAP


def drop(name):
    sh = shapes.get(name)
    if sh is not None:
        sh._element.getparent().remove(sh._element)


def bar(col, y, text, height=BAR_H, size=BAR_PT, fill=UNT_GREEN):
    """A filled UNT-green section header with white text."""
    if DRY:
        return y + height + 0.12
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


def body(col, y, items, size=BODY_PT, bullet=True, width=None, italic=False,
         colour=INK):
    """Bulleted body text; returns the estimated y just below it."""
    if not items:
        return y
    w = width if width else COL_W
    # 21 pt wraps at ~52 chars in a 7.26 in column; scale with size and width.
    per_line = int(52 * (21.0 / size) * (w / COL_W))
    lines = sum(max(1, -(-len(t) // per_line)) for t in items)
    h = lines * (LINE21 * size / 21.0) + 0.135 * len(items) + 0.10
    if DRY:
        return y + h + G

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
        r.font.name, r.font.size, r.font.italic = FONT, Pt(size), italic
        r.font.color.rgb = colour
    return y + h + G


NO_STYLE = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"   # "No Style, No Grid"


def _unstyle(tbl):
    """Strip the inherited blue table style so our own fills are what shows."""
    tbl.first_row = False
    tbl.horz_banding = False
    tblPr = tbl._tbl.find(qn("a:tblPr"))
    for el in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(el)
    sid = tblPr.makeelement(qn("a:tableStyleId"), {})
    sid.text = NO_STYLE
    tblPr.append(sid)                       # schema puts tableStyleId last


def table(col, y, rows, widths, size=TBL_PT, row_h=0.42, hdr_h=0.66,
          highlight=None, centre_from=1):
    """A UNT-green-headed table with zebra rows; returns the y below it.

    `highlight` is the body row index to mark as the control; `centre_from`
    is the first column to centre (earlier columns stay left-aligned).
    """
    n, m = len(rows), len(rows[0])
    h = hdr_h + (n - 1) * row_h
    if DRY:
        return y + h + G
    gf = slide.shapes.add_table(n, m, Inches(COL_X[col]), Inches(y),
                                Inches(sum(widths)), Inches(h))
    tbl = gf.table
    _unstyle(tbl)
    for j, cw in enumerate(widths):
        tbl.columns[j].width = Inches(cw)
    tbl.rows[0].height = Inches(hdr_h)
    for i in range(1, n):
        tbl.rows[i].height = Inches(row_h)

    for i, row in enumerate(rows):
        marked = highlight is not None and i == highlight
        for j, txt in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.10)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = UNT_GREEN
            elif marked:
                c.fill.fore_color.rgb = UNT_BAND
            else:
                c.fill.fore_color.rgb = WHITE if i % 2 else PAPER
            tf = c.text_frame
            tf.word_wrap = True
            align = PP_ALIGN.CENTER if j >= centre_from else PP_ALIGN.LEFT
            for k, line in enumerate(txt.split("\n")):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                p.alignment = align
                r = p.add_run()
                r.text = line
                r.font.name = FONT
                r.font.size = Pt(size)
                r.font.bold = (i == 0) or marked
                r.font.color.rgb = WHITE if i == 0 else INK
    return y + h + G


def figure(col, y, fn, height, width=None, left=None, num=None):
    """Place a figure and, when it carries a number, its caption below it."""
    w = width if width else COL_W
    x = left if left is not None else COL_X[col]
    if not DRY:
        slide.shapes.add_picture(os.path.join(FIGDIR, fn), Inches(x), Inches(y),
                                 Inches(w), Inches(height))
    y = y + height + (0.10 if num else G)
    if num is None:
        return y
    return body(col, y, [CAPTIONS[num]], size=CAP_PT, bullet=False,
                colour=CAP_INK)


# ── header ───────────────────────────────────────────────────────────────
hdr = shapes["TextBox 4"]
hdr.width = Inches(18.4)
hdr_paras = hdr.text_frame.paragraphs
lines = [(TITLE, 42), (AUTHORS, 28)] + [(a, 24) for a in AFFIL]

for i, (txt, size) in enumerate(lines):
    if i < len(hdr_paras):
        para = hdr_paras[i]
        runs = para.runs
        runs[0].text = txt
        runs[0].font.size = Pt(size)
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        # clone the last styled paragraph so the new lines inherit its run
        # properties instead of falling back to the theme default
        src = hdr.text_frame.paragraphs[-1]._p
        new = copy.deepcopy(src)
        src.getparent().append(new)
        para = hdr.text_frame.paragraphs[-1]
        para.runs[0].text = txt
        para.runs[0].font.size = Pt(size)
        for r in para.runs[1:]:
            r._r.getparent().remove(r._r)

# crop the College of Information lockup back to the primary UNT mark
logo = shapes["Picture 2"]
logo.crop_bottom = LOGO_CROP
logo.left, logo.top = Inches(LOGO_BOX[0]), Inches(LOGO_BOX[1])
logo.width, logo.height = Inches(LOGO_BOX[2]), Inches(LOGO_BOX[3])

for n in ("TextBox 7", "TextBox 10", "TextBox 12", "TextBox 14",
          "TextBox 17", "TextBox 21", "TextBox 23", "Picture 5"):
    drop(n)

# The UNT .docx version of this template closes with a green footer bar; the
# .pptx version does not. Clone the header rectangle so the footer carries the
# template's own gradient rather than a colour we invented.
_hdr_rect = shapes["Rectangle 15"]
_hdr_rect._element.getparent().append(copy.deepcopy(_hdr_rect._element))
footer = list(slide.shapes)[-1]
footer.left, footer.top = 0, Inches(34.40)
footer.width, footer.height = Inches(24.0), Inches(1.60)

# ═════════════════════════════════════════════════════════════════════════
# Columns. Each is a function of the section gap so it can be measured with
# DRY set and then re-run with a gap that lands the column just above the
# footer -- otherwise the three columns stop wherever their content happens
# to end and leave a dead band across the bottom of the board.
# ═════════════════════════════════════════════════════════════════════════
def column_1():
    y = TOP
    y = bar(0, y, "Motivation & Questions")
    y = body(0, y, MOTIVATION)
    y = bar(0, y, "Background: how federated learning works")
    y = body(0, y, BACKGROUND)
    y = figure(0, y, "posterD_federated.png", 3.95, num=1)
    y = bar(0, y, "Dataset: two health datasets, split 20 ways")
    y = body(0, y, DATASET)
    y = figure(0, y, "posterF_partition.png", 5.00, num=2)
    y = body(0, y, DATASET_AFTER)
    # References sit bottom-left, where the Bylinskii example puts its paper
    # and dataset box. Chien carries no reference section at all; keeping one
    # is the single deliberate departure from the examples.
    y = bar(0, y, "References")
    return body(0, y, REFERENCES, size=REF_PT, bullet=False)


def column_2():
    y = TOP
    y = bar(1, y, "Methods: seven merge rules and one control")
    y = body(1, y, METHODS)
    y = table(1, y, DESIGN_TABLE, DESIGN_W, centre_from=9)
    y = bar(1, y, "Results")
    y = bar(1, y, RES_A_SUB, height=SUB_H, size=SUB_PT, fill=UNT_SUB)
    y = figure(1, y, "posterA_decomposition.png", 5.30, num=3)
    y = table(1, y, RESULTS_TABLE, RESULTS_W, highlight=4)
    y = body(1, y, [RESULTS_CAPTION], size=REF_PT, bullet=False, italic=True,
             colour=RGBColor(0x44, 0x44, 0x44))
    y = body(1, y, RES_A)
    y = bar(1, y, RES_C_SUB, height=SUB_H, size=SUB_PT, fill=UNT_SUB)
    y = figure(1, y, "posterC_runlevel.png", 5.40, num=4)
    return body(1, y, RES_C)


def column_3():
    y = TOP
    y = bar(2, y, "Results, continued")
    y = bar(2, y, RES_G_SUB, height=SUB_H, size=SUB_PT, fill=UNT_SUB)
    y = figure(2, y, "posterG_stressors.png", 4.30, num=5)
    y = body(2, y, RES_G)
    y = bar(2, y, RES_B_SUB, height=SUB_H, size=SUB_PT, fill=UNT_SUB)
    y = figure(2, y, "posterB_both_datasets.png", 5.10, num=6)
    y = body(2, y, RES_B)
    y = bar(2, y, "Future Directions")
    y = body(2, y, FUTURE)
    y = bar(2, y, "Conclusions")
    y = body(2, y, CONCLUSIONS)
    # the code + QR box sits bottom-right, where both examples put theirs
    y = bar(2, y, "Code & Data")
    qr_top = y
    y = body(2, y, CODE, bullet=False, width=COL_W - 1.80)
    figure(2, qr_top + 0.02, "posterE_qr.png", 1.55, width=1.55,
           left=COL_X[2] + COL_W - 1.60)
    y = max(y, qr_top + 1.75)
    y = bar(2, y, "Acknowledgements")
    return body(2, y, ACK)


TARGET = BOTTOM - 0.40        # leave a little air above the footer bar
G_MIN, G_MAX = 0.20, 0.60     # below 0.20 sections crowd; above 0.60 they drift

for n, draw in enumerate([column_1, column_2, column_3], start=1):
    # the column end is affine in G, so two dry runs give the slope exactly
    DRY = True
    G = 0.20
    e1 = draw()
    G = 0.40
    e2 = draw()
    slope = (e2 - e1) / 0.20
    base = e1 - slope * 0.20
    G = min(G_MAX, max(G_MIN, (TARGET - base) / slope))
    DRY = False
    end = draw()
    print(f"  column {n}: gap {G:4.2f} in, ends at {end - G:5.2f} in "
          f"(target {TARGET}, footer {BOTTOM + 0.2})")

prs.save(OUT)
print("\nwrote", OUT)
