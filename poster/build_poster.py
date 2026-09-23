r"""
build_poster.py — fill the UNT COI 2x3 template with the FL-BRFSS content.

Edits the real template so UNT branding, the logo, the green bars and the
24x36 in canvas are preserved exactly. Body paragraphs are rewritten in place
(run 0's text is replaced, its siblings dropped) so each paragraph keeps the
template's own character formatting rather than collapsing to an unstyled run.

A body line given as a ("Bold lead.", " rest of sentence") tuple is split into
two runs so each block opens with a scannable bold phrase. Poster readers skim
first and commit second; the bold lead is what they skim.

PowerPoint does not reflow one text box around another, so section lengths and
the y positions below are matched by hand: roughly 0.40 in per wrapped line at
the template's 24 pt, 45 characters to a line in a 6.6 in column, plus BODY_GAP
between paragraphs and about 0.55 in for the heading.

    python poster/build_poster.py
"""
import copy
import os
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.text.text import _Run
from pptx.util import Inches, Pt

TEMPLATE = os.path.expanduser(r"~/Downloads/COI Research Poster Template 2x3.pptx")
OUT = "poster/FL_Poster_Chevuri_Aledhari.pptx"
FIGDIR = "poster/figures"

BODY_GAP = Pt(14)   # the template's own paragraph gap is ~0.6 in, far too airy

TITLE = "Robustness or Weighting? Re-Evaluating Federated Aggregation on Heterogeneous Health Data"
AUTHORS = "Kunal Chevuri and Dr. Mohammed Aledhari"
AFFIL = "University of North Texas"

# One line, directly under the header, for someone deciding whether to stop.
# The title is the paper's and carries three hard words; this is the plain
# version of the same sentence.
HOOK = ("Hospitals cannot share patient data. Federated learning gets around "
        "that. We found the field is crediting the wrong thing.")

INTRODUCTION = [
    "Hospitals, clinics and health agencies each hold records that could "
    "sharpen how we predict disease. Privacy law keeps them from pooling "
    "those records in one place.",
    "Federated learning is the workaround, sketched below. The merge step is "
    "the real design decision, because real sites are lopsided and "
    "self-reported health data arrives with errors.",
]

PURPOSE = [
    "Robust merge rules are always scored against FedAvg, which weights each "
    "site by how many records it holds. In effect the biggest site gets the "
    "loudest vote. Robust rules drop that weighting entirely, so two things "
    "change at once.",
    ("The question.", "  When a robust rule wins, is it the robust statistic "
     "winning, or just the end of the loudest vote?"),
]

METHODS = [
    ("Seven merge rules, head to head.", "  FedAvg, FedProx, CS-Agg, Krum, "
     "trimmed mean, coordinate-wise median, and an unweighted mean we added "
     "ourselves as a control."),
    ("The control is the experiment.", "  It uses no robust statistic at all, "
     "and differs from FedAvg in one respect: it ignores how much data each "
     "site holds. Anything it gains is weighting."),
    ("The data.", "  CDC's 2023 BRFSS survey, 382,709 U.S. adults, predicting "
     "self-reported depression, with Breast Cancer Wisconsin alongside as a "
     "clean clinical benchmark. The shared model is a small neural network."),
    ("Splitting it up.", "  Each dataset is dealt out to 20 simulated sites, "
     "deliberately unevenly. The dial, alpha, sets how lopsided: at alpha = 0.1 "
     "one site holds over a third of the data and others hold none."),
    ("Breaking the labels.", "  We flip labels at 20 percent of sites, across "
     "0 to 30 percent of their records, only the way real people get it wrong: "
     "reporting no depression when they have it."),
    ("The scale of it.", "  624 conditions covering every rule, dataset, "
     "split, noise level and seed. 50 rounds each, scored on a clean held-out "
     "test set. Holm-corrected Wilcoxon tests throughout."),
]

RESULTS = [
    "Every rule below is scored against the unweighted-mean control, not "
    "against FedAvg alone. Figures show BRFSS at the harshest split "
    "(alpha = 0.1) unless the panel says otherwise.",
    ("Reading the charts.", "  AUC-ROC measures how well the model tells "
     "apart people who have depression from people who do not. 0.5 is a coin "
     "flip, 1.0 is perfect."),
]

KEY_FINDINGS = [
    ("1.  Most of the benefit is bookkeeping, not robustness.",
     "  Simply declining to over-weight the biggest site recovers 79 to 98 "
     "percent of everything the three strongest robust rules gain over FedAvg."),
    ("2.  Only one rule earns its keep.",
     "  Coordinate-wise median clears the control by +0.0163 AUC "
     "(p < 0.0001). Trimmed mean's +0.0026 is real but too small to matter, "
     "and CS-Agg cannot be told apart from the control at all (p = 0.22)."),
    ("3.  A popular rule does worse than doing nothing.",
     "  Krum lands below the control, and on one run collapses to 0.328 AUC, "
     "worse than a coin flip, from a rule chosen for its robustness."),
    ("4.  Lopsided data hurts far more than broken labels.",
     "  About a hundred times more. That gap narrows when each site tracks "
     "more health variables, but it never closes."),
    ("5.  A clean benchmark would have hidden all of this.",
     "  On Breast Cancer every rule but Krum reaches 0.997 AUC or better, "
     "where no difference is visible. Messy self-reported population health "
     "data is where these rules actually separate."),
]

CONCLUSION = [
    "Across the datasets and partitioning studied here, most of the advantage "
    "robust aggregators hold over FedAvg comes from how they weight sites, "
    "not from the robust statistic they are named for.",
    ("What to do about it.", "  Any robustness evaluation run on size-skewed "
     "data should include an unweighted-mean control. It is one line of code, "
     "and without it a reported gain cannot be credited to robustness."),
    ("Why it matters in the field.", "  A rule's reputation is not evidence. "
     "Krum is widely cited and failed badly here. Coordinate-wise median is "
     "about as simple as merge rules get, and it held."),
]

CODE_AND_DATA = [
    "Code, data and the full experimental harness are open, with a script "
    "that re-checks every number on this poster.",
    "osf.io/d5u2q",
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
    "[6]  Centers for Disease Control and Prevention. BRFSS 2023 survey data "
    "and documentation.",
]

SECTIONS = {
    "TextBox 10": INTRODUCTION,
    "TextBox 12": PURPOSE,
    "TextBox 14": METHODS,
    "TextBox 7":  RESULTS,
    "TextBox 17": KEY_FINDINGS,
    "TextBox 21": CONCLUSION,
    "TextBox 23": REFERENCES,
}

# section -> top edge, in inches. The columns now start below the hook line.
TOPS = {
    "TextBox 10":  5.0,   # INTRODUCTION
    "TextBox 12": 12.8,   # PURPOSE  (below the schematic)
    "TextBox 14": 17.7,   # METHODS
    "TextBox 7":   5.0,   # RESULTS
    "TextBox 17":  5.0,   # KEY FINDINGS
    "TextBox 21": 17.0,   # CONCLUSION
    "TextBox 23": 24.8,   # REFERENCES
}

# figure, left, top, width, height (inches) — middle column is x=8.3, 7.4 wide
FIGURES = [
    ("posterD_federated.png",     0.5,  9.4, 6.6, 2.9),
    ("posterA_decomposition.png", 8.3,  9.3, 7.4, 8.4),
    ("posterB_both_datasets.png", 8.3, 18.1, 7.4, 8.2),
    ("posterC_runlevel.png",      8.3, 26.7, 7.4, 7.1),
    ("posterE_qr.png",            5.35, 31.8, 1.7, 1.7),
]


def _preserve_space(run):
    """Keep leading/trailing spaces in a run that abuts another run."""
    t = run._r.find(qn("a:t"))
    if t is not None:
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def set_para_text(para, text):
    """Replace a paragraph's text, keeping run 0's character formatting.

    `text` is either a string or a (bold lead, remainder) tuple, which is laid
    down as two runs cloned from run 0 so the lead inherits the template's font.
    """
    runs = para.runs
    r0 = runs[0]
    for r in runs[1:]:
        r._r.getparent().remove(r._r)

    if isinstance(text, tuple):
        lead, rest = text
        r0.text = lead
        r0.font.bold = True
        _preserve_space(r0)
        tail_el = copy.deepcopy(r0._r)
        r0._r.addnext(tail_el)
        tail = _Run(tail_el, para)
        tail.text = rest
        tail.font.bold = False
        _preserve_space(tail)
    else:
        r0.text = text


def fill(shape, body_lines):
    """Keep paragraph 0 (the section heading); rewrite the body beneath it."""
    tf = shape.text_frame
    paras = list(tf.paragraphs)
    body = [p for p in paras[1:] if p.runs]
    if not body:
        raise RuntimeError(f"{shape.name}: no styled body paragraph to clone")
    template_p = body[0]._p

    # drop every existing body paragraph
    for p in paras[1:]:
        p._p.getparent().remove(p._p)

    parent = tf._txBody
    for line in body_lines:
        new_p = copy.deepcopy(template_p)
        parent.append(new_p)
    # re-read and populate
    for para, line in zip(list(tf.paragraphs)[1:], body_lines):
        set_para_text(para, line)
        para.space_before = Pt(0)
        para.space_after = BODY_GAP


def clone(slide, src):
    """Append a copy of an existing shape and return the new shape object."""
    src._element.getparent().append(copy.deepcopy(src._element))
    return list(slide.shapes)[-1]


prs = Presentation(TEMPLATE)
slide = prs.slides[0]
shapes = {sh.name: sh for sh in slide.shapes}

# ---- header -------------------------------------------------------------
# At the template's 60 pt the title runs to three lines and pushes the white
# author text off the green bar onto the pale background. 44 pt keeps the
# whole block inside the 3.3 in bar.
hdr = shapes["TextBox 4"]
hdr.width = Inches(18.4)
title_tf = hdr.text_frame
for para, txt, size in zip(title_tf.paragraphs, [TITLE, AUTHORS, AFFIL], [44, 30, 30]):
    set_para_text(para, txt)
    para.runs[0].font.size = Pt(size)

# ---- geometry -----------------------------------------------------------
for name, top in TOPS.items():
    shapes[name].top = Inches(top)
shapes["TextBox 10"].height = Inches(4.2)   # INTRODUCTION, now two paragraphs
shapes["TextBox 7"].width = Inches(7.4)     # RESULTS, to match the figures below

# ---- body sections ------------------------------------------------------
for name, lines in SECTIONS.items():
    fill(shapes[name], lines)
    print(f"  filled {name:<12} ({len(lines)} paragraphs)")

# ---- the plain-language hook line, spanning all three columns -----------
banner = clone(slide, shapes["TextBox 12"])
btf = banner.text_frame
for p in list(btf.paragraphs)[1:]:            # drop PURPOSE's body
    p._p.getparent().remove(p._p)
set_para_text(btf.paragraphs[0], HOOK)
run = btf.paragraphs[0].runs[0]
run.font.size, run.font.bold, run.font.italic = Pt(28), True, True
banner.left, banner.top = Inches(0.5), Inches(3.5)
banner.width, banner.height = Inches(23.0), Inches(1.1)
print("  added hook line")

# ---- CODE & DATA panel in the lower-left, beside the QR ----------------
panel = clone(slide, shapes["TextBox 12"])    # clone PURPOSE for its styling
panel.left, panel.top = Inches(0.5), Inches(31.3)
panel.width, panel.height = Inches(4.6), Inches(2.6)
set_para_text(panel.text_frame.paragraphs[0], "CODE & DATA")
fill(panel, CODE_AND_DATA)
print("  added CODE & DATA panel")

# ---- swap the placeholder chart art for the real figures ----------------
ph = shapes.get("Picture 5")
if ph is not None:
    ph._element.getparent().remove(ph._element)
    print("  removed placeholder chart art")

for fn, l, t, w, h in FIGURES:
    slide.shapes.add_picture(os.path.join(FIGDIR, fn),
                             Inches(l), Inches(t), Inches(w), Inches(h))
    print(f"  placed {fn} at y={t}")

prs.save(OUT)
print("\nwrote", OUT)
