r"""
render_diagram.py -- rasterise diagram_federated.html with headless Chrome.

The diagram is authored as HTML + inline SVG (the cathrynlavery/diagram-design
output format) rather than drawn in matplotlib, so the hairlines, corner radii
and text tracking come out exactly as specified. Chrome renders it at 3x so
the 1080x396 viewBox lands at 3240x1188 px -- about 450 dpi once the PNG is
placed 7.26 in wide on the board.

Run from the repo root:  python poster/render_diagram.py
"""
import os
import subprocess
import sys
import tempfile

HTML = os.path.abspath("poster/diagram_federated.html")
OUT = os.path.abspath("poster/figures/posterD_federated.png")
SCALE = 3
W, H = 1080, 456

CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

browser = next((p for p in CANDIDATES if os.path.exists(p)), None)
if browser is None:
    sys.exit("no Chrome or Edge found; install one or render the SVG by hand")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with tempfile.TemporaryDirectory() as profile:
    subprocess.run([
        browser,
        "--headless=new",
        "--disable-gpu",
        f"--user-data-dir={profile}",
        f"--screenshot={OUT}",
        f"--window-size={W},{H}",
        f"--force-device-scale-factor={SCALE}",
        "--hide-scrollbars",
        "--virtual-time-budget=4000",
        f"file:///{HTML.replace(os.sep, '/')}",
    ], check=True, capture_output=True)

print(f"  saved {OUT}  ({W * SCALE}x{H * SCALE})")
