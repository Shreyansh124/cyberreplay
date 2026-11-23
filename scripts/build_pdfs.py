"""Build the PDF deliverables: ``reports/report.pdf`` and ``reports/notebook.pdf``.

* report: ``reports/report.md`` -> HTML (Python-Markdown, print CSS) -> PDF via
  headless Chrome/Chromium (no LaTeX needed).
* notebook: ``notebooks/CyberReplay.ipynb`` (already executed) -> HTML via
  nbconvert -> PDF via headless Chrome.

Usage::

    python scripts/build_pdfs.py            # both
    python scripts/build_pdfs.py report     # only the report

Prints the page count of each PDF (the report must be 3-5 pages).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

REPORT_CSS = """
@page { size: A4; margin: 13mm 13mm 13mm 13mm; }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 9.6pt; line-height: 1.32; color: #111; }
h1 { font-size: 15pt; margin: 0 0 2pt 0; text-align: center; }
h2 { font-size: 11.5pt; margin: 8pt 0 3pt 0; border-bottom: 1px solid #999; }
h3 { font-size: 10pt; margin: 6pt 0 2pt 0; }
p { margin: 2pt 0 4pt 0; text-align: justify; }
ul, ol { margin: 2pt 0 4pt 0; padding-left: 16pt; }
li { margin: 0 0 1pt 0; }
table { border-collapse: collapse; margin: 4pt auto; font-size: 8.2pt; }
th, td { border: 1px solid #bbb; padding: 1.5pt 4pt; text-align: right; }
th { background: #eee; }
td:first-child, th:first-child { text-align: left; }
img { display: block; margin: 3pt auto; max-width: 100%; }
.fig { text-align: center; font-size: 8pt; color: #333; margin: 0 0 5pt 0; }
.title-block { text-align: center; font-size: 9.5pt; margin-bottom: 6pt; }
.cols { display: flex; gap: 6pt; } .cols img { width: 49%; }
code { font-size: 8.5pt; }
"""


def find_chrome() -> str:
    """Return the path of a headless-capable Chrome/Chromium binary.

    The real binaries are preferred over ``google-chrome`` because that name is
    often a wrapper script that pins a shared profile, which blocks headless printing.
    """
    for name in ("/opt/google/chrome/chrome", "google-chrome-stable", "chromium", "chromium-browser",
                 "google-chrome", "chrome"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("Chrome/Chromium not found; install it to build the PDFs.")


def html_to_pdf(html_path: Path, pdf_path: Path) -> int:
    """Print an HTML file to PDF with headless Chrome and return the page count.

    Args:
        html_path: Input HTML (relative image paths resolve against its folder).
        pdf_path: Output PDF.

    Returns:
        Number of pages in the PDF.
    """
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([find_chrome(), "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        "--no-first-run", f"--user-data-dir={profile}", "--allow-file-access-from-files",
                        "--virtual-time-budget=20000", f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    return len(PdfReader(str(pdf_path)).pages)


def build_report() -> int:
    """Convert ``reports/report.md`` to ``reports/report.pdf``.

    Returns:
        Page count.

    Side effects:
        Writes ``reports/report.html`` (intermediate) and ``reports/report.pdf``.
    """
    md = (REPORTS / "report.md").read_text()
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "attr_list", "md_in_html"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{REPORT_CSS}</style></head>" \
           f"<body>{body}</body></html>"
    html_path = REPORTS / "report.html"
    html_path.write_text(html)
    pages = html_to_pdf(html_path, REPORTS / "report.pdf")
    html_path.unlink()
    print(f"report.pdf: {pages} pages")
    return pages


def build_notebook() -> int:
    """Export the executed notebook to ``reports/notebook.pdf``.

    Returns:
        Page count.
    """
    nb = ROOT / "notebooks" / "CyberReplay.ipynb"
    out_dir = REPORTS / "_nb_html"
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", str(nb), "--output-dir",
                    str(out_dir), "--output", "CyberReplay"], check=True)
    pages = html_to_pdf(out_dir / "CyberReplay.html", REPORTS / "notebook.pdf")
    shutil.rmtree(out_dir)
    print(f"notebook.pdf: {pages} pages")
    return pages


if __name__ == "__main__":
    what = sys.argv[1:] or ["report", "notebook"]
    if "report" in what:
        build_report()
    if "notebook" in what:
        build_notebook()
