"""Build the PDF deliverables: ``reports/report.pdf`` and ``reports/notebook.pdf``.

* report: ``reports/report.md`` -> HTML (Python-Markdown, print CSS) -> PDF via
  headless Chrome/Chromium (no LaTeX needed). Appendix A (the complete program
  listing) is generated from the current source by ``build_code_appendix.py``
  and inserted where the Markdown contains its marker.
* notebook: ``notebooks/CyberReplay.ipynb`` (already executed) -> HTML via
  nbconvert -> PDF via headless Chrome.

Usage::

    python scripts/build_pdfs.py            # both
    python scripts/build_pdfs.py report     # only the report

Prints the page count of each PDF. The main report (sections 1-7 and the
References) must fit in 3-5 pages; Section 8 (key code), the figure appendix and
Appendix A each start on a new page after the References and are not counted.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_code_appendix import APPENDIX_MARKER, appendix_html  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Figures are drawn at print size (src/cyberreplay/plotstyle.py: 7.2 in full width), so the
# 12.5 mm side margins give a text width of 185 mm and full-width images print almost 1:1.
REPORT_CSS = """
@page { size: A4; margin: 11mm 12.5mm 11mm 12.5mm; }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 9pt; line-height: 1.27; color: #111; }
h1 { font-size: 14pt; margin: 0 0 1pt 0; text-align: center; line-height: 1.2; }
.subtitle { text-align: center; font-size: 8.8pt; font-style: italic; margin: 0 0 3pt 0; }
h2 { font-size: 10.6pt; margin: 6pt 0 2pt 0; border-bottom: 0.8px solid #777; break-after: avoid; }
h3 { font-size: 9.4pt; margin: 4pt 0 1pt 0; break-after: avoid; }
p { margin: 1.5pt 0 3pt 0; text-align: justify; hyphens: auto; overflow-wrap: anywhere; }
ul, ol { margin: 1pt 0 3pt 0; padding-left: 15pt; }
li { margin: 0 0 1pt 0; text-align: justify; }
ol.refs { padding-left: 20pt; font-size: 7.9pt; line-height: 1.17; margin-top: 0; }
ol.refs li { margin: 0 0 0.5pt 0; text-align: left; }
table { border-collapse: collapse; margin: 3pt auto 4pt auto; font-size: 7.9pt; break-inside: avoid; }
th, td { border: 0.7px solid #444; padding: 1pt 3.5pt; text-align: right; }
th { font-weight: bold; background: none; }
td:first-child, th:first-child { text-align: left; }
.tight table { font-size: 7.5pt; } .tight th, .tight td { padding: 0.8pt 2.6pt; } .tight td { white-space: nowrap; }
.left table td, .left table th { text-align: left; }
.left table td.r, .left table td:nth-child(n+4) { text-align: right; }
.hp table td:last-child { text-align: left; }
img { display: block; margin: 2pt auto 0 auto; max-width: 100%; }
.figure { break-inside: avoid; margin: 3pt 0 4pt 0; }
.figure img.full { width: 100%; }
.row { display: flex; gap: 8pt; justify-content: space-between; align-items: flex-start; }
.row > .col { flex: 1 1 0; min-width: 0; }
.row > .col img { width: 100%; }
.side { display: flex; gap: 10pt; align-items: flex-start; margin: 2pt 0 3pt 0; }
.side > .figcol { flex: 0 0 49%; break-inside: avoid; }
.side > .figcol img { width: 100%; }
.side > .textcol { flex: 1 1 0; min-width: 0; }
.cap { text-align: left; font-size: 7.9pt; line-height: 1.2; color: #222; margin: 1.5pt 0 0 0; }
.title-block { text-align: center; font-size: 9pt; margin-bottom: 4pt; }
.summary { border: 1px solid #888; background: #f4f4f4; padding: 3pt 7pt; margin: 2pt 0 5pt 0; }
.summary p { margin: 1pt 0; }
code { font-size: 8pt; }
pre { font-family: "DejaVu Sans Mono", monospace; font-size: 7.1pt; line-height: 1.2; background: #f6f6f6;
      border: 1px solid #ddd; padding: 3pt 4pt; margin: 2pt 0 3pt 0; white-space: pre-wrap; overflow-wrap: anywhere;
      break-inside: avoid; }
pre code { font-size: inherit; }
.newpage { break-before: page; }
.code-appendix .banner { font-family: "DejaVu Sans Mono", monospace; font-size: 7.4pt; font-weight: bold;
      background: #e4e4e4; border: 0.6px solid #bbb; padding: 1.5pt 4pt; margin: 6pt 0 2pt 0; break-after: avoid; }
.code-appendix pre.listing { font-size: 6.7pt; line-height: 1.13; background: none; border: none; padding: 0;
      margin: 0; white-space: pre; overflow-wrap: normal; break-inside: auto; }
.code-appendix pre.cellbanner { color: #555; font-weight: bold; margin-top: 3pt; break-after: avoid; }
.code-appendix .no { color: #777; }
.keep { break-inside: avoid; }
"""

# The main body ends where Section 8 (code) begins on a new page.
MAIN_BODY_END_MARKER = "8. Key code, explained"
# First line of each part of the report, in order (used to print page ranges).
PART_MARKERS = {"Section 8": MAIN_BODY_END_MARKER, "Figure appendix": "Figure appendix",
                "Appendix A": "Appendix A. Complete program"}


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
    if APPENDIX_MARKER not in body:
        raise RuntimeError(f"{APPENDIX_MARKER} missing from report.md")
    body = body.replace(APPENDIX_MARKER, appendix_html())
    html = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><style>{REPORT_CSS}</style></head>" \
           f"<body>{body}</body></html>"
    html_path = REPORTS / "report.html"
    html_path.write_text(html)
    pages = html_to_pdf(html_path, REPORTS / "report.pdf")
    html_path.unlink()
    main = main_body_pages(REPORTS / "report.pdf")
    print(f"report.pdf: {pages} pages (main body, sections 1-7 + References: {main} pages)")
    for part, (first, last) in part_ranges(REPORTS / "report.pdf").items():
        print(f"  {part}: pages {first}-{last}")
    return pages


def part_ranges(pdf_path: Path) -> dict[str, tuple[int, int]]:
    """Page range (1-based, inclusive) of the main body and of each appendix part.

    A part starts on the first page whose text begins with its marker.

    Args:
        pdf_path: Built report PDF.

    Returns:
        ``{part: (first page, last page)}`` in document order.
    """
    texts = [(p.extract_text() or "").lstrip() for p in PdfReader(str(pdf_path)).pages]
    starts = {"Main body": 1}
    for part, marker in PART_MARKERS.items():
        page = next((i + 1 for i, t in enumerate(texts) if t.startswith(marker)), None)
        if page:
            starts[part] = page
    names = list(starts)
    ends = [starts[n] - 1 for n in names[1:]] + [len(texts)]
    return {n: (starts[n], e) for n, e in zip(names, ends)}


def main_body_pages(pdf_path: Path) -> int:
    """Count the pages before the one on which Section 8 starts.

    Args:
        pdf_path: Built report PDF.

    Returns:
        Number of main-body pages (all pages if the marker is missing).
    """
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages):
        if MAIN_BODY_END_MARKER in (page.extract_text() or ""):
            return i
    return len(reader.pages)


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
