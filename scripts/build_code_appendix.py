"""Generate Appendix A of the report: the complete program, every line with its line number.

The listing is built from the current files at PDF build time, so it always
matches the source:

    (a) every code cell of ``notebooks/CyberReplay.ipynb`` in run order, each under
        a ``# ====== Cell N ======`` banner (line numbers restart in every cell);
    (b) every module of ``src/cyberreplay/``;
    (c) ``app/streamlit_app.py``.

Lines are copied verbatim (tabs expanded) and prefixed with their line number.
Lines longer than ``WRAP`` characters are wrapped, and each continuation line is
marked with ``↪`` in the line-number column. The only change is ``REDACTIONS``:
the repository clone URL in notebook cell 1 is replaced by a placeholder,
because the report must not contain a repository URL.

``scripts/build_pdfs.py`` calls ``appendix_html`` and inserts the result where
``reports/report.md`` contains ``APPENDIX_MARKER``; ``scripts/check_report.py``
uses ``expected_lines`` to verify the printed listing.

Usage::

    python scripts/build_code_appendix.py    # print the line count of every listed file
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "notebooks/CyberReplay.ipynb"
MODULES = ["__init__.py", "config.py", "utils.py", "plotstyle.py", "data.py", "clean.py", "eda.py", "features.py",
           "models.py", "tuning.py", "evaluate.py", "diagnostics.py", "explain.py", "replay.py", "pipeline.py"]
FILES = [NOTEBOOK] + [f"src/cyberreplay/{m}" for m in MODULES] + ["app/streamlit_app.py"]
APPENDIX_MARKER = "<!-- CODE_APPENDIX -->"
# Characters of code per printed line; with the 5-digit number column this fits the
# 185 mm text width at 6.7 pt DejaVu Sans Mono.
WRAP = 118
REDACTIONS = {"https://github.com/Shreyansh124/cyberreplay.git": "<repository URL, omitted in this listing>"}


def redact(line: str) -> str:
    """Apply ``REDACTIONS`` to one source line."""
    for old, new in REDACTIONS.items():
        line = line.replace(old, new)
    return line


def notebook_cells(path: Path) -> list[list[str]]:
    """Source lines of every code cell of an executed notebook, in run order.

    Args:
        path: ``.ipynb`` file.

    Returns:
        One list of lines per code cell, ordered by execution count (cells that
        were never executed keep their position at the end).
    """
    nb = json.loads(path.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    cells.sort(key=lambda c: c.get("execution_count") or 10**9)
    return ["".join(c["source"]).splitlines() for c in cells]


def file_units(rel: str) -> list[tuple[str | None, list[str]]]:
    """Split a listed file into numbered units.

    Args:
        rel: Path relative to the project root (one of ``FILES``).

    Returns:
        ``[(cell banner or None, lines)]``: one unit per notebook code cell, or a
        single unit for a Python file. Lines are redacted and tab-expanded.
    """
    path = ROOT / rel
    if rel.endswith(".ipynb"):
        return [(f"# ====== Cell {i} ======", [redact(l).expandtabs(4) for l in lines])
                for i, lines in enumerate(notebook_cells(path), 1)]
    return [(None, [redact(l).expandtabs(4) for l in path.read_text().splitlines()])]


def expected_lines() -> list[tuple[str, int, str]]:
    """Every listed line in print order.

    Returns:
        ``[(file, line number, text)]`` (line numbers restart in each notebook cell).
    """
    return [(rel, n, line) for rel in FILES for _, lines in file_units(rel) for n, line in enumerate(lines, 1)]


def line_counts() -> dict[str, int]:
    """Number of listed lines per file (all code cells together for the notebook)."""
    return {rel: sum(len(lines) for _, lines in file_units(rel)) for rel in FILES}


def wrap(text: str, width: int = WRAP) -> list[str]:
    """Cut a line into chunks of at most ``width`` characters (no characters dropped)."""
    return [text[i:i + width] for i in range(0, len(text), width)] or [""]


def listing_html(lines: list[str]) -> str:
    """One ``<pre>`` block with numbered, wrapped lines.

    Args:
        lines: Source lines of one unit (numbered from 1).

    Returns:
        HTML string.
    """
    out = []
    for n, line in enumerate(lines, 1):
        for k, chunk in enumerate(wrap(line)):
            mark = f"{n:>5}" if k == 0 else f"{'↪':>5}"
            out.append(f'<span class="no">{mark}</span>  {html.escape(chunk)}')
    return '<pre class="listing">' + "\n".join(out) + "</pre>"


def appendix_html() -> str:
    """The whole of Appendix A as HTML (heading, note, one banner and listing per file).

    Returns:
        HTML string, starting on a new page.
    """
    counts = line_counts()
    parts = [
        '<div class="newpage"></div>',
        '<h2 id="appendix-a">Appendix A. Complete program</h2>',
        "<p>We list the complete program here: every code cell of the notebook in run order, every module of the "
        "<code>cyberreplay</code> package and the Streamlit dashboard, each line with its line number in that file "
        "(line numbers restart in each notebook cell, and ↪ marks the continuation of a long line). The unit tests in "
        "<code>tests/</code> and the scripts that build the notebook and the PDFs are not part of the ML program, so "
        "we omit them, and the repository URL in notebook cell 1 is replaced by a placeholder. Running the notebook "
        "from top to bottom reproduces the results in this report.</p>",
    ]
    for rel in FILES:
        what = "code cells in run order" if rel.endswith(".ipynb") else "module"
        parts.append(f'<div class="banner">{html.escape(rel)} &nbsp;·&nbsp; {what} &nbsp;·&nbsp; '
                     f'{counts[rel]:,} lines</div>')
        for banner, lines in file_units(rel):
            if banner:
                parts.append(f'<pre class="listing cellbanner">{html.escape(banner)}</pre>')
            parts.append(listing_html(lines))
    return '<div class="code-appendix">' + "\n".join(parts) + "</div>"


if __name__ == "__main__":
    for rel, n in line_counts().items():
        print(f"{n:>6}  {rel}")
    print(f"{sum(line_counts().values()):>6}  total")
