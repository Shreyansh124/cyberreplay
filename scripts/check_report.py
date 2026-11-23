"""Automated checks of the built report (run after ``python scripts/build_pdfs.py report``).

Checks:
    * ``reports/report.pdf`` and ``report.md`` never contain the word "github";
    * all four team members, their roll numbers and the course appear on page 1;
    * the main body (sections 1-7 and the References) is at most 5 pages and
      Section 8 starts at the top of a new page;
    * every line-numbered code excerpt in Section 8 matches the source file;
    * every number in the results tables of a baseline version of the report
      (``--baseline-ref``, a git revision) still appears in the current report;
    * Section 1 has 90-120 words, and the prose avoids stock phrases and arrows;
    * figure captions are numbered 1..N in order and every analysis figure in
      ``reports/figures`` is used.

Usage::

    python scripts/check_report.py [--baseline-ref 1e63a0f]

Exits with status 1 and lists the failures if any check fails.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "reports" / "report.md"
REPORT_PDF = ROOT / "reports" / "report.pdf"
# Report version before the layout rewrite; its table numbers must all survive.
DEFAULT_BASELINE = "1e63a0f"
PAGE1_TEXT = ["Shreyansh Krishna", "2026CS05055", "Aamir Lanker", "2026CS05054", "Harshith G. D", "2026CS05053",
              "Anil Pandey", "2026CS05031", "Course: Machine Learning"]
STOCK_PHRASES = ["delve", "leverage", "robust", "seamless", "crucial"]
SECTION8 = "## 8. Key code, explained"
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:-\d[\d,]*(?:\.\d+)?)?%?")


def check_pdf(failures: list[str]) -> int:
    """Check the PDF text: no "github", names on page 1, page budget, Section 8 on a new page.

    Args:
        failures: List that receives failure messages.

    Returns:
        Number of main-body pages.
    """
    pages = [p.extract_text() or "" for p in PdfReader(str(REPORT_PDF)).pages]
    if any("github" in t.lower() for t in pages):
        failures.append("report.pdf contains 'github'")
    flat1 = " ".join(pages[0].split())
    for s in PAGE1_TEXT:
        if s not in flat1:
            failures.append(f"page 1 is missing {s!r}")
    start = next((i for i, t in enumerate(pages) if "8. Key code, explained" in t), None)
    if start is None:
        failures.append("Section 8 heading not found in the PDF")
        return len(pages)
    if not pages[start].lstrip().startswith("8. Key code, explained"):
        failures.append("Section 8 does not start at the top of a page")
    if start > 5:
        failures.append(f"main body is {start} pages (limit 5)")
    return start


def check_excerpts(md: str, failures: list[str]) -> int:
    """Compare every numbered line of the Section 8 code blocks with the source file.

    Args:
        md: Report Markdown.
        failures: List that receives failure messages.

    Returns:
        Number of excerpt lines checked.
    """
    body = md[md.index(SECTION8):]
    checked = 0
    for sec in re.split(r"\n### ", body)[1:]:
        heading = sec.splitlines()[0]
        default = re.search(r"`(src/[\w/.]+\.py)`", heading)
        current = default.group(1) if default else None
        block = re.search(r"```\n(.*?)```", sec, flags=re.S)
        if not block:
            continue
        for line in block.group(1).splitlines():
            path = re.match(r"\s*\[(src/[\w/.]+\.py)\]$", line)
            if path:
                current = path.group(1)
                continue
            m = re.match(r"\s*(\d+)  (.*)$", line)
            if not m:
                continue
            src = (ROOT / current).read_text().splitlines()
            n, text = int(m.group(1)), m.group(2)
            if n > len(src) or src[n - 1].rstrip() != text.rstrip():
                failures.append(f"excerpt mismatch {current}:{n}: {text.strip()[:60]!r}")
            checked += 1
    return checked


def baseline_numbers(ref: str) -> set[str]:
    """Numbers that appear in the Markdown tables of the report at a git revision.

    Args:
        ref: Git revision of the baseline report.

    Returns:
        Set of number tokens (thousands separators, decimals, ranges and % kept).
    """
    old = subprocess.run(["git", "show", f"{ref}:reports/report.md"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    nums: set[str] = set()
    for line in old.splitlines():
        if line.startswith("|") and not set(line) <= set("|:- "):
            nums.update(NUMBER.findall(line))
    return nums


def main_body(md: str) -> str:
    """Report text before Section 8, with code spans and HTML tags removed."""
    text = md[:md.index(SECTION8)]
    text = re.sub(r"`[^`]*`", " ", text)
    return re.sub(r"<[^>]+>", " ", text)


def check_prose(md: str, failures: list[str]) -> int:
    """Section 1 length, stock phrases and arrows in the main body.

    Args:
        md: Report Markdown.
        failures: List that receives failure messages.

    Returns:
        Word count of Section 1.
    """
    s1 = md[md.index("## 1."):md.index("## 2.")].split("\n", 1)[1]
    words = len(re.sub(r"[`*]", "", s1).split())
    if not 90 <= words <= 120:
        failures.append(f"Section 1 has {words} words (90-120 wanted)")
    prose = main_body(md).lower()
    for w in STOCK_PHRASES:
        if re.search(rf"\b{w}", prose):
            failures.append(f"stock phrase {w!r} in the main body")
    if "→" in prose or "->" in prose:
        failures.append("arrow in the main body")
    return words


def check_figures(md: str, failures: list[str]) -> int:
    """Figure captions numbered in order and every analysis PNG used.

    Args:
        md: Report Markdown.
        failures: List that receives failure messages.

    Returns:
        Number of figures.
    """
    nums = [int(n) for n in re.findall(r"<b>Fig (\d+)\.</b>", md)]
    if nums != list(range(1, len(nums) + 1)):
        failures.append(f"figure captions out of order: {nums}")
    used = set(re.findall(r'src="figures/([\w.]+\.png)"', md))
    for png in sorted((ROOT / "reports" / "figures").glob("[0-9]*.png")):
        if png.name not in used:
            failures.append(f"figure not in the report: {png.name}")
    return len(nums)


def main(argv: list[str] | None = None) -> None:
    """Run every check, print a summary and exit 1 on failure."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-ref", default=DEFAULT_BASELINE, help="git revision of the baseline report")
    args = ap.parse_args(argv)
    md = REPORT_MD.read_text()
    failures: list[str] = []
    if "github" in md.lower():
        failures.append("report.md contains 'github'")
    pages = check_pdf(failures)
    lines = check_excerpts(md, failures)
    current = set(NUMBER.findall(md))
    old = baseline_numbers(args.baseline_ref)
    missing = sorted(old - current)
    if missing:
        failures.append(f"baseline table numbers missing: {missing}")
    words = check_prose(md, failures)
    figs = check_figures(md, failures)
    print(f"main body pages: {pages}; excerpt lines checked: {lines}; baseline table numbers: {len(old)} "
          f"(missing {len(missing)}); Section 1 words: {words}; figures: {figs}")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("all report checks passed")


if __name__ == "__main__":
    main()
