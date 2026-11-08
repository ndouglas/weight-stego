"""Build the report: strip claim tags, render PDF (and .docx for pasting into Apart's Google Doc template), count pages.

Claim tags: write a number as [claim.id=value]; `make check` verifies value against results/claims.md, and the build
replaces the tag with its value. Everything after the line `<!-- appendices -->` (references, appendices) is excluded
from the page count, as Apart's 8-page limit excludes them.

The page count is an estimate (11pt, 1in margins, single column, close to the Google Doc template); confirm once in the
template itself.
"""
import re, subprocess, sys
from hosts import ROOT

TAG = re.compile(r"\[([a-zA-Z0-9_.\-]+)=([^\]]+)\]")
SRC, OUT = ROOT / "report" / "report.md", ROOT / "report" / "build"
OPTS = ["-V", "geometry:margin=1in", "-V", "fontsize=11pt", "-V", "mainfont=Helvetica", "--pdf-engine=xelatex"]

def pages(pdf):
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", info).group(1))

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    md = TAG.sub(lambda m: m.group(2), SRC.read_text())
    body = md.split("<!-- appendices -->")[0]
    (OUT / "report.md").write_text(md); (OUT / "body.md").write_text(body)
    for name in ("report", "body"):
        subprocess.run(["pandoc", str(OUT / f"{name}.md"), "-o", str(OUT / f"{name}.pdf"), *OPTS], check=True)
    subprocess.run(["pandoc", str(OUT / "report.md"), "-o", str(OUT / "report.docx")], check=True)
    words = len(re.sub(r"<!--.*?-->", "", body, flags=re.S).split())
    n = pages(OUT / "body.pdf")
    print(f"main body: {n} pages (limit 8), ~{words} words; full PDF: {pages(OUT / 'report.pdf')} pages -> {OUT}")
    if n > 8: sys.exit("over the 8-page limit")
