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

LOREM = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore "
         "magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo").split()

def lorem(md):
    """Fill each drafting note's '~N words' budget with N words of filler (paragraphs of ~120), to preview length."""
    def fill(m):
        n = int(m.group(1)); w = [LOREM[i % len(LOREM)] for i in range(n)]
        paras = [" ".join(w[i:i + 120]).capitalize() + "." for i in range(0, n, 120)]
        return m.group(0) + "\n\n" + "\n\n".join(paras)
    return re.sub(r"<!-- ~(\d+) words.*?-->", fill, md, flags=re.S)

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    md = TAG.sub(lambda m: m.group(2), SRC.read_text())
    if "--lorem" in sys.argv: md = lorem(md); OUT = OUT / "lorem"; OUT.mkdir(exist_ok=True)
    body = md.split("<!-- appendices -->")[0]
    (OUT / "report.md").write_text(md); (OUT / "body.md").write_text(body)
    for name in ("report", "body"):
        subprocess.run(["pandoc", str(OUT / f"{name}.md"), "-o", str(OUT / f"{name}.pdf"), *OPTS], check=True)
    subprocess.run(["pandoc", str(OUT / "report.md"), "-o", str(OUT / "report.docx")], check=True)
    words = len(re.sub(r"<!--.*?-->", "", body, flags=re.S).split())
    n = pages(OUT / "body.pdf")
    print(f"main body: {n} pages (limit 8), ~{words} words; full PDF: {pages(OUT / 'report.pdf')} pages -> {OUT}")
    if n > 8: sys.exit("over the 8-page limit")
