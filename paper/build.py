#!/usr/bin/env python3
"""Compile paper.md -> paper.typ -> paper.pdf.

Edit paper.md (and put images under figures/). Run:

  python build.py          # one-shot compile
  python watch.py          # recompile on save

Requires: typst on PATH (brew install typst).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD_PATH = HERE / "paper.md"
TYP_PATH = HERE / "paper.typ"
PDF_PATH = HERE / "paper.pdf"
BIB_PATH = HERE / "refs.bib"

# ---------------------------------------------------------------------------
# Typst style template (layout only). Content comes from paper.md.
# Spacing is intentionally airy between paragraphs and headings.
# ---------------------------------------------------------------------------
TEMPLATE_HEAD = r'''// Auto-generated from paper.md by build.py — do not edit by hand.
// Edit paper.md, then run: python build.py   (or python watch.py)

#set document(
  title: "{title}",
  author: ({authors}),
)

#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  numbering: "1",
  number-align: center,
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en",
)

// More air between paragraphs
#set par(
  justify: true,
  leading: 0.78em,
  first-line-indent: 1.2em,
  spacing: 1.05em,
)

#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {{
  set text(size: 12pt, weight: "bold")
  v(2.0em, weak: true)
  block(below: 1.15em, it)
}}
#show heading.where(level: 2): it => {{
  set text(size: 11pt, weight: "bold")
  v(1.55em, weak: true)
  block(below: 0.95em, it)
}}

#show figure: set block(breakable: false, spacing: 1.6em)
#show figure.caption: set text(size: 9pt)
#set figure(gap: 0.7em)
#set math.equation(numbering: "(1)")

// Title block
#align(center)[
  #text(size: 14pt, weight: "bold")[
{title_lines}
  ]
  #v(1.2em)
  #text(size: 10.5pt)[
{author_line}
  ]
  #v(0.6em)
  #text(size: 9.5pt)[
    #super[1]{affiliation}
  ]
]

#v(1.6em)

// Abstract: centered label, then body
#align(center)[
  #text(size: 11pt, weight: "bold")[Abstract]
]
#v(0.7em)
#par(first-line-indent: 0em, leading: 0.78em, spacing: 1.05em)[
  {abstract}
]

#v(1.5em)

'''

TEMPLATE_TAIL = r'''
#pagebreak()

#bibliography("refs.bib", title: "References", style: "apa")
'''


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    raw = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    meta: dict = {"authors": [], "title": "", "affiliation": "", "abstract": ""}
    key = None
    abs_lines: list[str] = []
    in_abstract = False
    for line in raw.splitlines():
        if in_abstract:
            if line.startswith("  ") or line.startswith("\t") or line.strip() == "":
                abs_lines.append(line[2:] if line.startswith("  ") else line.lstrip("\t"))
                continue
            in_abstract = False
            meta["abstract"] = "\n".join(abs_lines).strip()
            abs_lines = []
        m = re.match(r"^([a-zA-Z_]+):\s*(.*)$", line)
        if m:
            key, val = m.group(1), m.group(2)
            if key == "abstract" and val.strip() in ("|", ">"):
                in_abstract = True
                abs_lines = []
            elif key == "authors":
                meta["authors"] = []
            elif key in ("title", "affiliation"):
                meta[key] = val.strip().strip('"').strip("'")
            else:
                meta[key] = val.strip().strip('"').strip("'")
            continue
        if key == "authors" and re.match(r"^\s+-\s+", line):
            meta["authors"].append(re.sub(r"^\s+-\s+", "", line).strip())
    if in_abstract:
        meta["abstract"] = "\n".join(abs_lines).strip()
    return meta, body



def fix_inline(s: str) -> str:
    # Order matters: bold then italic
    s = re.sub(r"\*\*([^*]+)\*\*", r"#strong[\1]", s)
    s = re.sub(r"(?<![\\*])\*([^*]+)\*(?!\*)", r"#emph[\1]", s)
    s = re.sub(r"(?<![\\_])_([^_]+)_(?!_)", r"#emph[\1]", s)
    s = re.sub(r"\^([^\^]+)\^", r"#super[\1]", s)
    return s


def parse_image_line(line: str) -> dict | None:
    """Parse ![caption](path){#id width=NN%}"""
    m = re.match(
        r"!\[(.*)\]\(([^)]+)\)(?:\{([^}]*)\})?\s*$",
        line,
    )
    if not m:
        return None
    caption, path, attrs = m.group(1), m.group(2), m.group(3) or ""
    label = None
    width = "72%"
    for part in attrs.split():
        if part.startswith("#"):
            label = part[1:]
        elif part.startswith("width="):
            w = part.split("=", 1)[1]
            width = w if w.endswith("%") else f"{w}%"
    return {"caption": caption, "path": path, "label": label, "width": width}


def is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and "|" in line.strip()[1:]


def is_table_sep(line: str) -> bool:
    s = line.strip()
    return bool(re.match(r"^\|[\s:|-]+\|$", s)) or bool(
        re.match(r"^\|[-:| ]+\|$", s)
    )


def parse_table_caption(line: str) -> tuple[str, str | None] | None:
    """': Caption text. {#tab:id}'"""
    m = re.match(r"^:\s+(.*?)(?:\s+\{#([^}]+)\})?\s*$", line)
    if not m:
        return None
    return m.group(1).strip(), m.group(2)


def table_to_typst(rows: list[list[str]], caption: str, label: str | None) -> str:
    ncols = max(len(r) for r in rows) if rows else 0
    # bold header cells that are plain text
    header = rows[0] if rows else []
    body_rows = rows[1:] if len(rows) > 1 else []

    def cell(c: str, header_cell: bool = False) -> str:
        c = c.strip()
        c = fix_inline(c)
        if header_cell and not c.startswith("*") and not c.startswith("#"):
            return f"[*{c}*]"
        return f"[{c}]"

    cols = ", ".join(["auto"] * ncols)
    # right-align numeric-looking columns (all but first)
    aligns = ["left"] + ["right"] * max(0, ncols - 1)
    align_s = ", ".join(aligns)

    lines = [
        "#figure(",
        "  table(",
        f"    columns: ({cols}),",
        f"    align: ({align_s}),",
        "    stroke: none,",
        "    inset: (x: 0.55em, y: 0.5em),",
        "    table.hline(stroke: 0.8pt),",
        "    table.header(",
        "      " + ", ".join(cell(c, True) for c in header) + ",",
        "    ),",
        "    table.hline(stroke: 0.5pt),",
    ]
    for r in body_rows:
        # pad short rows
        while len(r) < ncols:
            r.append("")
        lines.append("    " + ", ".join(cell(c) for c in r[:ncols]) + ",")
    lines.append("    table.hline(stroke: 0.8pt),")
    lines.append("  ),")
    lines.append(f"  caption: [{fix_inline(caption)}],")
    lines.append(")")
    if label:
        lines[-1] = f") <{label}>"
    return "\n".join(lines)


def figure_to_typst(info: dict) -> str:
    cap = fix_inline(info["caption"])
    # allow $math$ in captions
    lines = [
        "#figure(",
        f'  image("{info["path"]}", width: {info["width"]}),',
        f"  caption: [{cap}],",
        "  placement: top,",
        ")",
    ]
    if info.get("label"):
        lines[-1] = f") <{info['label']}>"
    return "\n".join(lines)


def convert_math_fence(lang: str, content: str) -> str:
    """```typst-math [eq:label] ... ``` -> Typst equation."""
    parts = lang.split()
    label = None
    for p in parts[1:]:
        if p.startswith("eq:") or p.startswith("eq."):
            label = p
    body = content.strip()
    if label:
        return f"$\n{body}\n$ <{label}>"
    return f"$\n{body}\n$"


def body_md_to_typst(body: str) -> str:
    lines = body.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # fenced code
        if line.startswith("```"):
            lang = line[3:].strip()
            i += 1
            buf: list[str] = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            if i < n:
                i += 1  # closing fence
            content = "\n".join(buf)
            if lang.startswith("typst-math") or lang.startswith("math"):
                out.append(convert_math_fence(lang, content))
            elif lang == "typst":
                out.append(content)
            else:
                # plain code block as raw text
                out.append("```\n" + content + "\n```")
            out.append("")
            continue

        # headings
        if line.startswith("# "):
            out.append("= " + line[2:].strip())
            out.append("")
            i += 1
            continue
        if line.startswith("## "):
            out.append("== " + line[3:].strip())
            out.append("")
            i += 1
            continue
        if line.startswith("### "):
            out.append("=== " + line[4:].strip())
            out.append("")
            i += 1
            continue

        # image
        img = parse_image_line(line.strip())
        if img:
            out.append(figure_to_typst(img))
            out.append("")
            i += 1
            continue

        # table
        if is_table_row(line):
            table_lines = []
            while i < n and is_table_row(lines[i]):
                table_lines.append(lines[i])
                i += 1
            # optional caption line (allow a blank line between table and : caption)
            caption, label = "", None
            j = i
            while j < n and not lines[j].strip():
                j += 1
            if j < n:
                tc = parse_table_caption(lines[j].strip())
                if tc:
                    caption, label = tc
                    i = j + 1
            rows: list[list[str]] = []
            for tl in table_lines:
                if is_table_sep(tl):
                    continue
                cells = [c.strip() for c in tl.strip().strip("|").split("|")]
                rows.append(cells)
            if rows:
                out.append(table_to_typst(rows, caption, label))
                out.append("")
            continue

        # blank
        if not line.strip():
            out.append("")
            i += 1
            continue

        # normal paragraph line (may continue)
        para = [line]
        i += 1
        while i < n and lines[i].strip() and not lines[i].startswith("#") and not lines[i].startswith("```") and not lines[i].startswith("![") and not is_table_row(lines[i]) and not lines[i].strip().startswith(":"):
            para.append(lines[i])
            i += 1
        text = " ".join(p.strip() for p in para)
        text = fix_inline(text)
        # citations already @key in md are valid typst
        # convert [@key] pandoc style to @key
        text = re.sub(r"\[@([a-zA-Z0-9_:-]+)\]", r"@\1", text)
        out.append(text)
        out.append("")

    # collapse excess blanks
    result = "\n".join(out)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip() + "\n"


def split_title_lines(title: str) -> str:
    """Break long title into two Typst lines if needed."""
    # Prefer natural break near "Using" / "of a"
    for needle in (" Using ", " of a "):
        if needle in title:
            a, b = title.split(needle, 1)
            return f"    {a}{needle.rstrip()}\\\n    {b}"
    words = title.split()
    if len(words) <= 8:
        return "    " + title
    mid = len(words) // 2
    return "    " + " ".join(words[:mid]) + "\\\n    " + " ".join(words[mid:])


def build_typst(meta: dict, body_typ: str) -> str:
    authors = meta.get("authors") or []
    authors_doc = ",\n    ".join(f'"{a}"' for a in authors)
    author_line = ",\n    ".join(f"{a}#super[1]" for a in authors)
    title = meta.get("title") or "Untitled"
    affiliation = meta.get("affiliation") or ""
    abstract = meta.get("abstract") or ""
    # Escape nothing much; abstract is plain prose
    head = TEMPLATE_HEAD.format(
        title=title.replace('"', '\\"'),
        authors=authors_doc,
        title_lines=split_title_lines(title),
        author_line="    " + author_line if author_line else "",
        affiliation=affiliation,
        abstract=abstract,
    )
    return head + body_typ + TEMPLATE_TAIL


def compile_pdf() -> None:
    if not MD_PATH.is_file():
        sys.exit(f"Missing {MD_PATH}")
    if not BIB_PATH.is_file():
        print(f"warning: {BIB_PATH.name} not found (bibliography will fail)", file=sys.stderr)

    text = MD_PATH.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    body_typ = body_md_to_typst(body)
    typ = build_typst(meta, body_typ)
    TYP_PATH.write_text(typ, encoding="utf-8")

    cmd = ["typst", "compile", str(TYP_PATH), str(PDF_PATH)]
    print("→", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr or r.stdout or "typst failed\n")
        sys.exit(r.returncode)
    if r.stderr:
        sys.stderr.write(r.stderr)
    print(f"✓ wrote {TYP_PATH.name} and {PDF_PATH.name}")


if __name__ == "__main__":
    compile_pdf()
