# Paper manuscript (Markdown → Typst → PDF)

**Edit `paper.md`.** Figures live in `figures/`. Bibliography is `refs.bib`.

```bash
cd paper
python build.py          # one-shot: paper.md → paper.typ → paper.pdf
python watch.py          # recompile on every save of paper.md / figures / refs.bib
```

Requires [Typst](https://typst.app/) (`brew install typst`). Uses system Python 3 (no extra packages).

`paper.typ` is **generated** by `build.py` — do not hand-edit it. Layout (spacing, fonts, margins) is controlled in `build.py`.

## Markdown conventions

| What | How to write it in `paper.md` |
|------|-------------------------------|
| Section | `# Introduction` / `## Subsection` |
| Figure | `![Caption text.](figures/foo.jpg){#fig:id width=72%}` |
| Table | Markdown table, then `: Caption. {#tab:id}` |
| Citation | `@smith-2020` (or `[@smith-2020]`) — keys from `refs.bib` |
| Display math | Fenced ` ```typst-math eq:rate ` block with Typst math inside |
| Inline math | `$D$`, `$P_"ann"$` (Typst math) |
| Superscript | `yr^-1^` → yr⁻¹ in PDF |
| Raw Typst | Fenced ` ```typst ` block (passthrough) |

Front matter at the top of `paper.md` sets title, authors, affiliation, and abstract (YAML).

## Figures

Place JPGs/PNGs under `figures/` and reference them from markdown as above. Width is a percentage of the text column (typical: maps 72%, scatter 65%, emissions 58%).
