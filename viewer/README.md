# Autoresearch viewer

This is the local, project-agnostic viewer for an autoresearch workspace. The graph is navigation and lineage only. Selecting an experiment renders that experiment's authored `experiment.md`; the viewer does not synthesize a second result narrative or infer artifacts from run directories.

Run it from this directory with `pnpm install` and `pnpm dev`, then open `http://127.0.0.1:4173`. By default it reads the repository one level above this directory. To point the same viewer at another autoresearch project, set `AUTORESEARCH_PROJECT_ROOT` to that project's root before starting it.

The source watches `research/experiments` and `research/framings`. Changes to experiment metadata, Markdown, figures, and linked outputs appear without a page refresh while the current selection is preserved.

Only files explicitly linked or embedded in `experiment.md` can be requested through the artifact endpoint. Frontmatter, filesystem paths, unlinked run files, raw logs, hashes, and working directories are not added to the interface by the viewer. Authors control the public research record by controlling the Markdown document and its links.

Linked image outputs open in the viewer's figure modal, including image links inside tables. Embedded figures use the same modal. Other linked outputs reveal the file in Finder on macOS or Explorer on Windows; Linux opens the containing folder. Both actions use the same explicit-link and path-safety checks as the artifact endpoint.

Image URLs are generated from the experiment ID and the repository-relative reference in `experiment.md`. They use the viewer's same-origin artifact endpoint, so the client contains no hostname, localhost URL, absolute project root, or collaborator-specific filesystem path.

Use `pnpm test` for the source and path-safety tests and `pnpm build` for the TypeScript and client build check. The local live reader is provided by `pnpm dev`; `pnpm preview` only previews the compiled static client and does not provide the project-reading endpoints.
