# A reading site for the atlas

A rendered site would help readers move from a worked example to a circuit and its evidence without navigating repository files.
It would give narrow screens more predictable typography and table handling, while keeping GitHub Markdown as a complete reading route.
This is a proposal for an owner decision. No site, hosting setting or build workflow has been enabled.

## Proposed reading experience

The home page would use the [README opening](README.md): a plain introduction, an explanatory picture, linked finding stories and
the [three-entry tour](README.md#start-here). A topic browser would use the [index](INDEX.md#browse-by-topic). Neither would lead
with a wall of evidence columns.

Each entry would remain a single article. Its task example, trained-versus-supplied parts, result and deciding limitation would
come before the detailed evidence. The contents navigation would link to results, mechanism tests, limitations and appendices.
The full evidence would stay on the page, with a local scroll area for wide tables rather than horizontal scrolling of the whole
page. A reader could follow the story or jump directly to a check. No limitation, failed seed, rejected formula, source or flag
would be removed or hidden behind an interaction.

Use a single reading column, system fonts, comfortable line spacing and a restrained blue/amber palette. Respect the reader's
light/dark preference. Figures would use direct labels, meaningful alternative text and an adjacent text equivalent. A static
text version would remain available wherever a diagram is shown. The glossary would remain one link away, with visible definitions
at first use rather than hover-only explanations.

## Concrete build proposal

Use GitHub Pages to host static HTML, generated from the existing Markdown with Jekyll and its GFM-compatible parser. A small local
layout and stylesheet would be enough; no downloaded theme, web font, analytics, third-party script or runtime CDN is needed.
GitHub documents both [Pages and Jekyll](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/about-github-pages-and-jekyll)
and [custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
(platform documentation). These describe a possible deployment, not an approval or an existing atlas service.

The proposed build would:

- Render the README as the home page, INDEX as the topic browser and each entry at a stable article URL. Keep glossary, methods,
  references, changelog, licence and citation pages accessible from the same navigation.
- Read each scientific statement directly from its Markdown source. Copy status and scope together; do not calculate new rankings,
  aggregate success rates or a live-looking count from dated external-zoo snapshots.
- Render Mermaid at build time with a pinned Mermaid 11 toolchain, producing local light/dark SVGs. Preserve captions and
  accessible descriptions, and keep the original diagrams in the repository. The [Mermaid CLI](https://github.com/mermaid-js/mermaid-cli)
  (software documentation) describes static SVG generation; it would not run in the reader's browser.
- Map repository-relative Markdown links to the generated article URLs, preserving existing heading anchors. Include a visible
  link back to the exact source revision on each page. Keep links to archived networks and original public sources unchanged.
- Ship only static HTML, CSS and local images. Loading a page must not fetch fonts, code, models or evidence from another service.
  Source and archive links would be followed only when the reader chooses them.

The Markdown would remain the sole editorial source. Generated HTML would not become a second set of articles to maintain.
The build would publish from an explicitly reviewed revision, not silently change scientific scope when a source archive changes.

## Conditions before publication

Check every generated article link and fragment, navigation order, image path, caption and source link. Compare numerical text,
failed-test records and qualifications against the Markdown. Check tables, code blocks and diagrams at phone width, enlarged text
and desktop width in both themes. Test keyboard navigation and a screen reader; automated checks and screenshots alone do not
establish accessibility conformance. Serve a local preview with external requests blocked to check that reading is self-contained.

Keep CC BY 4.0 attribution and the [licence](LICENSE), preserve third-party notices and make the [MIT code licence](code/LICENSE)
clear. Any optional search should use a locally built index with a complete static topic-browser alternative. Search is not needed
for the first site version.

## Decision requested

The owner would need to approve hosting a Pages site, its public URL, the Markdown-to-HTML build approach and who reviews deployments.
A reviewer could then prepare a non-published preview and audit the generated content before a separate enablement decision.
Until then, improve the [repository front page](README.md) and keep all reading routes functional there.
