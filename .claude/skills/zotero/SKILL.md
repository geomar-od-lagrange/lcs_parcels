---
name: zotero
description: Search, browse, and read citations from the user's local Zotero 7 library via its built-in HTTP API at http://localhost:23119. Use whenever the user asks about their papers, references, citations, bibliography, library, tags, collections, or PDFs they've saved — including phrasings like "find papers about X", "list my tags", "what's in collection Y", "give me the BibTeX for Z", "open the PDF of paper W", "what does my library say about topic X", or follow-ups that require scanning titles, abstracts, or PDF full text in their reference manager. Returns structured metadata, formatted citations (BibTeX, BibLaTeX, CSL-JSON, RIS, or any CSL style), and local filesystem paths to bundled PDFs that the agent can then open and read directly.
compatibility: Requires Zotero 7 running locally with the API enabled (Settings → Advanced → "Allow other applications on this computer to communicate with Zotero"). Needs python3 (stdlib only) on PATH.
---

# Zotero

A read-only window into the user's local Zotero library. The Zotero desktop client exposes the same JSON API as `api.zotero.org` on `http://localhost:23119/api`; this skill wraps the most useful parts in a single Python CLI so you don't have to hand-craft URLs.

## Prerequisites — sanity check first

Before doing anything else, confirm the API is up:

```bash
curl -sf http://localhost:23119/api/ >/dev/null && echo OK || echo "Zotero API not reachable"
```

If the check fails, **stop and tell the user**. Do not pretend to query the library, do not guess at item keys, do not invent results. Relay this checklist verbatim:

1. **Is Zotero 7 running?** The local API is only available while the Zotero desktop app is open. Launch it and wait for the library to finish loading.
2. **Is the local API enabled?** Go to *Settings → Advanced* and tick **"Allow other applications on this computer to communicate with Zotero"**. Restart Zotero after toggling.
3. **Is it Zotero 7?** Zotero 6 and earlier do not expose this API. *Help → About Zotero* shows the version.
4. **Is port 23119 reachable on localhost?** If the first three are OK, another process may have bound the port or a firewall may be blocking it. The user can verify with `lsof -iTCP:23119 -sTCP:LISTEN` (macOS / Linux) or `netstat -ano | findstr 23119` (Windows) — Zotero should be the listener.

After the user reports the issue is fixed, re-run the `curl` check before proceeding. If it still fails, surface the user's report and ask for next steps rather than retrying silently.

## The data model in 30 seconds

- An **item** is a paper / book / preprint / etc. Each has an 8-character key (e.g. `74S59DZC`).
- An item can have **child items**: `attachment` (PDFs, snapshots) and `note`. PDFs sit on local disk under `~/Zotero/storage/<attachmentKey>/...`.
- **Collections** are folders. Items can live in 0+ collections.
- **Tags** are flat strings attached to items; the API surfaces them with per-tag item counts.
- The library prefix is `users/0` for the local user (default). Group libraries would be `groups/<id>` — pass `--lib groups/123` to the script.

## The `zot` script

All operations go through `scripts/zot.py`. It speaks to `http://localhost:23119/api/users/0` by default, paginates automatically, and emits JSON to stdout / TSV for list commands. Diagnostics go to stderr so output is pipe-safe.

```bash
python3 scripts/zot.py --help                       # all subcommands
python3 scripts/zot.py <subcommand> --help          # per-subcommand flags
```

Subcommands:

- **`search QUERY`** — full-text search by default (`--mode everything` covers titles, abstracts, AND indexed PDF contents). Returns a slim JSON list per hit: `key, itemType, title, creators, year, publication, DOI, url, abstract, tags, collections, numChildren, attachment{key,contentType,size}`. Flags: `--mode {everything,titleCreatorYear}`, `--limit N` (default 50; `0` = all), `--collection KEY`, `--tag T` (repeatable, AND), `--type TYPE`, `--full` (raw Zotero JSON).
- **`get KEY`** — fetch one item. `--bib --style apa` adds an HTML-formatted citation. `--full` for raw JSON.
- **`children KEY`** — list child notes/attachments of an item. Each attachment record includes `path` (the local file path, ready to open directly).
- **`tags`** — TSV `count<TAB>tag`, sorted by count desc. Optional `--prefix`, `--contains`, `--min-count N`.
- **`collections`** — TSV `count<TAB>key<TAB>name<TAB>parent`.
- **`pdf KEY`** — print the local filesystem path of the PDF for `KEY`. `KEY` may be the parent item (resolves to its first PDF child) or an attachment key directly. `--all` prints every PDF under the parent (one per line).
- **`cite KEY [KEY...]`** — emit citations. `--format {bibtex,biblatex,csljson,ris,bib}`. With `--format bib`, pass `--style apa|nature|chicago-author-date|...` (any CSL style installed in Zotero).

Override the endpoint or library with `--base URL` / `--lib users/N|groups/N`, or env vars `ZOTERO_API_URL` / `ZOTERO_LIBRARY`.

## Recipes

These match common questions. Adapt as needed.

### List all tags in the library

```bash
python3 scripts/zot.py tags                    # all tags, sorted by count
python3 scripts/zot.py tags --min-count 5      # only tags with ≥5 papers
python3 scripts/zot.py tags --contains ocean   # case-insensitive substring filter
```

### Find papers about a topic

`--mode everything` searches indexed PDF text in addition to metadata, so it picks up papers that don't mention the term in the title:

```bash
python3 scripts/zot.py search "stokes drift" --limit 20
```

For "papers I read on a topic" prefer `everything`. For "papers by author Schmidt about turbulence" use `--mode titleCreatorYear` and combine fields:

```bash
python3 scripts/zot.py search "Schmidt turbulence" --mode titleCreatorYear
```

### Triage a topic by titles + abstracts

Run a search at a generous `--limit`, read the JSON, and decide which items deserve a full read. The slim output already contains `title` and `abstract` so you can scan without a second roundtrip. For ambiguous hits, follow up by reading the PDF (next recipe). When the topic spans synonyms (e.g. "genetic algorithm" / "evolutionary optimisation" / "evolution strategies"), run a few searches and merge by `key`.

### Read a paper's PDF

```bash
PDF=$(python3 scripts/zot.py pdf 7UFI6E7S)
# then open "$PDF" with whichever file-reading capability the host agent has
```

Open the printed path directly. Zotero stores PDFs locally under `~/Zotero/storage/<attachmentKey>/`, so reads are instant. If `pdf` exits with "No PDF attachment found", the item has no PDF (e.g. a webpage snapshot or note); fall back to the item's `url` field.

### Generate a bibliography

```bash
python3 scripts/zot.py cite 74S59DZC 7UFI6E7S --format bibtex      # for LaTeX
python3 scripts/zot.py cite 74S59DZC --format csljson              # for pandoc/CSL pipelines
python3 scripts/zot.py cite 74S59DZC --format bib --style nature   # rendered HTML in Nature style
```

`--format bib` returns CSL-rendered HTML; strip tags if you want plain text.

### Worked example — extract a result from a topic's PDFs

End-to-end pattern for "find papers about X and pull a specific result Y from them":

1. `python3 scripts/zot.py search "X" --limit 30 --mode everything` — get candidates.
2. Scan the returned `title` + `abstract` fields. Drop irrelevant hits.
3. For each remaining candidate: `python3 scripts/zot.py pdf <key>` → open the file → look for Y.
4. Cite the papers that contributed: `python3 scripts/zot.py cite K1 K2 K3 --format bibtex`.

Step 3 is where most of the agent's work happens — the skill's job is to surface metadata and PDF paths cheaply so you can spend context on the actual reading.

## Gotchas

- **Read-only.** The local API rejects POST/PUT/PATCH/DELETE (`400 Endpoint does not support method` / `501 Method not implemented`). To add items, edit metadata, or change tags, ask the user to do it in the Zotero UI. Don't pretend to write.
- **Duplicates exist.** A real library often contains the same paper imported twice with different keys (different DOIs, different import sources). Don't dedupe silently — surface both, and if it matters, mention the duplication.
- **Full-text only covers indexed PDFs.** Items with no PDF or whose PDF Zotero hasn't indexed yet won't appear in `--mode everything` matches. If a search comes up empty and you suspect that, retry with `--mode titleCreatorYear` and broader terms.
- **Zsh and `?` in URLs.** When you write curl commands directly (rather than through the script), single-quote the URL or zsh's globbing will eat the query string: `curl 'http://localhost:23119/api/users/0/items?limit=1'`.
- **Atom is unsupported.** The local API only returns JSON / keys / versions / export formats (`bibtex`, `biblatex`, `csljson`, `ris`, `mods`, `refer`, `rdf_*`). `format=atom` returns 501.
- **The script's `pdf` subcommand may print a path with spaces.** Always capture it into a quoted variable before opening, e.g. `PDF="$(python3 scripts/zot.py pdf KEY)"`.

## When the basics aren't enough

Reach for `references/api-cheatsheet.md` when the user wants something the subcommands don't cover directly: incremental sync via `since=` and `format=versions`, item-type/field metadata, boolean search syntax (`||`, `-`), or sort/direction options. The cheat sheet maps the official Zotero web API v3 conventions onto the local endpoint.
