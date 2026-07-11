# Zotero local API cheat sheet

This is a quick reference for direct `curl` work against `http://localhost:23119/api`. The bundled `scripts/zot.py` covers the common path; reach for this file when you need a parameter the script doesn't expose.

The local API mirrors the [Zotero web API v3](https://www.zotero.org/support/dev/web_api/v3/basics) but with a few caveats (read-only, no Atom — see Gotchas in `SKILL.md`).

## Base URLs

| Resource | Path |
|---|---|
| Local user library | `/users/0/...` (resolves to the actual numeric user id) |
| Group library | `/groups/<groupID>/...` |

The host is always `http://localhost:23119/api`. No auth required.

## Items

| Want | URL |
|---|---|
| All items | `GET /users/0/items` |
| Top-level items only | `GET /users/0/items/top` |
| One item | `GET /users/0/items/<key>` |
| Children of an item | `GET /users/0/items/<key>/children` |
| Items in collection | `GET /users/0/collections/<collKey>/items` (or `.../items/top`) |
| Items with a tag | `GET /users/0/items?tag=<urlencoded>` |
| Trashed items | `GET /users/0/items/trash` |
| File for attachment | `GET /users/0/items/<attachmentKey>/file` → 302 redirect to a `file://` URL on local disk; `links.enclosure.href` on the attachment item gives the same path without an extra request |

### Filters & search params

| Param | Notes |
|---|---|
| `q` | Search string. Boolean syntax: `term1 term2` (AND), `term1 \|\| term2` (OR), `-term3` (NOT) |
| `qmode` | `titleCreatorYear` (default) or `everything` (full text including indexed PDFs) |
| `itemType` | e.g. `journalArticle`, `preprint`, `book`, `bookSection`, `report`, `thesis`, `webpage`. Combine with `\|\|` and `-` |
| `tag` | Repeatable for AND; use `\|\|` inside one value for OR; `-` prefix to exclude |
| `itemKey` | Comma-separated, max 50 keys per request — efficient bulk fetch |
| `since` | Library version number; returns objects modified since that version (incremental sync) |
| `sort` | `dateModified` (default), `dateAdded`, `title`, `creator`, `itemType`, `date`, `publisher`, `journalAbbreviation` |
| `direction` | `asc` or `desc` |
| `start`, `limit` | Pagination; `limit` max 100. Response carries `Total-Results` header and `Link: rel="next"` |

### `format=` and `include=`

`format` controls the response shape; `include` adds fields to the default JSON.

| `format=` | What you get |
|---|---|
| `json` (default) | Full Zotero item JSON |
| `keys` | Newline-separated item keys, plain text — cheapest way to enumerate |
| `versions` | `{ "<key>": <version>, ... }` — pair with `since=` for sync |
| `bibtex` / `biblatex` | LaTeX-friendly bibliography |
| `csljson` | CSL-JSON for pandoc / Citeproc pipelines |
| `ris`, `mods`, `refer`, `rdf_zotero`, `rdf_bibliontology`, `rdf_dc` | Other export formats |
| `atom` | **Not supported on local API (501).** |

| `include=` | Adds |
|---|---|
| `data` | Full item data (default for `json` already includes this) |
| `bib` | HTML-formatted citation block. Style via `style=<csl-name>`, locale via `locale=<bcp47>` |
| `citation` | Inline citation (e.g. "(Smith, 2017)") — same `style`/`locale` |
| Multiple | Comma-separate: `include=data,bib,citation` |

`style=` accepts any style installed in Zotero (`apa`, `nature`, `chicago-author-date`, `ieee`, ...) or a remote CSL URL. Default is `chicago-note-bibliography`.

## Collections

| URL | Returns |
|---|---|
| `GET /users/0/collections` | All collections (flat) |
| `GET /users/0/collections/top` | Root-level collections only |
| `GET /users/0/collections/<key>` | One collection |
| `GET /users/0/collections/<key>/collections` | Sub-collections |
| `GET /users/0/collections/<key>/items` | Items in a collection (any level) |
| `GET /users/0/collections/<key>/items/top` | Top-level items in a collection |
| `GET /users/0/collections/<key>/tags` | Tags used inside this collection |

Each collection's `data.parentCollection` is `false` for top-level or another collection key.

## Tags

| URL | Notes |
|---|---|
| `GET /users/0/tags` | All tags with `meta.numItems` per tag |
| `GET /users/0/tags?q=<prefix>` | Filter by tag-name match |
| `GET /users/0/tags/<urlencoded-tag>` | One tag |
| `GET /users/0/items/<key>/tags` | Tags on a specific item |

Tag types: `0` = manual, `1` = automatic (added by import). Surfaced as `meta.type`.

## Saved searches

`GET /users/0/searches[/{key}]` — list and retrieve. They live in the library but only return their **definitions** (conditions); to execute one, replicate its conditions as `q` / `tag` / `itemType` / etc. parameters yourself.

## Global metadata (no library prefix)

These describe the Zotero schema itself; cache aggressively (the schema rarely changes).

| URL | Notes |
|---|---|
| `GET /api/itemTypes` | All item types, optional `?locale=de-DE` etc. |
| `GET /api/itemFields` | All possible fields |
| `GET /api/itemTypeFields?itemType=preprint` | Fields valid for one type |
| `GET /api/itemTypeCreatorTypes?itemType=book` | Allowed creator roles for one type |
| `GET /api/creatorFields` | Field names on a creator object |
| `GET /api/items/new?itemType=journalArticle` | Empty template for creating an item (ignored on the local API since writes are blocked, but useful as a schema reference) |

## Headers worth knowing

- `Total-Results` — total count for paged endpoints, even when `limit` truncates the page.
- `Link` — pagination, with `rel="next"`, `rel="last"`, `rel="alternate"` (web URL).
- `Last-Modified-Version` — the library version at the time of the response. Save it; pass it back as `If-Modified-Since-Version: <n>` to get `304 Not Modified` if nothing changed.
- `Zotero-Schema-Version` — schema number (currently 40 on Zotero 7).

## Examples

```bash
# 50 most-recently-added preprints, just keys
curl -s 'http://localhost:23119/api/users/0/items/top?itemType=preprint&sort=dateAdded&direction=desc&limit=50&format=keys'

# CSL-JSON for a handful of items
curl -s 'http://localhost:23119/api/users/0/items?itemKey=K1,K2,K3&format=csljson'

# All items modified since library version 1500 (incremental sync)
curl -s 'http://localhost:23119/api/users/0/items?since=1500&format=versions'

# Items tagged "Stokes drift" AND "Coastal upwelling"
curl -s --get 'http://localhost:23119/api/users/0/items' \
  --data-urlencode 'tag=Stokes drift' --data-urlencode 'tag=Coastal upwelling' \
  --data-urlencode 'format=keys'

# Boolean search: stokes drift OR wave-induced, excluding "tutorial"
curl -s --get 'http://localhost:23119/api/users/0/items' \
  --data-urlencode 'q=stokes drift || wave-induced -tutorial' \
  --data-urlencode 'qmode=everything' --data-urlencode 'format=keys'
```
