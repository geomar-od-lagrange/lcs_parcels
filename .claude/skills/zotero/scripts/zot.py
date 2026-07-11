#!/usr/bin/env python3
"""Read-only CLI over the Zotero local API at http://localhost:23119/api.

Stdlib only. Subcommands: search, get, children, tags, collections, pdf, cite.
JSON to stdout, diagnostics to stderr, non-zero exit on error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE = os.environ.get("ZOTERO_API_URL", "http://localhost:23119/api")
DEFAULT_LIB = os.environ.get("ZOTERO_LIBRARY", "users/0")
PAGE = 100
LINK_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


def _url(base: str, lib: str, path: str, params: dict | None = None) -> str:
    u = f"{base}/{lib}{path}"
    if params:
        clean = {k: v for k, v in params.items() if v is not None}
        if clean:
            u += "?" + urllib.parse.urlencode(clean, doseq=True)
    return u


def _get(url: str) -> tuple[bytes, dict]:
    req = urllib.request.Request(url, headers={"Zotero-API-Version": "3"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read(), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace").strip()
        sys.stderr.write(f"Zotero API {e.code} on {url}\n  {msg}\n")
        sys.exit(1)
    except urllib.error.URLError as e:
        sys.stderr.write(
            f"Cannot reach Zotero at {url}: {e}\n"
            "Is Zotero 7 running with the local API enabled "
            "(Settings → Advanced → Allow other applications on this computer to communicate with Zotero)?\n"
        )
        sys.exit(2)


def _next_link(headers: dict) -> str | None:
    link = headers.get("link")
    if not link:
        return None
    m = LINK_RE.search(link)
    return m.group(1) if m else None


def _paged_json(url: str, hard_limit: int | None = None):
    out = []
    while url:
        body, h = _get(url)
        page = json.loads(body)
        out.extend(page)
        if hard_limit and len(out) >= hard_limit:
            return out[:hard_limit]
        url = _next_link(h)
    return out


def _slim_item(it: dict) -> dict:
    d = it.get("data", {})
    meta = it.get("meta", {})
    links = it.get("links", {})
    creators = [
        " ".join(p for p in (c.get("firstName"), c.get("lastName")) if p).strip()
        or c.get("name", "")
        for c in d.get("creators", [])
    ]
    year = ""
    pd = meta.get("parsedDate") or d.get("date") or ""
    m = re.search(r"\d{4}", pd)
    if m:
        year = m.group(0)
    out = {
        "key": d.get("key"),
        "itemType": d.get("itemType"),
        "title": d.get("title", ""),
        "creators": creators,
        "year": year,
        "publication": d.get("publicationTitle") or d.get("bookTitle") or d.get("repository") or "",
        "DOI": d.get("DOI", ""),
        "url": d.get("url", ""),
        "abstract": d.get("abstractNote", ""),
        "tags": [t.get("tag") for t in d.get("tags", [])],
        "collections": d.get("collections", []),
        "numChildren": meta.get("numChildren", 0),
    }
    att = links.get("attachment")
    if att:
        out["attachment"] = {
            "key": att["href"].rsplit("/", 1)[-1],
            "contentType": att.get("attachmentType", ""),
            "size": att.get("attachmentSize", 0),
        }
    return out


def _resolve_keys(base: str, lib: str, keys: list[str], top_only: bool = False) -> list[dict]:
    if not keys:
        return []
    out = []
    for i in range(0, len(keys), 50):
        chunk = keys[i : i + 50]
        path = "/items/top" if top_only else "/items"
        url = _url(base, lib, path, {"itemKey": ",".join(chunk), "limit": PAGE})
        body, _ = _get(url)
        out.extend(json.loads(body))
    by_key = {it["data"]["key"]: it for it in out if "data" in it}
    return [by_key[k] for k in keys if k in by_key]


def cmd_search(args):
    params = {
        "q": args.query,
        "qmode": args.mode,
        "limit": PAGE,
        "format": "keys",
    }
    if args.tag:
        params["tag"] = args.tag
    if args.type:
        params["itemType"] = args.type
    path = "/items/top"
    if args.collection:
        path = f"/collections/{args.collection}/items/top"
    body, _ = _get(_url(args.base, args.lib, path, params))
    keys = [k for k in body.decode().split() if k]
    if args.limit:
        keys = keys[: args.limit]
    items = _resolve_keys(args.base, args.lib, keys, top_only=True)
    if args.full:
        json.dump(items, sys.stdout, indent=2, ensure_ascii=False)
    else:
        json.dump([_slim_item(it) for it in items], sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stderr.write(f"# {len(items)} hit(s) for q={args.query!r} qmode={args.mode}\n")


def cmd_get(args):
    params = {}
    includes = ["data"]
    if args.bib:
        includes.append("bib")
        params["style"] = args.style
    params["include"] = ",".join(includes)
    body, _ = _get(_url(args.base, args.lib, f"/items/{args.key}", params))
    obj = json.loads(body)
    if args.full:
        json.dump(obj, sys.stdout, indent=2, ensure_ascii=False)
    else:
        slim = _slim_item(obj)
        if "bib" in obj:
            slim["bib_html"] = obj["bib"]
        json.dump(slim, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def cmd_children(args):
    body, _ = _get(_url(args.base, args.lib, f"/items/{args.key}/children", {"limit": PAGE}))
    items = json.loads(body)
    out = []
    for it in items:
        d = it.get("data", {})
        links = it.get("links", {})
        rec = {
            "key": d.get("key"),
            "itemType": d.get("itemType"),
            "title": d.get("title", ""),
            "contentType": d.get("contentType", ""),
            "filename": d.get("filename", ""),
            "linkMode": d.get("linkMode", ""),
            "note_html": d.get("note", "") if d.get("itemType") == "note" else None,
        }
        enc = links.get("enclosure")
        if enc:
            rec["path"] = _file_url_to_path(enc["href"])
        out.append(rec)
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def _file_url_to_path(href: str) -> str:
    if href.startswith("file://"):
        return urllib.parse.unquote(href[len("file://") :])
    return href


def cmd_pdf(args):
    body, _ = _get(_url(args.base, args.lib, f"/items/{args.key}"))
    item = json.loads(body)
    d = item.get("data", {})
    if d.get("itemType") == "attachment":
        href = item.get("links", {}).get("enclosure", {}).get("href")
        if not href:
            sys.stderr.write(f"Attachment {args.key} has no local file (linkMode={d.get('linkMode')}).\n")
            sys.exit(3)
        print(_file_url_to_path(href))
        return
    body, _ = _get(_url(args.base, args.lib, f"/items/{args.key}/children"))
    children = json.loads(body)
    pdfs = [
        c for c in children
        if c["data"].get("itemType") == "attachment"
        and (c["data"].get("contentType") == "application/pdf" or c["data"].get("filename", "").lower().endswith(".pdf"))
    ]
    if not pdfs:
        sys.stderr.write(f"No PDF attachment found under item {args.key}.\n")
        sys.exit(3)
    if args.all:
        for c in pdfs:
            href = c.get("links", {}).get("enclosure", {}).get("href", "")
            if href:
                print(_file_url_to_path(href))
        return
    href = pdfs[0].get("links", {}).get("enclosure", {}).get("href")
    if not href:
        sys.stderr.write(f"PDF child {pdfs[0]['data']['key']} has no local file (linkMode={pdfs[0]['data'].get('linkMode')}).\n")
        sys.exit(3)
    print(_file_url_to_path(href))


def cmd_tags(args):
    items = _paged_json(_url(args.base, args.lib, "/tags", {"limit": PAGE}))
    rows = [(t.get("meta", {}).get("numItems", 0), t["tag"]) for t in items]
    if args.prefix:
        rows = [r for r in rows if r[1].lower().startswith(args.prefix.lower())]
    if args.contains:
        rows = [r for r in rows if args.contains.lower() in r[1].lower()]
    if args.min_count:
        rows = [r for r in rows if r[0] >= args.min_count]
    rows.sort(key=lambda r: (-r[0], r[1].lower()))
    for n, t in rows:
        print(f"{n}\t{t}")
    sys.stderr.write(f"# {len(rows)} tag(s)\n")


def cmd_collections(args):
    items = _paged_json(_url(args.base, args.lib, "/collections", {"limit": PAGE}))
    rows = []
    for c in items:
        d = c["data"]
        rows.append((c["meta"].get("numItems", 0), d["key"], d["name"], d.get("parentCollection") or ""))
    rows.sort(key=lambda r: (-r[0], r[2].lower()))
    for n, k, name, parent in rows:
        print(f"{n}\t{k}\t{name}\t{parent}")
    sys.stderr.write(f"# {len(rows)} collection(s)\n")


def cmd_cite(args):
    keys = args.keys
    if args.format in {"bibtex", "biblatex", "csljson", "ris"}:
        params = {"itemKey": ",".join(keys), "format": args.format, "limit": PAGE}
        body, _ = _get(_url(args.base, args.lib, "/items", params))
        sys.stdout.buffer.write(body)
        if not body.endswith(b"\n"):
            sys.stdout.write("\n")
        return
    params = {"itemKey": ",".join(keys), "include": "bib", "style": args.style, "limit": PAGE}
    body, _ = _get(_url(args.base, args.lib, "/items", params))
    items = json.loads(body)
    for it in items:
        bib = it.get("bib", "").strip()
        if bib:
            print(bib + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zot", description=__doc__.splitlines()[0])
    p.add_argument("--base", default=DEFAULT_BASE, help=f"Zotero API base URL (default: {DEFAULT_BASE})")
    p.add_argument("--lib", default=DEFAULT_LIB, help=f"library prefix (default: {DEFAULT_LIB})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="search the library; returns slim JSON list")
    s.add_argument("query")
    s.add_argument("--mode", choices=["everything", "titleCreatorYear"], default="everything",
                   help="everything = full-text incl. PDFs (default); titleCreatorYear = metadata only")
    s.add_argument("--limit", type=int, default=50, help="max results (default 50, pass 0 for all)")
    s.add_argument("--collection", help="restrict to a collection key")
    s.add_argument("--tag", action="append", help="restrict to a tag (repeatable)")
    s.add_argument("--type", help="restrict to itemType (e.g. journalArticle, preprint, book)")
    s.add_argument("--full", action="store_true", help="emit full Zotero JSON instead of slim")
    s.set_defaults(func=cmd_search)

    g = sub.add_parser("get", help="fetch one item by key")
    g.add_argument("key")
    g.add_argument("--full", action="store_true", help="full Zotero JSON")
    g.add_argument("--bib", action="store_true", help="include formatted bibliography HTML")
    g.add_argument("--style", default="apa", help="CSL style (default apa)")
    g.set_defaults(func=cmd_get)

    c = sub.add_parser("children", help="list child notes/attachments of an item")
    c.add_argument("key")
    c.set_defaults(func=cmd_children)

    t = sub.add_parser("tags", help="list tags as TSV: count<TAB>tag, sorted by count desc")
    t.add_argument("--prefix")
    t.add_argument("--contains")
    t.add_argument("--min-count", type=int)
    t.set_defaults(func=cmd_tags)

    co = sub.add_parser("collections", help="list collections as TSV: count<TAB>key<TAB>name<TAB>parent")
    co.set_defaults(func=cmd_collections)

    pdf = sub.add_parser("pdf", help="print local file path of the PDF for an item")
    pdf.add_argument("key", help="parent item key OR attachment key")
    pdf.add_argument("--all", action="store_true", help="print every PDF under the parent")
    pdf.set_defaults(func=cmd_pdf)

    ci = sub.add_parser("cite", help="emit citations for one or more keys")
    ci.add_argument("keys", nargs="+")
    ci.add_argument("--format", choices=["bibtex", "biblatex", "csljson", "ris", "bib"], default="bibtex")
    ci.add_argument("--style", default="apa", help="CSL style for --format bib (default apa)")
    ci.set_defaults(func=cmd_cite)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "limit", None) == 0:
        args.limit = None
    args.func(args)


if __name__ == "__main__":
    main()
