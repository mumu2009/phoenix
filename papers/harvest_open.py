"""Enumerate 2018-01-01 .. 2026-10-24 works whose venue we can name.

CCF A/B/C venues are resolved to an OpenAlex source id, then that
source's works are paged. A source is accepted only when its type
matches the catalogue and the distinctive-token Jaccard is at least
0.75. Workshop, forum, demo, and student venues are not substituted
for the main series.

IEEE and ACL publisher totals come from Crossref member ids 263 and
1643. ACL works are paged from that member. The IEEE window is about
three million works; this script records the count and does not
download that dump. SCI and EI master lists are not a public API.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLLECTED = ROOT / "collected"
CCF_PATH = ROOT / "ccf_2022.json"
DATE_FROM = "2018-01-01"
DATE_TO = "2026-10-24"
MAILTO = "19559858@qq.com"
UA = "phoenix-papers/1.0 (mailto:19559858@qq.com)"
CROSSREF_FILTER = f"from-pub-date:{DATE_FROM},until-pub-date:{DATE_TO}"
OPENALEX_WINDOW = (
    f"from_publication_date:{DATE_FROM},to_publication_date:{DATE_TO},"
    "is_paratext:false"
)
SELECT = "id,doi,display_name,publication_year,primary_location"
JACCARD_MIN = 0.75
# OpenAlex page= is capped at 10,000 works. Cursor pages returned
# 401 without an API key, so this crawl uses page numbers.
PER_PAGE = 200
MAX_PAGES = 50
STOP = {
    "the", "of", "and", "on", "for", "a", "an", "in", "to", "with",
    "from", "by", "at", "proceedings", "conference", "international",
    "annual", "symposium", "acm", "ieee", "society", "association",
    "meeting",
}
SIDE_VENUE = ("workshop", "forum", "demonstration", "student", "companion")
CHECKS = (
    ("TOCS", "ACM Transactions on Computer Systems", "Journal"),
    ("CVPR", "IEEE/CVF Computer Vision and Pattern Recognition Conference", "Conference"),
    ("ICML", "International Conference on Machine Learning", "Conference"),
    ("ACL", "Annual Meeting of the Association for Computational Linguistics", "Conference"),
    ("NeurIPS", "Conference on Neural Information Processing Systems", "Conference"),
    ("SIGIR", "International ACM SIGIR Conference on Research and Development in Information Retrieval", "Conference"),
)


def get(url: str) -> dict:
    last = None
    for attempt in range(8):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            text = str(exc)
            if "429" in text or "401" in text or "503" in text:
                time.sleep(min(120, 20 * (attempt + 1)))
            else:
                time.sleep(min(30, 2 ** attempt))
    raise last


def tokens(text: str) -> set[str]:
    cleaned = (text or "").replace("&amp;", " ").replace("&", " ")
    cleaned = re.sub(r"[()]", " ", cleaned).lower()
    noise = {"volume", "vol", "pp", "pages", "issue", "part"}
    out = set()
    for tok in re.findall(r"[a-z0-9]+", cleaned):
        if tok in STOP or tok in noise or tok.isdigit():
            continue
        if re.fullmatch(r"\d+[a-z]?", tok):
            continue
        if re.fullmatch(r"\d+(st|nd|rd|th)", tok):
            continue
        if re.fullmatch(r"(19|20)\d{2}", tok):
            continue
        out.add(tok)
    return out


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def side_venue(name: str) -> bool:
    blob = (name or "").lower()
    return any(word in blob for word in SIDE_VENUE)


def source_score(ccf_name: str, ccf_type: str, source: dict) -> float:
    display = source.get("display_name") or ""
    kind = (source.get("type") or "").lower()
    want = "journal" if ccf_type == "Journal" else "conference"
    if want not in kind:
        return 0.0
    if side_venue(display) and not side_venue(ccf_name):
        return 0.0
    return jaccard(tokens(ccf_name), tokens(display))


def oa(url: str) -> str:
    join = "&" if "?" in url else "?"
    return url + join + "mailto=" + urllib.parse.quote(MAILTO)


def search_sources(query: str) -> list[dict]:
    url = oa(
        "https://api.openalex.org/sources?search="
        + urllib.parse.quote(query)
        + "&per-page=10&select=id,display_name,type,works_count"
    )
    data = get(url)
    time.sleep(0.4)
    return data.get("results") or []


def shortened_name(name: str) -> str:
    lead = {
        "conference", "international", "annual", "proceedings",
        "symposium", "meeting", "the", "on", "of", "for",
    }
    words = name.split()
    index = 0
    while index < len(words) and words[index].lower().strip(".,/") in lead:
        index += 1
    short = " ".join(words[index:]).strip()
    if len(short) < 12 or short.lower() == name.lower():
        return ""
    return short


def choose_source(name: str, ccf_type: str, abbr: str) -> tuple[dict | None, float]:
    found: dict[str, dict] = {}
    queries = [name]
    short = shortened_name(name)
    if short:
        queries.append(short)
    if len(abbr) >= 4 and abbr.isalpha():
        queries.append(abbr)
    for query in queries:
        for source in search_sources(query):
            found[source.get("id") or ""] = source
        best = None
        best_score = 0.0
        for source in found.values():
            score = source_score(name, ccf_type, source)
            if score > best_score:
                best = source
                best_score = score
        if best is not None and best_score >= JACCARD_MIN:
            return best, best_score
    best = None
    best_score = 0.0
    for source in found.values():
        score = source_score(name, ccf_type, source)
        if score > best_score:
            best = source
            best_score = score
    return None, best_score


def source_id(source: dict) -> str:
    return (source.get("id") or "").rsplit("/", 1)[-1]


def marks(venue: str, publisher: str, abbr: str) -> list[str]:
    blob = f"{venue} {publisher}".lower()
    out = []
    if "ieee" in blob:
        out.append("ieee")
    if "computational linguistics" in blob or abbr.lower() == "acl":
        out.append("acl")
    return out


def slim_openalex(work: dict, extra: dict) -> dict:
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    venue = source.get("display_name") or ""
    publisher = source.get("host_organization_name") or ""
    rank = extra.get("ccf_rank")
    tagged = marks(venue, publisher, extra.get("ccf_abbr") or "")
    if rank:
        tagged.insert(0, "ccf-" + rank)
    doi = work.get("doi") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/") :]
    return {
        "doi": doi,
        "openalex_id": work.get("id"),
        "title": work.get("display_name") or "",
        "year": work.get("publication_year"),
        "venue": venue,
        "publisher": publisher,
        "marks": tagged,
        **extra,
    }


def page_source(sid: str):
    reported = None
    for page_i in range(MAX_PAGES):
        url = oa(
            "https://api.openalex.org/works?filter=primary_location.source.id:"
            + sid
            + ","
            + OPENALEX_WINDOW
            + "&per-page="
            + str(PER_PAGE)
            + "&page="
            + str(page_i + 1)
            + "&select="
            + SELECT
        )
        data = get(url)
        meta = data.get("meta") or {}
        if reported is None:
            reported = meta.get("count")
        rows = data.get("results") or []
        yield rows, reported, page_i + 1
        if not rows:
            break
        if reported is not None and (page_i + 1) * PER_PAGE >= int(reported):
            break
        time.sleep(0.4)
    return reported


def check_sources() -> bool:
    ok = True
    for abbr, name, kind in CHECKS:
        source, score = choose_source(name, kind, abbr)
        label = source.get("display_name") if source else None
        print(f"check {abbr} score {score:.3f} -> {label}", flush=True)
        if source is None:
            ok = False
    return ok


def finished_abbrs() -> set[str]:
    log = COLLECTED / "ccf_venue_log.txt"
    done = set()
    if not log.exists():
        return done
    for line in log.read_text(encoding="utf-8").splitlines():
        if "incomplete" in line or "truncated" in line:
            continue
        if line.startswith("ccf\t"):
            cols = line.split("\t")
            if len(cols) >= 2:
                done.add(cols[1])
        elif line.startswith("ccf "):
            parts = line.split()
            if len(parts) >= 2:
                done.add(parts[1])
    return done


def harvest_ccf() -> None:
    catalogue = json.loads(CCF_PATH.read_text(encoding="utf-8"))
    path = COLLECTED / "ccf_works.jsonl"
    misses = COLLECTED / "ccf_misses.txt"
    log = COLLECTED / "ccf_venue_log.txt"
    done = finished_abbrs()
    seen = set()
    if done and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                doi = json.loads(line).get("doi")
            except json.JSONDecodeError:
                continue
            if doi:
                seen.add(doi)
    print(f"ccf resume skip {len(done)}", flush=True)
    mode = "a" if done else "w"
    with path.open(mode, encoding="utf-8") as handle, misses.open(
        mode, encoding="utf-8"
    ) as miss, log.open(mode, encoding="utf-8") as venue_log:
        for item in catalogue["list"]:
            if item.get("rank") not in {"A", "B", "C"}:
                continue
            name = (item.get("name") or "").split("(")[0].strip()
            abbr = item.get("abbr") or ""
            if abbr in done:
                continue
            try:
                source, score = choose_source(name, item.get("type") or "", abbr)
            except Exception as exc:
                miss.write(f"{abbr}\t{name}\tlookup {exc}\n")
                miss.flush()
                print("fail", abbr, exc, flush=True)
                continue
            if source is None:
                miss.write(f"{abbr}\t{name}\tno source score {score:.3f}\n")
                miss.flush()
                venue_log.write(
                    "\t".join(
                        [
                            "ccf",
                            abbr,
                            str(item.get("rank") or ""),
                            "0",
                            "0",
                            f"{score:.3f}",
                            "",
                            "miss",
                        ]
                    )
                    + "\n"
                )
                venue_log.flush()
                print(f"ccf {abbr} {item.get('rank')} miss {score:.3f}", flush=True)
                continue
            sid = source_id(source)
            saved = 0
            reported = None
            pages = 0
            try:
                for rows, total, page_i in page_source(sid):
                    reported = total
                    pages = page_i
                    for work in rows:
                        key = work.get("doi") or work.get("id")
                        if key and key in seen:
                            continue
                        if key:
                            seen.add(key)
                        handle.write(
                            json.dumps(
                                slim_openalex(
                                    work,
                                    {
                                        "ccf_abbr": abbr,
                                        "ccf_rank": item.get("rank"),
                                        "ccf_type": item.get("type"),
                                        "ccf_query": name,
                                        "openalex_source": sid,
                                        "source_score": round(score, 3),
                                    },
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        saved += 1
                    handle.flush()
            except Exception as exc:
                miss.write(f"{abbr}\t{name}\tpage {exc}\n")
                miss.flush()
                print("fail", abbr, exc, flush=True)
            incomplete = reported is not None and saved < int(reported)
            flag = "ok"
            if incomplete and pages >= MAX_PAGES:
                flag = "truncated"
            elif incomplete:
                flag = "incomplete"
            print(
                f"ccf {abbr} {item.get('rank')} saved {saved} "
                f"reported {reported} score {score:.3f} source {sid} {flag}",
                flush=True,
            )
            venue_log.write(
                "\t".join(
                    [
                        "ccf",
                        abbr,
                        str(item.get("rank") or ""),
                        str(saved),
                        str(reported),
                        f"{score:.3f}",
                        sid,
                        flag,
                    ]
                )
                + "\n"
            )
            venue_log.flush()


def crossref_total(url: str) -> int | None:
    data = get(url)
    return (data.get("message") or {}).get("total-results")


def census() -> dict:
    ieee = crossref_total(
        "https://api.crossref.org/members/263/works?filter="
        + CROSSREF_FILTER
        + "&rows=0"
    )
    acl = crossref_total(
        "https://api.crossref.org/members/1643/works?filter="
        + CROSSREF_FILTER
        + "&rows=0"
    )
    out = {
        "window": [DATE_FROM, DATE_TO],
        "ieee_crossref_member_263": ieee,
        "acl_crossref_member_1643": acl,
        "ieee_downloaded": False,
        "sci_enumerable": False,
        "ei_enumerable": False,
        "ti_reading": "No public index named TI. EI/Compendex is the reading used here.",
    }
    (COLLECTED / "scope_counts.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"census ieee {ieee} acl {acl}", flush=True)
    return out


def slim_crossref(item: dict, extra: dict | None = None, abbr: str = "ACL") -> dict:
    title = item.get("title") or [""]
    container = item.get("container-title") or []
    issued = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
    publisher = item.get("publisher") or ""
    venue = container[0] if container else ""
    extra = extra or {}
    tagged = marks(venue, publisher, extra.get("ccf_abbr") or abbr)
    rank = extra.get("ccf_rank")
    if rank:
        tagged.insert(0, "ccf-" + rank)
    return {
        "doi": item.get("DOI"),
        "title": title[0] if title else "",
        "year": issued[0] if issued else None,
        "venue": venue,
        "publisher": publisher,
        "type": item.get("type"),
        "url": item.get("URL"),
        "marks": tagged,
        **extra,
    }


def crossref_offset(url: str, max_pages: int, rows: int = 50):
    reported = None
    for page_i in range(max_pages):
        page = get(url + "&rows=" + str(rows) + "&offset=" + str(page_i * rows))
        message = page.get("message") or {}
        if reported is None:
            reported = message.get("total-results")
        batch = message.get("items") or []
        yield batch, reported, page_i + 1
        if not batch:
            break
        if reported is not None and (page_i + 1) * rows >= int(reported):
            break
        time.sleep(0.25)


def crossref_cursor(url: str, max_pages: int, rows: int = 100):
    cursor = "*"
    reported = None
    joiner = "&" if "?" in url else "?"
    for page_i in range(max_pages):
        page = get(
            url
            + joiner
            + "rows="
            + str(rows)
            + "&cursor="
            + urllib.parse.quote(cursor, safe="*")
        )
        message = page.get("message") or {}
        if reported is None:
            reported = message.get("total-results")
        batch = message.get("items") or []
        yield batch, reported, page_i + 1
        cursor = message.get("next-cursor")
        if not batch or not cursor:
            break
        time.sleep(0.25)


def name_forms(name: str) -> list[str]:
    forms = []

    def add(text: str) -> None:
        text = " ".join((text or "").split())
        if len(text) >= 8 and text not in forms:
            forms.append(text)

    if ":" in name:
        add(name.split(":")[0])
    add(name)
    for base in list(forms):
        add(shortened_name(base))
    return forms or [name]


def resolve_issn(name: str, ccf_type: str) -> tuple[str, float, str]:
    if ccf_type != "Journal":
        return "", 0.0, ""
    queries = name_forms(name)
    best_issn = ""
    best_score = 0.0
    best_title = ""
    for query in queries:
        url = (
            "https://api.crossref.org/journals?query="
            + urllib.parse.quote(query)
            + "&rows=5"
        )
        data = get(url)
        time.sleep(0.25)
        for item in (data.get("message") or {}).get("items") or []:
            title = item.get("title") or ""
            score = 0.0
            for form in name_forms(name):
                score = max(
                    score,
                    source_score(
                        form, "Journal", {"display_name": title, "type": "journal"}
                    ),
                )
            if score > best_score:
                issns = item.get("ISSN") or []
                best_score = score
                best_title = title
                best_issn = issns[0] if issns else ""
        if best_score >= JACCARD_MIN and best_issn:
            return best_issn, best_score, best_title
    return "", best_score, best_title


def container_ok(name: str, container: str, abbr: str = "") -> bool:
    if side_venue(container) and not side_venue(name):
        return False
    ctoks = tokens(container)
    if any(jaccard(tokens(form), ctoks) >= JACCARD_MIN for form in name_forms(name)):
        return True
    key = "".join(ch for ch in (abbr or "").lower() if ch.isalnum())
    if len(key) < 3 or key not in ctoks:
        return False
    for form in name_forms(name):
        need = tokens(form)
        if not need or not need <= ctoks:
            continue
        if len(need) >= 2 or max(len(tok) for tok in need) >= 10:
            return True
    return False


def query_candidates(name: str, abbr: str) -> list[str]:
    wide = STOP | {
        "acm", "ieee", "sigplan", "sigda", "sigops", "sigcomm", "usenix",
        "ifip", "siam", "aaai", "cvf", "symposium", "european",
    }
    found = []

    def add(text: str) -> None:
        text = " ".join((text or "").replace("&", " and ").split())
        if len(text) >= 3 and text not in found:
            found.append(text)

    words = name.replace("&", " and ").split()
    index = 0
    while index < len(words):
        word = words[index].strip(".,/:")
        if word.lower() in wide or (word.isupper() and len(word) >= 2):
            index += 1
            continue
        break
    add(" ".join(words[index:]))
    add(name)
    key = "".join(ch for ch in (abbr or "") if ch.isalnum())
    if len(key) >= 3:
        add(key)
    return found


def best_container_query(name: str, abbr: str, work_type: str) -> str:
    best_query = ""
    best_rank = None
    for query in query_candidates(name, abbr):
        url = (
            "https://api.crossref.org/works?query.container-title="
            + urllib.parse.quote(query)
            + "&filter=type:"
            + work_type
            + ","
            + CROSSREF_FILTER
            + "&rows=50&select=container-title"
        )
        data = get(url)
        time.sleep(0.25)
        message = data.get("message") or {}
        items = message.get("items") or []
        if not items:
            continue
        kept = 0
        for item in items:
            container = (item.get("container-title") or [""])[0]
            if container_ok(name, container, abbr):
                kept += 1
        if kept < 8:
            continue
        rank = (kept / len(items), -int(message.get("total-results") or 0))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_query = query
    return best_query


def harvest_ccf_crossref() -> None:
    catalogue = json.loads(CCF_PATH.read_text(encoding="utf-8"))
    path = COLLECTED / "ccf_works.jsonl"
    misses = COLLECTED / "ccf_misses.txt"
    log = COLLECTED / "ccf_venue_log.txt"
    done = finished_abbrs()
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                doi = json.loads(line).get("doi")
            except json.JSONDecodeError:
                continue
            if doi:
                seen.add(doi)
    print(f"crossref resume skip {len(done)} seen {len(seen)}", flush=True)
    mode = "a" if path.exists() else "w"
    select = "&select=DOI,title,container-title,issued,publisher,type,URL"
    with path.open(mode, encoding="utf-8") as handle, misses.open(
        "a", encoding="utf-8"
    ) as miss, log.open("a", encoding="utf-8") as venue_log:
        for item in catalogue["list"]:
            if item.get("rank") not in {"A", "B", "C"}:
                continue
            name = (item.get("name") or "").split("(")[0].strip()
            abbr = item.get("abbr") or ""
            if abbr in done:
                continue
            saved = 0
            reported = None
            pages = 0
            flag = "ok"
            sid = ""
            score = 0.0
            try:
                issn, score, title = resolve_issn(name, item.get("type") or "")
                if issn:
                    sid = "issn:" + issn
                    url = (
                        "https://api.crossref.org/journals/"
                        + urllib.parse.quote(issn)
                        + "/works?filter="
                        + CROSSREF_FILTER
                        + select
                    )
                    for rows, total, page_i in crossref_cursor(url, 80, rows=100):
                        reported = total
                        pages = page_i
                        for work in rows:
                            doi = work.get("DOI")
                            if doi and doi in seen:
                                continue
                            if doi:
                                seen.add(doi)
                            handle.write(
                                json.dumps(
                                    slim_crossref(
                                        work,
                                        {
                                            "ccf_abbr": abbr,
                                            "ccf_rank": item.get("rank"),
                                            "ccf_type": item.get("type"),
                                            "ccf_query": name,
                                            "crossref_id": sid,
                                            "source_score": round(score, 3),
                                        },
                                        abbr,
                                    ),
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            saved += 1
                        handle.flush()
                    if (
                        reported is not None
                        and saved < int(reported)
                        and pages >= 80
                    ):
                        flag = "truncated"
                else:
                    work_type = (
                        "journal-article"
                        if item.get("type") == "Journal"
                        else "proceedings-article"
                    )
                    query = best_container_query(name, abbr, work_type)
                    if not query:
                        flag = "miss"
                        score = 0.0
                        print(
                            f"ccf {abbr} {item.get('rank')} miss no precise query",
                            flush=True,
                        )
                        miss.write(f"{abbr}\t{name}\tno precise query\n")
                        miss.flush()
                        venue_log.write(
                            "\t".join(
                                [
                                    "ccf",
                                    abbr,
                                    str(item.get("rank") or ""),
                                    "0",
                                    "0",
                                    "0.000",
                                    "",
                                    "miss",
                                ]
                            )
                            + "\n"
                        )
                        venue_log.flush()
                        continue
                    url = (
                        "https://api.crossref.org/works?query.container-title="
                        + urllib.parse.quote(query)
                        + "&filter=type:"
                        + work_type
                        + ","
                        + CROSSREF_FILTER
                        + select
                    )
                    sid = "query:" + query
                    empty_tail = 0
                    kept_last = False
                    # Cursor paging reorders this query. Offset keeps the venue page.
                    for rows, total, page_i in crossref_offset(url, 40, rows=50):
                        reported = total
                        pages = page_i
                        page_kept = 0
                        for work in rows:
                            container = (work.get("container-title") or [""])[0]
                            if not container_ok(name, container, abbr):
                                continue
                            doi = work.get("DOI")
                            if doi and doi in seen:
                                continue
                            if doi:
                                seen.add(doi)
                            handle.write(
                                json.dumps(
                                    slim_crossref(
                                        work,
                                        {
                                            "ccf_abbr": abbr,
                                            "ccf_rank": item.get("rank"),
                                            "ccf_type": item.get("type"),
                                            "ccf_query": name,
                                            "crossref_id": sid,
                                            "source_score": round(
                                                jaccard(tokens(name), tokens(container)),
                                                3,
                                            ),
                                        },
                                        abbr,
                                    ),
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            saved += 1
                            page_kept += 1
                        handle.flush()
                        kept_last = page_kept > 0
                        empty_tail = 0 if page_kept else empty_tail + 1
                        if empty_tail >= 2:
                            break
                        if rows and page_kept / len(rows) < 0.2:
                            break
                    if pages >= 40 and kept_last:
                        flag = "truncated"
                    if saved == 0:
                        flag = "miss"
                        score = 0.0
            except Exception as exc:
                flag = "incomplete"
                miss.write(f"{abbr}\t{name}\t{exc}\n")
                miss.flush()
                print("fail", abbr, exc, flush=True)
            if flag == "miss":
                miss.write(f"{abbr}\t{name}\tno crossref rows\n")
                miss.flush()
            print(
                f"ccf {abbr} {item.get('rank')} saved {saved} "
                f"reported {reported} score {score:.3f} source {sid} {flag}",
                flush=True,
            )
            venue_log.write(
                "\t".join(
                    [
                        "ccf",
                        abbr,
                        str(item.get("rank") or ""),
                        str(saved),
                        str(reported),
                        f"{score:.3f}",
                        sid,
                        flag,
                    ]
                )
                + "\n"
            )
            venue_log.flush()


def harvest_acl() -> None:
    path = COLLECTED / "acl_works.jsonl"
    url = (
        "https://api.crossref.org/members/1643/works?filter="
        + CROSSREF_FILTER
        + "&rows=100&select=DOI,title,container-title,issued,publisher,type,URL"
    )
    cursor = "*"
    seen = set()
    saved = 0
    with path.open("w", encoding="utf-8") as handle:
        for page_i in range(700):
            page = get(url + "&cursor=" + urllib.parse.quote(cursor, safe="*"))
            message = page.get("message") or {}
            rows = message.get("items") or []
            if page_i == 0:
                print(f"acl reported {message.get('total-results')}", flush=True)
            for item in rows:
                doi = item.get("DOI")
                if doi and doi in seen:
                    continue
                if doi:
                    seen.add(doi)
                handle.write(
                    json.dumps(slim_crossref(item), ensure_ascii=False) + "\n"
                )
                saved += 1
            cursor = message.get("next-cursor")
            if not rows or not cursor:
                break
            if page_i % 25 == 24:
                handle.flush()
                print(f"acl page {page_i + 1} saved {saved}", flush=True)
            time.sleep(0.12)
    print(f"acl saved {saved}", flush=True)


def main() -> None:
    COLLECTED.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "check":
        sys.exit(0 if check_sources() else 2)
    if mode == "resume":
        harvest_ccf_crossref()
        harvest_acl()
        return
    if not check_sources():
        print("source check failed; crawl not started", flush=True)
        sys.exit(2)
    if mode in {"all", "census"}:
        census()
    if mode in {"all", "ccf"}:
        harvest_ccf()
    if mode in {"all", "acl"}:
        harvest_acl()


if __name__ == "__main__":
    main()
