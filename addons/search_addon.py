from __future__ import annotations

from typing import Any, Dict

from addon import AddonResult, invokeAddonOnlineLookup


def _trim(s: str) -> str:
    return (s or "").strip()


def _lower(s: str) -> str:
    return _trim(s).lower()


class SearchAddon:
    def __init__(self, name: str) -> None:
        self._name = name or "search"

    def name(self) -> str:
        return self._name

    def type(self) -> str:
        return "search"

    def handle(self, text: str, payload: Dict[str, Any]) -> AddonResult:
        addon_type = _lower(str(payload.get("__addonType", "")))
        if addon_type not in {"", "search", "research", "web"}:
            return AddonResult()

        query = _trim(text)
        for p in ["search:", "web:", "lookup:", "research:", "搜索:", "检索:", "查询:"]:
            if _lower(query).startswith(p):
                query = _trim(query[len(p) :])
                break
        if not query:
            return AddonResult()

        options = payload.get("searchOptions", {}) if isinstance(payload.get("searchOptions"), dict) else {}
        options.setdefault("preferIndex", True)
        lookup = invokeAddonOnlineLookup(query, options)
        if not lookup:
            return AddonResult()

        reply = str(lookup.get("snippet") or lookup.get("text") or "").strip()
        if not reply and isinstance(lookup.get("suggestions"), list) and lookup["suggestions"]:
            first = lookup["suggestions"][0]
            words = first.get("words", []) if isinstance(first, dict) else []
            reply = " ".join(str(w) for w in words)
        if not reply:
            reply = str(lookup)
        if len(reply) > 1200:
            reply = reply[:1200]

        return AddonResult(
            handled=True,
            reply=reply,
            meta={"addon": "search", "name": self._name, "query": query, "source": lookup.get("source", "unknown")},
        )


def createSearchAddon(name: str) -> SearchAddon:
    return SearchAddon(name or "search")
