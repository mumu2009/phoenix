from __future__ import annotations


def createSecurityAddon(name: str):
    return {"name": name or "security", "type": "security", "defaultEnabled": False}
