#!/usr/bin/env python3
"""Regenerate a valid unified-diff patch from a corrupt patch body.

The two corrupt Phoenix patches fail `git apply` for structural reasons only:
  * phx_chunked_encode.patch : single complete hunk whose @@ old/new counts are wrong
  * phx_generate_cancel.patch: truncated last hunk + wrong @@ counts + leading comments

This script does NOT hand-edit content. It parses the hunks that are present in
the file, recomputes each hunk header count from the actual body lines, and
rewrites a syntactically valid patch.  Missing trailing context is legal for a
unified diff (the hunk simply ends earlier); no feature line is invented.
"""
import re
import sys

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def repair(src: str, dst: str) -> int:
    with open(src, "r", encoding="utf-8", errors="replace", newline="") as fh:
        raw = fh.read().splitlines()

    # Drop any leading prose/comments; a patch starts at a file header or diff --git.
    start = 0
    for idx, line in enumerate(raw):
        if line.startswith("diff --git ") or line.startswith("--- "):
            start = idx
            break
    body = raw[start:]

    out = []
    i = 0
    hunks = 0
    while i < len(body):
        line = body[i]
        if line.startswith("@@ "):
            m = HUNK_RE.match(line)
            if not m:
                raise SystemExit("bad hunk header at output line %d: %r" % (i, line))
            old_start = int(m.group(1))
            new_start = int(m.group(3))
            tail = m.group(5)
            i += 1
            hunk = []
            while i < len(body):
                nxt = body[i]
                if nxt.startswith("@@ ") or nxt.startswith("diff --git ") or nxt.startswith("--- "):
                    break
                if nxt == "":
                    nxt = " "  # tolerate an empty context line: canonical diff needs the space
                hunk.append(nxt)
                i += 1
            old_count = 0
            new_count = 0
            for hl in hunk:
                if hl.startswith("\\"):  # "\ No newline at end of file"
                    continue
                if hl[:1] in (" ", "-"):
                    old_count += 1
                if hl[:1] in (" ", "+"):
                    new_count += 1
            hunks += 1
            out.append("@@ -%d,%d +%d,%d @@%s" % (old_start, old_count, new_start, new_count, tail))
            out.extend(hunk)
            continue
        out.append(line)
        i += 1

    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    return hunks


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: repair_corrupt_patch.py <src.patch> <dst.patch>")
    n = repair(sys.argv[1], sys.argv[2])
    print("repaired hunks=%d -> %s" % (n, sys.argv[2]))
