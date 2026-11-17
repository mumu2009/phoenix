#!/usr/bin/env python3
"""Rebase the /phx/cancel feature from phx_generate_cancel.patch onto the new
canonical vendor base (pristine 733dd8df + phoenix_vendor_733dd8df.patch).

The old corrupt patch was written against a ~6000-line server.cpp that also
carried the unit-loop/chunked features (absent from the current tree).  Its
four low-risk hunks (epoch member, max_tokens floor, cancel endpoint) rebase
verbatim; the two mid-generation abort hooks are re-anchored onto the current
generate loops (the old `for (step < max_tokens; ++step)` loops were refactored
into the jacobi/speculative loops).

This script edits the TARGET TREE only; the diff itself is produced by git
afterwards, so no patch file is hand-written.
"""
import re
import sys

def must_replace(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit("anchor %r matched %d times (expected 1)" % (label, n))
    return text.replace(old, new, 1)


def main(path: str) -> None:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n") if crlf else raw

    # 1) per-request cancel epoch, next to the existing phx mutex.
    text = must_replace(
        text,
        "    std::mutex phx_mutex;\n",
        "    std::mutex phx_mutex;\n"
        "    std::atomic<uint64_t> phx_cancel_epoch{0};\n",
        "phx_mutex member",
    )

    # 2) max_tokens floor of 16 + snapshot the cancel epoch for this request.
    text = must_replace(
        text,
        "        const int32_t max_tokens = json_value(body, \"max_tokens\", 16);\n"
        "        if (max_tokens < 1) {\n",
        "        const int32_t max_tokens_req = json_value(body, \"max_tokens\", 16);\n"
        "        const int32_t max_tokens = std::max<int32_t>(16, max_tokens_req);\n"
        "        const uint64_t cancel_at = ctx_server.phx_cancel_epoch.load(std::memory_order_acquire);\n"
        "        if (max_tokens_req < 1) {\n",
        "max_tokens floor",
    )

    # 3) abort the main generate loop on client disconnect or /phx/cancel.
    loop_anchor = "            for (int32_t step = 0; step < max_tokens; ) {\n"
    guard_outer = (
        "                if (req.is_connection_closed() ||\n"
        "                    ctx_server.phx_cancel_epoch.load(std::memory_order_acquire) != cancel_at) {\n"
        "                    break;\n"
        "                }\n"
    )
    text = must_replace(text, loop_anchor, loop_anchor + guard_outer, "generate for-loop")

    # 4) abort the jacobi inner loop too (it consumes extra decode steps).
    inner_anchor = "                    while (guard < n_parallel && step < max_tokens) {\n"
    guard_inner = (
        "                        if (req.is_connection_closed() ||\n"
        "                            ctx_server.phx_cancel_epoch.load(std::memory_order_acquire) != cancel_at) {\n"
        "                            break;\n"
        "                        }\n"
    )
    text = must_replace(text, inner_anchor, inner_anchor + guard_inner, "jacobi while-loop")

    # 5) /phx/cancel endpoint registration, right after /phx/generate.
    route_anchor = '    svr->Post("/phx/generate",        handle_phx_generate);\n'
    route = (
        '    svr->Post("/phx/cancel",          [&ctx_server, &res_ok](const httplib::Request &, httplib::Response & res) {\n'
        "        ctx_server.phx_cancel_epoch.fetch_add(1, std::memory_order_acq_rel);\n"
        '        res_ok(res, json{{"ok", true}, {"cancelled", true}});\n'
        "    });\n"
    )
    text = must_replace(text, route_anchor, route_anchor + route, "route registration")

    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text.replace("\n", "\r\n") if crlf else text)
    print("rebuilt /phx/cancel feature on %s" % path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: rebuild_cancel_feature.py <server.cpp>")
    main(sys.argv[1])
