"""Extract the faulting stack from a WER minidump and resolve it against a
linker .map file.

Why: 0xC0000374 is raised inside ntdll (the heap checker), so the exception
address tells us nothing.  What matters is the RETURN-ADDRESS CHAIN of the
thread that tripped the check - that is the code that freed/allocated the
corrupted block.  This tool:

  1. reads the exception stream (code, faulting thread),
  2. reads that thread's stack memory,
  3. keeps 8-byte words that fall inside the main module's image range
     (those are return addresses - the minidump has no unwind info we can
     use without symbols),
  4. prints them as module-relative offsets, then resolves each one to the
     nearest preceding symbol in a MinGW/GNU ld .map file (compile.bat now
     passes -Wl,-Map=phoenix_main.map).

Usage:
  python tools/dump_stack_resolve.py <dump> [--map phoenix_main.map]
                                    [--module phoenix_main.exe]
"""
import argparse
import bisect
import struct
import sys

try:
    from minidump.minidumpfile import MinidumpFile
except Exception as exc:  # pragma: no cover - dependency error path
    sys.exit("minidump module missing (pip install minidump): %s" % exc)


def load_map(path):
    """Return (sorted_addrs, symbols) from a GNU ld .map file.

    Lines look like:
      0x0000000000401b23                phoenix::foo()
    Ignore lines whose address is not followed by a symbol name.
    """
    addrs, names = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("0x"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            try:
                addr = int(parts[0], 16)
            except ValueError:
                continue
            name = parts[1].strip()
            if not name:
                continue
            addrs.append(addr)
            names.append(name)
    order = sorted(range(len(addrs)), key=lambda i: addrs[i])
    return [addrs[i] for i in order], [names[i] for i in order]


def resolve(addrs, names, addr):
    if not addrs:
        return None
    i = bisect.bisect_right(addrs, addr) - 1
    if i < 0:
        return None
    return names[i], addr - addrs[i]


def module_range(dump, want):
    for mod in dump.modules.modules:
        name = (mod.name or "").lower()
        if want.lower() in name:
            base = mod.baseaddress
            return base, base + mod.size
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--map", default="phoenix_main.map")
    ap.add_argument("--module", default="phoenix_main.exe")
    ap.add_argument("--max-frames", type=int, default=40)
    args = ap.parse_args()

    dump = MinidumpFile.parse(args.dump)
    base, end = module_range(dump, args.module)
    if base is None:
        sys.exit("module %s not found in dump" % args.module)
    print("module %s base=0x%x end=0x%x" % (args.module, base, end))

    exc = dump.exception
    rec = None
    target_tid = None
    if exc is not None:
        recs = getattr(exc, "exception_records", None)
        if recs:
            rec = recs[0]
            target_tid = getattr(rec, "ThreadId", None)
        print("exception code=%s thread_id=%s"
              % (getattr(rec, "ExceptionCode", "?"), target_tid))
    else:
        print("exception stream absent")

    addrs, names = [], []
    try:
        addrs, names = load_map(args.map)
        print("map symbols: %d (from %s)" % (len(addrs), args.map))
    except FileNotFoundError:
        print("map file %s not found - offsets only" % args.map)

    # The thread that tripped the heap check.
    threads = dump.threads.threads if dump.threads else []
    chosen = None
    for t in threads:
        if target_tid is not None and t.ThreadId == target_tid:
            chosen = t
            break
    if chosen is None and threads:
        chosen = threads[0]
    if chosen is None:
        sys.exit("no threads in dump")

    tid = chosen.ThreadId
    stack = chosen.Stack
    print("thread %s stack start=0x%x" %
          (tid, stack.StartOfMemoryRange))
    try:
        segs = []
        if getattr(dump, "memory_segments_64", None) is not None:
            segs = dump.memory_segments_64.memory_segments
        elif getattr(dump, "memory_segments", None) is not None:
            segs = dump.memory_segments.segments
        start = int(stack.StartOfMemoryRange)
        seg = None
        for s in segs:
            a = int(s.start_virtual_address)
            n = int(s.size)
            if a <= start < a + n:
                seg = (s, a, n)
                break
        if seg is None:
            sys.exit("no memory segment covers the faulting stack")
        s, a, n = seg
        with open(args.dump, "rb") as fh:
            fh.seek(int(s.start_file_address) + (start - a))
            data = fh.read(n - (start - a))
    except Exception as exc2:
        sys.exit("cannot read stack: %s" % exc2)

    # The .map lists absolute VAs with the PE preferred image base
    # (0x140000000 for a default MinGW x64 link).  Convert each relocated
    # return address to that space before resolving.
    preferred = 0x140000000
    frames = []
    for off in range(0, len(data) - 8, 8):
        (val,) = struct.unpack_from("<Q", data, off)
        if base <= val < end:
            frames.append((off, val))
    print("plausible return addresses: %d" % len(frames))
    for off, val in frames[: args.max_frames]:
        rva = val - base
        line = "  stack+0x%04x 0x%016x (rva 0x%x)" % (off, val, rva)
        hit = resolve(addrs, names, preferred + rva)
        if hit:
            line += "  -> %s +0x%x" % hit
        print(line)


if __name__ == "__main__":
    main()
