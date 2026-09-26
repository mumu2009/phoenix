# manifold_farm.py -- unattended meme farm: iterate fresh frozen instruments
# over disjoint wikitext-103 slices, screen the need=2 pool, run the dialogue
# serial on llama :8082, classify, and accumulate confirmed memes until the
# target count is reached. Built for long unattended runs (paper data volume):
# every step is resumable and all state lives in runtime_store/farm_state.json.
#
# Pipeline per instrument slice (skip = 1280, 1792, ...):
#   1. meme_barrier_instrument.exe screen --skip K --units 512 --pool eligible
#      --relax  (builds/loads farm_instrument_K.txt; tokenizer cache makes the
#      screen ~8 min; debris filter applied at batch build, not screen)
#   2. batch build: typicalNeed==2 AND carrierEntropy>0 AND typicalCover>0
#      (the partC lesson: relax admits carrierEntropy=0 debris that compose
#      always rejects), dedupe against every id ever tested (meme_p_ ids are
#      content-addressed and collide across instruments), entropy desc, take N
#   3. pwsh tools/phoenix_gnn_sig_serial_dialogue.ps1 -OutName farm_K
#      -ScreenFile batch -SnapFile snap -ApiPort 8082   (~1.5-2 h per 65)
#   4. python tools/manifold_batch_classify.py <batch.json> -> merge verdicts
#      into farm_state.json (memes / antimemes / tested ids)
#
# Usage:
#   python tools\manifold_farm.py --target 88            # run until 88 memes
#   python tools\manifold_farm.py --target 88 --dry-run  # build only, no llama
#   python tools\manifold_farm.py --resume               # continue farm_state
# Stop conditions: target reached, corpus exhausted (slice too small), or a
# batch returns 0 groups (llama down -> abort instead of spinning dirty).
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RS = ROOT / "runtime_store"
EXE = ROOT / "build" / "meme_barrier_instrument.exe"
SERIAL = ROOT / "tools" / "phoenix_gnn_sig_serial_dialogue.ps1"
CLASSIFY = ROOT / "tools" / "manifold_batch_classify.py"
CLASSIFY_OUT = RS / "manifold_batch_classify.json"
CORPUS = ROOT / "robots" / "wikitext-103-all.txt"
STATE = RS / "farm_state.json"
SEED = RS / "farm_seed.json"
LOG = RS / "farm.log"
LOCK = RS / "farm.lock"


def acquire_lock():
    """Single-instance lock: two concurrent farms would corrupt the shared
    snap/screen files (observed: duplicate smoke runs double-writing
    farm_screen_1280.json). Lock holds our pid; stale locks (dead pid) are
    reclaimed."""
    if LOCK.exists():
        try:
            old_pid = int(LOCK.read_text().strip())
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, old_pid)
            if h:
                ctypes.windll.kernel32.CloseHandle(h)
                raise RuntimeError(
                    "another farm is running (pid %d); aborting" % old_pid)
        except ValueError:
            pass
    LOCK.write_text(str(os.getpid()))


def release_lock():
    try:
        LOCK.unlink(missing_ok=True)
    except OSError:
        pass

UNITS_PER_INSTRUMENT = 512
MIN_SLICE_UNITS = 200  # smaller tail slice = corpus nearly exhausted


def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    st = {"testedIds": [], "memes": [], "antimemes": 0, "batches": []}
    if SEED.exists():
        seed = json.loads(SEED.read_text(encoding="utf-8"))
        st["testedIds"] = seed.get("testedIds", [])
        st["memes"] = seed.get("memes", [])
        log("seeded from farm_seed.json: %d memes, %d tested"
            % (len(st["memes"]), len(st["testedIds"])))
    return st


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False),
                     encoding="utf-8")


def run(cmd, stdout_to=None, timeout_h=4):
    log("run: " + " ".join(str(c) for c in cmd)[:160])
    fh = (open(stdout_to, "w", encoding="utf-8") if stdout_to
          else subprocess.DEVNULL)
    try:
        with subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh,
                              stderr=subprocess.STDOUT) as p:
            try:
                rc = p.wait(timeout=int(timeout_h * 3600))
            except subprocess.TimeoutExpired:
                p.kill()
                raise RuntimeError("timeout: %s" % cmd[0])
    finally:
        if fh is not subprocess.DEVNULL:
            fh.close()
    if rc != 0:
        raise RuntimeError("exit %d: %s" % (rc, cmd[0]))


def build_instrument(skip):
    """Screen one corpus slice; returns (snap_path, screen_json)."""
    snap = RS / ("farm_instrument_%d.txt" % skip)
    screen = RS / ("farm_screen_%d.json" % skip)
    if screen.exists() and snap.exists():
        try:
            return snap, json.loads(screen.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log("stale/broken screen %s - re-screening" % screen.name)
    run([str(EXE), "screen", "--corpus", str(CORPUS), "--snap", str(snap),
         "--units", str(UNITS_PER_INSTRUMENT), "--skip", str(skip),
         "--take", "400", "--pool", "eligible", "--relax"],
        stdout_to=screen)
    return snap, json.loads(screen.read_text(encoding="utf-8"))


def build_batch(skip, screen, tested):
    cands = [m for m in screen.get("memes", [])
             if m.get("typicalNeed") == 2
             and (m.get("carrierEntropy") or 0) > 0
             and (m.get("typicalCover") or 0) > 0
             and m.get("id") not in tested]
    cands.sort(key=lambda m: -(m.get("carrierEntropy") or 0))
    take = cands[:ARGS.batch_size]
    out = dict(screen)
    out["note"] = ("farm skip=%d need=2 debris-filtered deduped entropy desc"
                   % skip)
    out["memes"] = take
    batch = RS / ("farm_batch_%d.json" % skip)
    batch.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return batch, take


def run_serial(skip, snap, batch):
    out_name = "farm_%d" % skip
    run(["pwsh", "-NoProfile", "-File", str(SERIAL),
         "-OutName", out_name, "-ScreenFile", str(batch),
         "-SnapFile", str(snap), "-ApiPort", str(ARGS.port)],
        stdout_to=RS / (out_name + ".stdout.log"), timeout_h=8)
    result = RS / (out_name + ".json")
    if not result.exists():
        raise RuntimeError("serial produced no output for %s" % out_name)
    return result


def classify_into(result, st):
    run([sys.executable, str(CLASSIFY), str(result)])
    rep = json.loads(CLASSIFY_OUT.read_text(encoding="utf-8"))
    new_memes = 0
    for row in rep["rows"]:
        rid = row["id"]
        if rid not in st["_tested"]:
            st["_tested"].add(rid)
            st["testedIds"].append(rid)
        if row["verdict"] == "meme" and rid not in st["_memes"]:
            st["_memes"].add(rid)
            st["memes"].append(rid)
            new_memes += 1
        elif row["verdict"] == "antimeme":
            st["antimemes"] += 1
    st["batches"].append({
        "file": result.name, "groups": len(rep["rows"]),
        "memes": rep["memeCount"], "antimemes": rep["antimemeCount"],
        "spearman": rep["spearmanFieldVsSurvivalRounds"],
        "finishedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return new_memes, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=88,
                    help="stop when total confirmed memes reaches this")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--start-skip", type=int, default=1280,
                    help="first unused corpus unit (0-1279 are instruments "
                         "merge/b2/b3)")
    ap.add_argument("--batch-size", type=int, default=65)
    ap.add_argument("--max-instruments", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true",
                    help="build instrument + batch only, never touch llama")
    ap.add_argument("--resume", action="store_true",
                    help="continue from farm_state.json (default anyway)")
    global ARGS
    ARGS = ap.parse_args()

    acquire_lock()
    try:
        return farm_loop()
    finally:
        release_lock()


def farm_loop():
    st = load_state()
    st["_tested"] = set(st["testedIds"])
    st["_memes"] = set(st["memes"])
    log("farm start: memes=%d/%d tested=%d"
        % (len(st["memes"]), ARGS.target, len(st["testedIds"])))

    skip = ARGS.start_skip
    for _ in range(ARGS.max_instruments):
        if len(st["memes"]) >= ARGS.target:
            break
        snap, screen = build_instrument(skip)
        if screen.get("ingestedUnits", 0) < MIN_SLICE_UNITS:
            log("corpus exhausted at skip=%d (units=%s)"
                % (skip, screen.get("ingestedUnits")))
            break
        batch, cands = build_batch(skip, screen, st["_tested"])
        log("skip=%d: %d clean need=2 candidates (batch %d)"
            % (skip, len(cands), len(cands)))
        if not cands:
            skip += UNITS_PER_INSTRUMENT
            continue
        if ARGS.dry_run:
            log("dry-run: skipping serial for %s" % batch.name)
            skip += UNITS_PER_INSTRUMENT
            continue
        result = run_serial(skip, snap, batch)
        groups = len(json.loads(result.read_text(encoding="utf-8"))
                     .get("groups", []))
        if groups == 0:
            log("FATAL: 0 groups - llama :%d down? aborting, fix and "
                "--resume" % ARGS.port)
            save_state(st)
            return 1
        new_memes, rep = classify_into(result, st)
        save_state(st)
        log("batch %s: +%d memes (total %d/%d), antimemes=%d, rho=%.3f"
            % (result.name, new_memes, len(st["memes"]), ARGS.target,
               st["antimemes"], rep["spearmanFieldVsSurvivalRounds"]))
        skip += UNITS_PER_INSTRUMENT

    st.pop("_tested", None)
    st.pop("_memes", None)
    save_state(st)
    log("farm done: memes=%d/%d antimemes=%d batches=%d"
        % (len(st["memes"]), ARGS.target, st["antimemes"],
           len(st["batches"])))
    return 0 if len(st["memes"]) >= ARGS.target else 2


if __name__ == "__main__":
    sys.exit(main())
