#!/usr/bin/env python3
"""audit_check.py - Phoenix static "did we already learn this?" checker.

Stdlib-only, Windows-friendly, READ-ONLY on product sources.  It encodes the
lessons of our real incidents as *rules* so that version drift, dead code/data
and repeat-offence bug patterns are detected before they bite again.

Groups
  R1 VERSION CONSISTENCY  (user worry: "don't die of version problems")
  R2 DEAD CODE / DATA     (heuristic evidence; consumes deadcode-audit's
                           build/tmp/dead_assets.json when present; NEVER
                           auto-deletes anything)
  R3 LESSON GATES         (patterns from our own postmortems:
                           0xC0000374 static-container races, unlocked member
                           reads, fromJson/importState restoring switches,
                           disabled code paths, locks held across IO)

Outputs (default):
  build/tmp/audit_check.json   machine-readable, rule + finding + evidence
  build/tmp/audit_check.md     human summary, top findings, per-rule detail

Usage:
  python tools/audit_check.py                 # all groups, write both files
  python tools/audit_check.py --only R1,R3    # subset
  python tools/audit_check.py --quiet --fail-on warn
  python tools/audit_check.py --no-subprocess # skip git/nvcc/g++ invocations

Exit code: 0 = no FAIL, 1 = at least one FAIL (or WARN with --fail-on warn).
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import fnmatch
import glob
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

TOOL_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# directories never scanned as "product source"
SKIP_DIRS = {
    "build", "outsides", ".git", ".venv", "__pycache__", ".pytest_cache",
    "runtime_store", "runtime", "runtime_opt", "snapshots", "checkpoints",
    "models", "GGUF_models", "gguf", "uploads", "dist", "addons/__pycache__",
    "catastrophe", "experiments", "hacktests", "test-tools", "tests",
}

# --------------------------------------------------------------------------
# rule registry
# --------------------------------------------------------------------------
RULES = []


def rule(rid, group, severity, title, rationale):
    def deco(fn):
        RULES.append(dict(id=rid, group=group, severity=severity, title=title,
                          rationale=rationale, fn=fn))
        return fn
    return deco


class Ctx:
    def __init__(self, root: Path, args):
        self.root = root
        self.args = args
        self.started = time.monotonic()
        self._cache = {}

    def get(self, key, fn):
        if key not in self._cache:
            try:
                self._cache[key] = fn()
            except Exception as exc:  # noqa: BLE001 - rules must not crash run
                self._cache[key] = exc
        val = self._cache[key]
        if isinstance(val, Exception):
            raise val
        return val


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------

def read_text(path: Path, limit: int = 12 * 1024 * 1024) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    except Exception:
        return None


def rel(ctx: Ctx, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ctx.root)).replace("\\", "/")
    except Exception:
        return str(p)


def evidence(ctx: Ctx, p: Path, lineno=None, text=None, limit: int = 200) -> str:
    loc = rel(ctx, p)
    if lineno is not None:
        loc += ":%d" % lineno
    if text is not None:
        t = " ".join(str(text).strip().split())
        if len(t) > limit:
            t = t[: limit - 3] + "..."
        return "%s | %s" % (loc, t)
    return loc


def run(ctx: Ctx, cmd, cwd=None, timeout=20, env=None):
    if ctx.args.no_subprocess:
        return None
    try:
        child_env = None
        if env:
            child_env = dict(os.environ)
            child_env.update(env)
        proc = subprocess.run(cmd, cwd=str(cwd or ctx.root), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, env=child_env)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError:
        return 127, "", "not found: %s" % cmd[0]
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:  # noqa: BLE001
        return 1, "", str(exc)


def runv(ctx: Ctx, cmd, **kw):
    """run() that never returns None (no-subprocess mode yields a failure tuple)."""
    res = run(ctx, cmd, **kw)
    return res if res is not None else (1, "", "subprocess probes disabled")


def which(ctx: Ctx, name: str):
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(p) / name
        if cand.is_file():
            return cand
        cand = Path(p) / (name + ".exe")
        if cand.is_file():
            return cand
    return None


def product_source_files(ctx: Ctx):
    """All C/C++ source fragments that can participate in a product build."""
    root = ctx.root
    files = []

    def take(pattern):
        for p in root.glob(pattern):
            if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
                files.append(p)

    take("*.cpp")
    take("*.hpp")
    take("*.inc")
    take("*.h")
    take("main_hub_parts/*.inc")
    take("main_hub_parts/*.hpp")
    take("addons/**/*.cpp")
    take("addons/**/*.hpp")
    take("module_overrides/**/*.cpp")
    take("module_overrides/**/*.hpp")
    # de-dup, stable order
    seen, out = set(), []
    for p in files:
        try:
            key = str(p.resolve()).lower()
        except OSError:
            key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def source_blob(ctx: Ctx):
    def build():
        parts = []
        for p in product_source_files(ctx):
            try:
                if p.stat().st_size > 8 * 1024 * 1024:
                    continue  # transformer_* giants: skip for token scan only
                parts.append(read_text(p, 2 * 1024 * 1024))
            except OSError:
                continue
        return "\n".join(parts)
    return ctx.get("source_blob", build)


def identifier_set(ctx: Ctx):
    def build():
        blob = source_blob(ctx)
        return set(re.findall(r"[A-Za-z0-9_]{2,}", blob))
    return ctx.get("identifiers", build)


def iter_lines(ctx: Ctx, pattern="*"):
    for p in product_source_files(ctx):
        if pattern != "*" and not fnmatch.fnmatch(p.name, pattern):
            continue
        text = read_text(p)
        if not text:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            yield p, i, line


def walk_limited(root: Path, max_depth=3, max_files=2000, time_budget=8.0):
    """Bounded recursive walk; returns list of Path (files only)."""
    out = []
    t0 = time.monotonic()
    root = root.resolve()
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            if time.monotonic() - t0 > time_budget or len(out) >= max_files:
                return out
            try:
                if e.is_dir(follow_symlinks=False):
                    if depth + 1 <= max_depth:
                        stack.append((Path(e.path), depth + 1))
                elif e.is_file(follow_symlinks=False):
                    out.append(Path(e.path))
            except OSError:
                continue
    return out


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return ("%d B" % n) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1024.0
    return str(n)


def file_version(path: Path):
    """Windows file version via version.dll, stdlib ctypes only."""
    try:
        size = ctypes.windll.version.GetFileVersionInfoSizeW(str(path), None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not ctypes.windll.version.GetFileVersionInfoW(str(path), 0, size, buf):
            return None
        p = ctypes.c_void_p()
        n = ctypes.c_uint()
        if not ctypes.windll.version.VerQueryValueW(
                buf, "\\", ctypes.byref(p), ctypes.byref(n)):
            return None

        class VS_FIXEDFILEINFO(ctypes.Structure):
            _fields_ = [("signature", ctypes.c_uint32),
                        ("struc_version", ctypes.c_uint32),
                        ("file_version_ms", ctypes.c_uint32),
                        ("file_version_ls", ctypes.c_uint32)]

        info = ctypes.cast(p, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
        ms, ls = info.file_version_ms, info.file_version_ls
        return "%d.%d.%d.%d" % (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)
    except Exception:
        return None


# --------------------------------------------------------------------------
# R1 - VERSION CONSISTENCY
# --------------------------------------------------------------------------
@rule("R1.1", "R1", "error",
      "Compiler version vs conan build settings",
      "Conan prebuilts (drogon/opencv/sqlite3/...) are linked into the same exe; "
      "if the conan profile compiler differs from the actual g++, any ABI change "
      "turns into undefined references at link time (the _GLIBCXX_DEBUG link "
      "failure that killed the debug sidecar is the same failure class).")
def r11_compiler_vs_conan(ctx: Ctx):
    findings, stats = [], {}
    # compile.bat resolves GXX_EXE via `where g++` at runtime; static parsing is
    # unreliable, so prefer PATH/MSYS2 and use compile.bat only as corroboration.
    gxx = which(ctx, "g++")
    if not gxx:
        cand = Path(r"C:\msys64\mingw64\bin\g++.exe")
        gxx = cand if cand.exists() else None
    if not gxx:
        findings.append(dict(severity="error",
                             message="g++ not found (compile.bat requires MSYS2 mingw64 g++)",
                             evidence=["PATH", r"C:\msys64\mingw64\bin\g++.exe"]))
        return findings, stats
    stats["gxx"] = str(gxx)
    ver_out = ""
    rc, out, err = runv(ctx, [str(gxx), "--version"], timeout=20)
    if rc == 0:
        ver_out = out.strip().splitlines()[0] if out.strip() else ""
    else:
        ver_out = (err.strip() or "compiler invocation failed")
    stats["gxx_version"] = ver_out
    gm = re.search(r"(\d+)\.(\d+)\.(\d+)", ver_out)
    gcc_major = int(gm.group(1)) if gm else None
    if gcc_major is None:
        findings.append(dict(severity="warn",
                             message="cannot determine g++ version (compile.bat assumes %s)" % gxx,
                             evidence=[str(gxx), ver_out[:160]]))
        return findings, stats

    # conan profile
    prof = ctx.root / "build" / "conan_home" / "profiles" / "default"
    ptext = read_text(prof)
    settings = dict(re.findall(r"^([A-Za-z0-9_.]+)=(.+)$", ptext, re.M))
    stats["conan_profile"] = rel(ctx, prof) if prof.exists() else "missing"
    stats["conan_compiler_version"] = settings.get("compiler.version")
    stats["conan_libcxx"] = settings.get("compiler.libcxx")
    stats["conan_cppstd"] = settings.get("compiler.cppstd")
    if not prof.exists():
        findings.append(dict(severity="warn",
                             message="conan profile build/conan_home/profiles/default is missing; "
                                     "compile.bat will run 'conan profile detect' and may pick a different ABI",
                             evidence=[rel(ctx, prof)]))
    else:
        cv = settings.get("compiler.version")
        if cv and gcc_major and int(cv) != gcc_major:
            findings.append(dict(
                severity="error",
                message="conan packages are pinned to gcc %s but compile.bat compiles with g++ %s"
                        % (cv, gcc_major),
                evidence=[evidence(ctx, prof, None, "compiler.version=%s libcxx=%s cppstd=%s"
                                   % (cv, settings.get("compiler.libcxx"),
                                      settings.get("compiler.cppstd"))),
                          "actual: %s" % ver_out[:160],
                          "why it matters: prebuilt libraries were built with gcc %s ABI" % cv]))
        # gather any conaninfo samples for corroboration
        infos = []
        for p in (ctx.root / "build" / "conan_home" / "p").rglob("conaninfo.txt"):
            infos.append(p)
            if len(infos) >= 400:
                break
        versions = {}
        for p in infos:
            t = read_text(p, 65536)
            mm = re.search(r"compiler\.version=(\d+)", t)
            if mm:
                versions[mm.group(1)] = versions.get(mm.group(1), 0) + 1
        stats["conan_package_compiler_versions"] = versions
        stats["conaninfo_samples"] = len(infos)
    return findings, stats


@rule("R1.2", "R1", "warn",
      "conanfile requirements vs installed package metadata",
      "conanfile.txt is the dependency contract; if build/ metadata disagrees, the "
      "exe was linked against unexpected library versions (opencv/nlohmann ABI and "
      "opset drift surface as silent behaviour changes, not compile errors).")
def r12_conan_requires(ctx: Ctx):
    findings, stats = [], {}
    text = read_text(ctx.root / "conanfile.txt")
    requires = re.findall(r"^([A-Za-z0-9_+\-]+)/([0-9][^@\s]*)", text, re.M)
    stats["requires"] = {"%s/%s" % (n, v): v for n, v in requires}
    installed = {}
    conan_home = str(ctx.root / "build" / "conan_home")
    conan = which(ctx, "conan")
    rc, out, err = (1, "", "not found")
    if conan:
        rc, out, err = runv(ctx, [str(conan), "list", "*", "--format=json"],
                           timeout=60, env={"CONAN_HOME": conan_home})
    conan_refs = []
    if rc == 0 and out.strip():
        try:
            data = json.loads(out)
        except Exception:
            data = {}

        def harvest(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if re.match(r"^[A-Za-z0-9_+\-]+/\d", k):
                        conan_refs.append(k)
                    harvest(v)
            elif isinstance(o, list):
                for v in o:
                    harvest(v)
        harvest(data)
    stats["conan_list_ok"] = rc == 0
    stats["installed_refs"] = sorted(conan_refs)
    for ref in conan_refs:
        name, _, ver = ref.partition("/")
        installed.setdefault(name, set()).add(ver.split("@")[0])
    # CMakeDeps version files as corroboration
    build = ctx.root / "build"
    for pkg in sorted({n for n, _v in requires}):
        for c in list(build.glob("%s-config-version.cmake" % pkg)) + \
                list(build.glob("%s-config-version.cmake" % pkg.lower())):
            t = read_text(c, 65536)
            mm = re.search(r'PACKAGE_VERSION\s+"?([0-9][^"\s]*)"?', t)
            if mm:
                installed.setdefault(pkg, set()).add(mm.group(1))
                break
    stats["installed"] = {k: sorted(v) for k, v in installed.items()}
    for pkg, want in requires:
        got = installed.get(pkg)
        if not got:
            findings.append(dict(severity="warn",
                                 message="required package %s/%s is not present in the conan cache" % (pkg, want),
                                 evidence=["conan list '*' with CONAN_HOME=build/conan_home did not report %s" % pkg,
                                           "a rebuild would fetch/resolve a different version"]))
        elif want not in got:
            findings.append(dict(severity="warn",
                                 message="conanfile requires %s/%s but cache has %s" % (pkg, want, ", ".join(sorted(got))),
                                 evidence=["installed: " + ", ".join("%s/%s" % (pkg, v) for v in sorted(got))]))
    extras = sorted(set(installed) - {n for n, _v in requires})
    stats["cache_only_packages"] = extras
    return findings, stats


@rule("R1.3", "R1", "error",
      "CUDA toolkit version vs driver (UMD) capability",
      "The repo ships nvcc 13.x on this box.  A toolkit newer than what the driver "
      "(UMD) supports compiles kernels that fail at runtime with 'no kernel image is "
      "available' - a version problem that only shows up on the GPU box.")
def r13_cuda(ctx: Ctx):
    findings, stats = [], {}
    nvcc = which(ctx, "nvcc")
    if not nvcc:
        cands = sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe"))
        nvcc = Path(cands[-1]) if cands else None
    if not nvcc:
        return findings, {"skipped": "nvcc not found"}
    stats["nvcc"] = str(nvcc)
    rc, out, err = runv(ctx, [str(nvcc), "--version"], timeout=25)
    um = re.search(r"release\s+(\d+\.\d+)", out + err)
    toolkit = um.group(1) if um else None
    stats["cuda_toolkit"] = toolkit
    drv_ver, umd = None, None
    if not ctx.args.no_subprocess:
        smi = which(ctx, "nvidia-smi") or Path(r"C:\Windows\System32\nvidia-smi.exe")
        rc2, out2, _err2 = runv(ctx, [str(smi)], timeout=25) if Path(str(smi)).exists() else (1, "", "")
        if rc2 == 0:
            mm = re.search(r"CUDA UMD Version:\s*(\d+\.\d+)", out2)
            umd = mm.group(1) if mm else None
            mm = re.search(r"(?:Driver Version|KMD Version):\s*([0-9.]+)", out2)
            drv_ver = mm.group(1) if mm else None
            gpu = re.search(r"(NVIDIA[^|]+?)\s+WDDM", out2)
            stats["gpu"] = gpu.group(1).strip() if gpu else None
    stats["driver"] = drv_ver
    stats["driver_cuda_umd"] = umd
    if toolkit and umd:
        try:
            tk = tuple(int(x) for x in toolkit.split("."))
            ud = tuple(int(x) for x in umd.split("."))
            if ud < tk:
                findings.append(dict(
                    severity="error",
                    message="CUDA toolkit %s is newer than driver UMD %s (driver %s)"
                            % (toolkit, umd, drv_ver),
                    evidence=[str(nvcc), "toolkit release %s" % toolkit,
                              "nvidia-smi CUDA UMD Version: %s (driver %s)" % (umd, drv_ver)]))
        except ValueError:
            pass
    elif toolkit and not umd:
        stats["note"] = "could not parse nvidia-smi UMD version; toolkit only"
    if stats.get("gpu") and "1660" in (stats.get("gpu") or ""):
        stats["note_gpu"] = ("GTX 1660 SUPER = Turing sm_75; CUDA 13 still supports sm_75, "
                             "but Toolkit >= 13.0 dropped sm_50/60/70 - do not target older arches")
    return findings, stats


def sha256_file(path: Path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def git_status_paths(text: str):
    """Parse `git status --porcelain` into a set of worktree paths."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rest = line[3:] if len(line) >= 3 else line
        if " -> " in rest:
            rest = rest.split(" -> ", 1)[1]
        rest = rest.strip()
        if rest.startswith('"') and rest.endswith('"'):
            rest = rest[1:-1]
        if rest:
            out.append(rest.replace("\\", "/"))
    return out


@rule("R1.4", "R1", "error",
      "llama.cpp vendor reproducibility (canonical patch + hash truth + dirty-file set)",
      "The gateway's split backend depends on 10 hand-modified llama.cpp files.  The old "
      "per-feature patch chain drifted (2 corrupt patches, 5 conflicting) and could not "
      "reproduce the live tree - the same class as the 'patch defect' incident where the "
      "enc-dec split path leaked template tokens and GGML_ASSERT'ed under load. "
      "Authoritative truth is now llama_server_mods/phoenix_vendor_733dd8df.patch + "
      "vendor_733dd8df.sha256.json (doc/v10.1/vendor_patch_stack.md).")
def r14_llamacpp_patches(ctx: Ctx):
    findings, stats = [], {}
    root = ctx.root
    mod = root / "llama_server_mods"
    repo = root / "outsides" / "llamacpp"
    truth_path = mod / "vendor_733dd8df.sha256.json"
    truth = load_json(truth_path) or {}
    canonical_rel = truth.get("canonicalPatch") or "llama_server_mods/phoenix_vendor_733dd8df.patch"
    canonical = root / canonical_rel
    recorded = sorted({f.get("path") for f in truth.get("files", [])
                       if isinstance(f, dict) and f.get("path")})
    stats["truth_file"] = rel(ctx, truth_path)
    stats["truth_head"] = truth.get("head")
    stats["recorded_files"] = len(recorded)
    stats["canonical_patch"] = rel(ctx, canonical) if canonical.exists() else (canonical_rel + " (MISSING)")
    if canonical.exists():
        stats["canonical_bytes"] = canonical.stat().st_size
        stats["canonical_sha256"] = sha256_file(canonical)
    if not (repo / ".git").exists():
        return findings, {"skipped": "outsides/llamacpp is not a git repo"}

    # superseded per-feature patches (kept until Lead approves deletion) are exempt
    sup_text = read_text(mod / "SUPERSEDED.md")
    superseded = set()
    for m in re.finditer(r"`([A-Za-z0-9_./-]+\.patch)`", sup_text):
        name = m.group(1)
        if "/" not in name:  # root-level old patches; regenerated/ is not part of the gate
            superseded.add(name)
    stats["superseded_patches"] = sorted(superseded)

    rc, out, err = runv(ctx, ["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=20)
    if rc != 0:
        return findings, {"skipped": "git rev-parse failed: %s" % (err.strip()[:120])}
    stats["live_vendor_head"] = out.strip()

    # --- R1.4a: canonical patch applies to pristine base and reproduces hash truth ---
    verify_script = root / "tools" / "vendor_patch_verify.ps1"
    last_path = root / "build" / "tmp" / "vendor_patch_verify" / "last_result.json"
    last = load_json(last_path)
    verify = "unknown"
    if not ctx.args.no_subprocess and verify_script.is_file():
        sw_start = time.monotonic()
        vrc, vout, verr = runv(ctx, ["pwsh", "-NoProfile", "-File", str(verify_script)],
                               timeout=240)
        stats["verify_exit"] = vrc
        stats["verify_elapsed_s"] = round(time.monotonic() - sw_start, 2)
        vm = re.search(r"RESULT:\s*(PASS|FAIL)", vout)
        verify = vm.group(1) if vm else ("PASS" if vrc == 0 else "FAIL")
        result_lines = [l.strip() for l in vout.splitlines() if "RESULT:" in l or "sha256" in l]
        stats["verify_source"] = rel(ctx, verify_script)
        if verify != "PASS":
            findings.append(dict(
                severity="error",
                message="R1.4a FAIL: canonical patch does not reproduce the recorded vendor tree",
                evidence=[(result_lines[-1] if result_lines else (vout or verr).strip()[-300:]),
                          rel(ctx, truth_path)]))
    elif last is not None:
        verify = "PASS" if last.get("pass") else "FAIL"
        stats["verify_source"] = rel(ctx, last_path) + " (cached; --no-subprocess)"
        if verify != "PASS":
            findings.append(dict(severity="error",
                                 message="R1.4a FAIL: cached vendor verification result is not PASS",
                                 evidence=[rel(ctx, last_path)]))
    else:
        stats["verify_source"] = "not run"
        findings.append(dict(severity="warn",
                             message="R1.4a could not run: tools/vendor_patch_verify.ps1 unavailable and "
                                     "no cached result (use pwsh, not --no-subprocess)",
                             evidence=[rel(ctx, verify_script)]))
    stats["verify"] = verify

    # --- R1.4b: non-superseded *.patch files must still apply cleanly ---
    exempt = superseded | {Path(canonical_rel).name}
    checked, needs_fix, details = [], [], {}
    if not ctx.args.no_subprocess:
        for patch in sorted(mod.glob("*.patch")):
            if patch.name in exempt:
                continue
            checked.append(patch.name)
            try:
                relp = os.path.relpath(str(patch), str(repo))
            except ValueError:
                relp = str(patch)
            fwd = runv(ctx, ["git", "-C", str(repo), "apply", "--check",
                             "--ignore-space-change", "--ignore-whitespace", relp], timeout=20)
            rev = runv(ctx, ["git", "-C", str(repo), "apply", "--check", "--reverse",
                             "--ignore-space-change", "--ignore-whitespace", relp], timeout=20)
            f_rc, _f_out, f_err = fwd
            r_rc, _r_out, _r_err = rev
            if f_rc != 0 and r_rc != 0:
                needs_fix.append(patch.name)
                first = (f_err or "").strip().splitlines()
                details[patch.name] = first[0][:180] if first else "does not apply either way"
    stats["checked_non_superseded"] = checked
    stats["needs_fix"] = needs_fix
    if needs_fix:
        findings.append(dict(
            severity="error",
            message="R1.4b FAIL: %d non-superseded patch(es) do not apply cleanly (add to "
                    "SUPERSEDED.md only if they are truly obsolete)" % len(needs_fix),
            evidence=[p + ": " + details.get(p, "") for p in needs_fix[:10]]))

    # --- R1.4c: dirty file set must equal the recorded set ---
    if not ctx.args.no_subprocess:
        src, sout, serr = runv(ctx, ["git", "-C", str(repo), "status", "--porcelain"], timeout=20)
        if src != 0:
            findings.append(dict(severity="warn",
                                 message="R1.4c could not read git status: %s" % serr.strip()[:120],
                                 evidence=[str(repo)]))
        else:
            actual = set(git_status_paths(sout))
            missing = sorted(set(recorded) - actual)
            extra = sorted(actual - set(recorded))
            stats["dirty_files"] = len(actual)
            stats["missing_from_worktree"] = missing
            stats["unexpected_dirty"] = extra
            if missing or extra:
                findings.append(dict(
                    severity="error",
                    message="R1.4c FAIL: live dirty-file set differs from vendor_733dd8df.sha256.json "
                            "(%d missing, %d unexpected)" % (len(missing), len(extra)),
                    evidence=(["missing: " + p for p in missing[:8]] +
                              ["unexpected: " + p for p in extra[:8]] +
                              [rel(ctx, truth_path)])))
    return findings, stats


@rule("R1.5", "R1", "error",
      "Python314 headers/libs vs embedded runtime DLL",
      "The exe embeds Python314/libpython and talks to Python314; a header/DLL patch "
      "mismatch (e.g. 3.14.3 headers + 3.14.1 DLL) corrupts PyObject layouts at runtime. "
      "This machine has a second Python 3.11 on PATH, so the resolved interpreter must "
      "also be the embedded one.")
def r15_python314(ctx: Ctx):
    findings, stats = [], {}
    root = ctx.root
    inc = root / "Python314" / "include"
    dll = root / "Python314" / "python314.dll"
    lib = root / "Python314" / "libs" / "python314.lib"
    ptext = read_text(inc / "patchlevel.h") if inc.exists() else ""
    m = re.search(r'#define\s+PY_VERSION\s+"([^"]+)"', ptext)
    header_ver = m.group(1) if m else None
    stats["header_python_version"] = header_ver
    stats["include_header"] = str(inc / "Python.h")
    stats["python_h_present"] = (inc / "Python.h").is_file()
    stats["lib_present"] = lib.is_file()
    dll_ver = file_version(dll) if dll.exists() else None
    stats["dll_file_version"] = dll_ver
    if not header_ver:
        findings.append(dict(severity="error",
                             message="Python314/include/patchlevel.h is missing/unreadable (PY_VERSION not found)",
                             evidence=[str(inc / "patchlevel.h")]))
    if not dll.exists():
        findings.append(dict(severity="error", message="Python314/python314.dll missing",
                             evidence=[str(dll)]))
    if header_ver and dll_ver:
        hmajmin = ".".join(header_ver.split(".")[:2])
        dmajmin = ".".join(dll_ver.split(".")[:2])
        # CPython Windows DLLs use a build-number version resource
        # (e.g. 3.14.3150.1013 for CPython 3.14.x), so only major.minor is
        # comparable; the real patch level is python.exe's sys.version.
        stats["dll_version_scheme"] = "windows-build-number (major.minor comparable only)"
        if hmajmin != dmajmin:
            findings.append(dict(severity="error",
                                 message="Python header version %s does not match DLL %s" % (header_ver, dll_ver),
                                 evidence=[str(inc / "patchlevel.h"), str(dll)]))
        elif dll_ver.split(".")[2] not in ("0",) and header_ver.split(".")[2] != "0":
            stats["patch_check"] = ("DLL resource %s carries a Windows build number; "
                                    "patch level is verified via python.exe below" % dll_ver)
    runtime = root / "Python314" / "python.exe"
    stats["embedded_python_exe"] = runtime.is_file()
    if runtime.is_file():
        rc, out, err = runv(ctx, [str(runtime), "--version"], timeout=20)
        stats["embedded_python_version"] = (out or err).strip()[:80]
        if rc == 0 and header_ver:
            vm = re.search(r"(\d+)\.(\d+)\.(\d+)", out + err)
            if vm and vm.group(0) != header_ver:
                findings.append(dict(severity="warn",
                                     message="embedded python.exe reports %s but headers say %s"
                                             % (vm.group(0), header_ver),
                                     evidence=[str(runtime), (out + err).strip()[:120]]))
    return findings, stats


@rule("R1.6", "R1", "warn",
      "ONNX Runtime version vs local_onnx.cpp usage",
      "local_onnx.cpp is a subprocess bridge to Python, so the effective ORT is the "
      "Python314 site-package (not a C++ C API).  A missing/old onnxruntime silently "
      "degrades every audio/video model to the deterministic fallback.")
def r16_onnx(ctx: Ctx):
    findings, stats = [], {}
    src = read_text(ctx.root / "local_onnx.cpp")
    uses_c_api = bool(re.search(r"onnxruntime_c_api|OrtGetApiBase|OrtCreateEnv", src))
    stats["local_onnx_uses_c_api"] = uses_c_api
    stats["bridge"] = "python-subprocess" if not uses_c_api else "C++ C-API"
    site = ctx.root / "Python314" / "Lib" / "site-packages" / "onnxruntime"
    stats["onnxruntime_package"] = str(site)
    stats["onnxruntime_present"] = site.is_dir()
    ort_ver = None
    # The effective runtime is the embedded interpreter; ask it directly.
    pyexe = ctx.root / "Python314" / "python.exe"
    if pyexe.is_file() and not ctx.args.no_subprocess:
        rc, out, err = runv(ctx, [str(pyexe), "-c",
                                 "import onnxruntime as o; print(o.__version__)"],
                           timeout=40)
        if rc == 0 and out.strip():
            ort_ver = out.strip().splitlines()[-1].strip()
    if not ort_ver:
        for cand in ("version.py", "capi/version.py", "_version.py", "capi/_pybind_state.py"):
            p = site / cand
            if p.is_file():
                t = read_text(p, 262144)
                m = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", t) or \
                    re.search(r"""version\s*=\s*["']([0-9][^"']+)["']""", t)
                if m:
                    ort_ver = m.group(1)
                    break
    stats["onnxruntime_version"] = ort_ver
    if not site.is_dir():
        findings.append(dict(severity="warn",
                             message="Python314 site-packages/onnxruntime not found; "
                                     "every local ONNX model will fall back to deterministic mode",
                             evidence=[str(site), "local_onnx.cpp runs the embedded Python bridge"]))
    elif not ort_ver:
        findings.append(dict(severity="info",
                             message="onnxruntime package present but version not parseable",
                             evidence=[str(site)]))
    return findings, stats


@rule("R1.7", "R1", "warn",
      "version.hpp vs documentation/version shelf",
      "version.hpp is what /api/system/status reports; README/doc are what humans and "
      "scripts believe.  Divergence is how 'which build am I running?' ambiguity starts "
      "(the user's explicit 'do not die of version problems' ask).")
def r17_versions(ctx: Ctx):
    findings, stats = [], {}
    vtext = read_text(ctx.root / "version.hpp")
    major = re.search(r"PHOENIX_VERSION_MAJOR\s*=\s*(\d+)", vtext)
    minor = re.search(r"PHOENIX_VERSION_MINOR\s*=\s*(\d+)", vtext)
    patch = re.search(r"PHOENIX_VERSION_PATCH\s*=\s*(\d+)", vtext)
    codename = re.search(r'PHOENIX_CODENAME\[\]\s*=\s*"([^"]+)"', vtext)
    ver = "%s.%s.%s" % (major.group(1), minor.group(1), patch.group(1)) if major else None
    stats["version_hpp"] = ver
    stats["codename"] = codename.group(1) if codename else None
    docs = sorted([p.name for p in (ctx.root / "doc").glob("v*") if p.is_dir()],
                  key=lambda s: [int(x) for x in re.findall(r"\d+", s)] or [0])
    stats["doc_version_shelves"] = docs
    readme = read_text(ctx.root / "README.md", 200000)
    rm = re.search(r"#\s*Phoenix\s+v([0-9][^\s\"]*)", readme)
    stats["readme_title_version"] = rm.group(1) if rm else None
    doc_readme = read_text(ctx.root / "doc" / "README.md", 200000)
    stats["doc_readme_has_version_shelf"] = bool(doc_readme)
    if ver and docs:
        newest = docs[-1].lstrip("v")
        if newest != ver:
            findings.append(dict(severity="warn",
                                 message="version.hpp says %s but newest doc shelf is %s" % (ver, docs[-1]),
                                 evidence=[evidence(ctx, ctx.root / "version.hpp", None,
                                                    "PHOENIX_VERSION_STRING=%s" % ver),
                                           "doc/%s/ exists (docs updated after the version bump?)" % docs[-1]]))
    manifest = load_json(ctx.root / "manifest.json")
    stats["manifest"] = manifest
    if ver and rm:
        rm_ver = rm.group(1).rstrip('"')
        def majmin(v):
            parts = re.findall(r"\d+", v)
            return ".".join(parts[:2])
        if majmin(rm_ver) != majmin(ver):
            findings.append(dict(severity="warn",
                                 message="README title version %s differs from version.hpp %s"
                                         % (rm_ver, ver),
                                 evidence=[evidence(ctx, ctx.root / "README.md", None,
                                                    "# Phoenix v%s" % rm_ver)]))
    return findings, stats


@rule("R1.8", "R1", "error",
      "config switches restored from persisted state (config-vs-state class)",
      "runtime_store/autonomy_state.json persists enabled-like switches (agiEnabled, "
      "missionEnabled, study.expert.enabled, ...).  importState/fromJson restores them, "
      "so editing config/phoenix.json is NOT enough after a restart - the stale state "
      "silently re-enables/disables modules.  We already found 4+ such switches.")
def r18_config_vs_state(ctx: Ctx):
    findings, stats = [], {}
    cfg = load_json(ctx.root / "config" / "phoenix.json")
    state = load_json(ctx.root / "runtime_store" / "autonomy_state.json")
    if cfg is None:
        findings.append(dict(severity="error", message="config/phoenix.json missing/unparsable",
                             evidence=[str(ctx.root / "config" / "phoenix.json")]))
        return findings, stats
    if state is None:
        return findings, {"state": "runtime_store/autonomy_state.json absent (long run not persisted yet)"}

    def enabled_paths(obj, prefix=""):
        out = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                q = (prefix + "." + k) if prefix else k
                if k == "enabled" or k.endswith("Enabled"):
                    out.append(q)
                out.extend(enabled_paths(v, q))
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:20]):
                out.extend(enabled_paths(v, prefix + "[%d]" % i))
        return out

    cfg_keys = enabled_paths(cfg)
    st_keys = enabled_paths(state)
    cfg_norm = {re.sub(r"[^a-z0-9]", "", k.lower()): k for k in cfg_keys}
    overlaps = []
    for sk in st_keys:
        norm = re.sub(r"[^a-z0-9]", "", sk.lower())
        if norm in cfg_norm:
            overlaps.append((sk, cfg_norm[norm]))
    stats["config_enabled_keys"] = len(cfg_keys)
    stats["state_enabled_keys"] = len(st_keys)
    stats["state_top_keys"] = sorted(state.keys())[:25]
    # de-duplicate: one normalized config key can be hit by two state spellings
    seen = set()
    overlaps = []
    for sk, ck in [(sk, cfg_norm.get(re.sub(r"[^a-z0-9]", "", sk.lower()))) for sk in st_keys]:
        if not ck or ck in seen:
            continue
        seen.add(ck)
        overlaps.append((sk, ck))
    stats["overlaps"] = overlaps[:20]
    if overlaps:
        findings.append(dict(
            severity="error",
            message="%d persisted state switch(es) mirror config switches and are restored on startup"
                    % len(overlaps),
            evidence=["%s <- config %s" % (a, b) for a, b in overlaps[:12]]
                     + [rel(ctx, ctx.root / "runtime_store" / "autonomy_state.json"),
                        "fix: delete/reconcile the persisted key or gate restore on a schema/flag"]))
    return findings, stats


# --------------------------------------------------------------------------
# R2 - DEAD CODE / DATA
# --------------------------------------------------------------------------
def build_source_lists(ctx: Ctx):
    root = ctx.root
    listed = {}
    comp = read_text(root / "build" / "compile_sources.txt")
    for line in comp.splitlines():
        line = line.strip()
        if line:
            listed[line.replace("/", "\\").lower()] = "build/compile_sources.txt"
    bat = read_text(root / "compile.bat")
    m = re.search(r'set\s+"COMMON_SOURCES=(.*?)"', bat, re.S)
    if m:
        for tok in re.findall(r"[^\s]+\.(?:cpp|cc|cxx)", m.group(1)):
            if tok.startswith("%"):
                continue
            listed[tok.replace("/", "\\").lower()] = "compile.bat COMMON_SOURCES"
    for other in ("compile_gtest.bat", "tools/build_rdk_x5.sh"):
        t = read_text(root / other)
        for tok in re.findall(r"[A-Za-z0-9_./\\\-]+\.cpp", t):
            listed[os.path.basename(tok).lower()] = other
    return listed


def huge_skipped(p: Path) -> bool:
    try:
        return p.stat().st_size > 8 * 1024 * 1024
    except OSError:
        return False


@rule("R2.0", "R2", "info",
      "deadcode-audit asset classification consumed (advisory)",
      "deadcode-audit owns the 20.9GB leftover classification and the deletion plan. "
      "R2 only consumes its JSON; this checker never duplicates or executes deletions.")
def r20_dead_assets(ctx: Ctx):
    findings, stats = [], {}
    path = ctx.root / "build" / "tmp" / "dead_assets.json"
    stats["path"] = rel(ctx, path)
    if not path.is_file():
        stats["state"] = "not published yet - R2 falls back to heuristics"
        return findings, stats
    data = load_json(path)
    if not isinstance(data, dict):
        findings.append(dict(severity="warn",
                             message="build/tmp/dead_assets.json exists but is not a JSON object",
                             evidence=[rel(ctx, path)]))
        return findings, stats
    assets = data.get("assets") or data.get("items") or []
    stats["assets"] = len(assets)
    summary = data.get("summary") or {}
    stats["summary"] = summary
    kinds = {}
    missing = []
    for a in assets[:5000]:
        if not isinstance(a, dict):
            continue
        kinds[a.get("verdict", "unknown")] = kinds.get(a.get("verdict", "unknown"), 0) + 1
        p = a.get("path")
        if p and not (ctx.root / p).exists() and not Path(p).exists():
            missing.append(p)
    stats["verdicts"] = kinds
    if missing:
        findings.append(dict(severity="warn",
                             message="%d asset path(s) in dead_assets.json no longer exist (stale audit)"
                                     % len(missing),
                             evidence=missing[:8]))
    return findings, stats


@rule("R2.1", "R2", "warn",
      "Product sources/headers not reachable from the main build lists",
      "The build is driven by build/compile_sources.txt + compile.bat COMMON_SOURCES; "
      "anything else is only reachable from tools/tests.  Heuristic only: a file can be "
      "a deliberate stub (RDK), a separate binary, or included via a generated list - "
      "verify with deadcode-audit before deleting anything.")
def r21_unreachable(ctx: Ctx):
    findings, stats = [], {}
    root = ctx.root
    listed = build_source_lists(ctx)
    main_text = read_text(root / "main.cpp")
    main_includes = set(re.findall(r'#include\s+"([^"]+)"', main_text))
    bats = read_text(root / "compile.bat") + "\n" + read_text(root / "compile_gtest.bat") + \
        "\n" + read_text(root / "tools/build_rdk_x5.sh")
    intentional = {"transformer_main_x5_stub.cpp", "transformer_ollama_ft_stub.cpp",
                   "mcu_posix_compat.cpp", "mcu_wolfssh_adapter.cpp"}
    candidates = []
    for p in sorted(root.glob("*.cpp")):
        key = p.name.lower()
        if key in listed or key in intentional:
            continue
        if p.name in main_includes:
            continue
        if p.name.lower() in bats.lower():
            continue
        candidates.append(p)
    stats["root_cpp_count"] = len(list(root.glob("*.cpp")))
    stats["declared_sources"] = len(listed)
    stats["unreachable_cpp"] = [rel(ctx, p) for p in candidates]
    if candidates:
        findings.append(dict(severity="warn",
                             message="%d root .cpp file(s) are not in the main build lists, "
                                     "main.cpp, the gtest build, or the RDK script" % len(candidates),
                             evidence=[rel(ctx, p) for p in candidates[:25]]))
    # headers: referenced by any compiled source / main / hub includes
    compiled = [p for p in product_source_files(ctx) if p.suffix.lower() in (".cpp", ".inc", ".hpp", ".h")]
    included = set()
    inc_re = re.compile(r'#include\s+["<]([^">]+)[">]')
    for p in compiled:
        if huge_skipped(p) and p.suffix.lower() == ".cpp":
            continue
        for m in inc_re.finditer(read_text(p, 2 * 1024 * 1024)):
            inc = m.group(1)
            included.add(os.path.basename(inc).lower())
            included.add(inc.replace("\\", "/").lower())
    orphan_headers = []
    for p in sorted(root.glob("*.hpp")):
        if p.name.lower() in included:
            continue
        if p.name.lower() in bats.lower():
            continue
        orphan_headers.append(p)
    stats["orphan_headers_count"] = len(orphan_headers)
    stats["orphan_headers"] = [rel(ctx, p) for p in orphan_headers[:30]]
    if orphan_headers:
        findings.append(dict(severity="info",
                             message="%d root header(s) are not #included by any product source "
                                     "(heuristic; generated/one-off headers may be expected)"
                                     % len(orphan_headers),
                             evidence=[rel(ctx, p) for p in orphan_headers[:20]]))
    return findings, stats


@rule("R2.2", "R2", "info",
      "Config keys whose leaf identifier never appears in product sources",
      "Every key in config/phoenix.json should be read by cfgOr/resolveConfig/parse code. "
      "Unread keys are the 'config dead keys' class (v8.0 audit: 550->503 keys).  Heuristic: "
      "keys built dynamically or read only via dotted strings may be false positives.")
def r22_config_unread(ctx: Ctx):
    findings, stats = [], {}
    cfg = load_json(ctx.root / "config" / "phoenix.json")
    if cfg is None:
        return findings, {"skipped": "config/phoenix.json unreadable"}
    leaves = []

    def walk(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                q = (p + "." + k) if p else k
                if isinstance(v, (dict, list)):
                    walk(v, q)
                else:
                    leaves.append(q)
        elif isinstance(o, list):
            for i, v in enumerate(o[:5]):
                walk(v, p + "[%d]" % i)

    walk(cfg)
    ids = identifier_set(ctx)
    unread = []
    for key in leaves:
        leaf = key.split(".")[-1]
        leaf = re.sub(r"\[\d+\]$", "", leaf)
        if leaf in ids:
            continue
        unread.append(key)
    stats["config_leaves"] = len(leaves)
    stats["unread_by_leaf"] = len(unread)
    stats["source_identifiers"] = len(ids)
    if unread:
        findings.append(dict(severity="info",
                             message="%d/%d config leaf keys have a leaf identifier that never appears in product sources"
                                     % (len(unread), len(leaves)),
                             evidence=unread[:40]))
    return findings, stats


@rule("R2.3", "R2", "info",
      "Large runtime_store model/data files not referenced by config or a manifest",
      "These are the '20.9GB leftover' class.  deadcode-audit owns classification/safe "
      "deletion; this rule only flags files >=32 MiB whose basename appears in no config, "
      "manifest.json or nearby *.manifest.json.  Advisory only - never delete from here.")
def r23_models_unreferenced(ctx: Ctx):
    findings, stats = [], {}
    root = ctx.root / "runtime_store"
    if not root.is_dir():
        return findings, {"skipped": "runtime_store absent"}
    cfg_text = read_text(ctx.root / "config" / "phoenix.json")
    docs_blob = ""
    for p in (ctx.root / "doc").rglob("*.md"):
        if p.stat().st_size < 512 * 1024:
            docs_blob += read_text(p, 512 * 1024)
    exts = {".gguf", ".onnx", ".bin", ".safetensors", ".pt", ".pth", ".lmdb", ".db"}
    files = []
    for d, depth in [("runtime_store/models", 4), ("runtime_store", 2)]:
        base = ctx.root / d
        if base.is_dir():
            for f in walk_limited(base, max_depth=depth, max_files=1500, time_budget=6.0):
                if f.suffix.lower() in exts:
                    files.append(f)
    seen, selected = set(), []
    for f in files:
        try:
            key = str(f).lower()
            size = f.stat().st_size
        except OSError:
            continue
        if key in seen or size < 32 * 1024 * 1024:
            continue
        seen.add(key)
        selected.append((f, size))
    selected.sort(key=lambda x: -x[1])
    selected = selected[:120]
    unreferenced = []
    for f, size in selected:
        name = f.name
        if name in cfg_text or name in docs_blob:
            continue
        ref = False
        parent = f.parent
        for _ in range(3):
            for mf in list(parent.glob("*.json"))[:6]:
                if name in read_text(mf, 2 * 1024 * 1024):
                    ref = True
                    break
            if ref or parent == parent.parent:
                break
            parent = parent.parent
        if not ref:
            unreferenced.append((f, size))
    stats["candidate_files"] = len(selected)
    stats["unreferenced"] = len(unreferenced)
    stats["unreferenced_bytes"] = sum(s for _f, s in unreferenced)
    stats["note"] = "runtime_store is ~200GB on this box; scan is bounded (depth/time), not exhaustive"
    if unreferenced:
        findings.append(dict(
            severity="info",
            message="%d large runtime_store file(s) (%s) have no config/manifest reference in the bounded scan"
                    % (len(unreferenced), human(stats["unreferenced_bytes"])),
            evidence=["%s (%s)" % (rel(ctx, f), human(s)) for f, s in unreferenced[:15]]
                     + ["advisory only - deadcode-audit owns deletion; re-verify by loading path"]))
    return findings, stats


@rule("R2.4", "R2", "info",
      "Stale/debug binaries lying around",
      "Multiple phoenix_main variants (prev/next/dbg/edge_disabled) and gtest binaries in the "
      "repo root and build/tmp invite 'which binary produced this dump/behaviour' mistakes.")
def r24_stale_binaries(ctx: Ctx):
    findings, stats = [], {}
    root = ctx.root
    ref = root / "phoenix_main.exe"
    ref_mtime = ref.stat().st_mtime if ref.exists() else None
    entries = []
    for p in root.glob("phoenix_main*.exe"):
        entries.append(p)
    for p in root.glob("gtest_*.exe"):
        entries.append(p)
    for p in (root / "build" / "tmp").glob("phoenix_main*.exe"):
        entries.append(p)
    other_build_bins = len(list((root / "build" / "tmp").glob("*.exe"))) - len(entries)
    stale = []
    for p in entries:
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_size == 0:
            stale.append((p, st, "zero-byte"))
            continue
        if ref_mtime and p.resolve() != ref.resolve() and \
                (ref_mtime - st.st_mtime) > 7 * 86400:
            stale.append((p, st, "older than phoenix_main.exe by >7d"))
        elif any(tag in p.name for tag in ("prev", "dbg", "edge_disabled")):
            stale.append((p, st, "sidecar/obsolete variant"))
    stats["binaries_scanned"] = len(entries)
    stats["other_build_tmp_exes"] = max(0, other_build_bins)
    stats["stale"] = len(stale)
    if stale:
        findings.append(dict(
            severity="info",
            message="%d stale/sidecar phoenix/gtest binary file(s) (possible dump/behaviour mix-ups)"
                    % len(stale),
            evidence=["%s (%s, %s)" % (rel(ctx, p), human(st.st_size), why) for p, st, why in stale[:20]]))
    return findings, stats


# --------------------------------------------------------------------------
# R3 - LESSON GATES
# --------------------------------------------------------------------------
STATIC_CONTAINER_RE = re.compile(
    r"\bstatic\s+(?!const\b|constexpr\b|thread_local\b|inline\b|void\b|int\b|bool\b|"
    r"double\b|float\b|char\b|long\b|unsigned\b|size_t\b|auto\b|std::atomic\b)"
    r"(?:std::)?(?:vector|map|unordered_map|unordered_set|set|multimap|multiset|deque|"
    r"list|string|wstring|array|shared_ptr|unique_ptr)\b")


@rule("R3.1", "R3", "error",
      "Lambda/function-level static mutable containers",
      "0xC0000374 postmortem: explorePaths / namedRealPath / sectionDropStreak were "
      "function-level mutable statics mutated concurrently (push_back/sort/operator[] "
      "while another mission thread iterated) -> use-after-free -> heap metadata corruption "
      "detected minutes later inside ntdll.  Fix pattern: immutable magic static or a "
      "mutex-wrapped ScopeCounter.")
def r31_static_containers(ctx: Ctx):
    findings, stats = [], {}
    hits = []
    for p, i, line in iter_lines(ctx):
        s = line.strip()
        if s.startswith("//") or s.startswith("*"):
            continue
        m = STATIC_CONTAINER_RE.search(line)
        if not m:
            continue
        # exclude static const / constexpr even if regex missed spacing
        if re.search(r"\bstatic\s+const\b", line) or "constexpr" in line:
            continue
        # a static *function* returning a container is not a mutable static:
        #   static std::vector<uint8_t> buildWavHeader(...)  -> skip
        #   static std::string toString(...)                 -> skip
        #   static std::map<...> sectionDropStreak;          -> hit
        rest = line[m.end():].lstrip()
        if rest.startswith("<"):
            # regex consumed nothing of the template args; skip the '<' we are looking at
            rest = rest_after_template_args(rest[1:]).lstrip()
        ident = re.match(r"([A-Za-z_]\w*)", rest)
        if ident:
            after = rest[ident.end():].lstrip()
            if after.startswith("("):
                continue
        hits.append((p, i, line))
    stats["hits"] = len(hits)
    if hits:
        findings.append(dict(
            severity="error",
            message="%d mutable static container declaration(s) - concurrency rule from crash_forensics_2026-11-15"
                    % len(hits),
            evidence=[evidence(ctx, p, i, line) for p, i, line in hits[:60]]))
    return findings, stats


def looks_definition(s: str) -> bool:
    """Conservative one-line C++ function-definition header detector."""
    if not s or s.startswith(("#", "//", "*", "/*", "{", "}", "[", "(")):
        return False
    if s.startswith(("return ", "if ", "for ", "while ", "switch ", "do ", "else ",
                     "std::", "json::", "auto ", "case ", "using ", "typedef ",
                     "template ", "namespace ")):
        return False
    if "=" in s or ";" in s or "(" not in s:
        return False
    return re.match(
        r"^(?:(?:inline|static|virtual|explicit|constexpr|friend|extern)\s+)*"
        r"(?:[A-Za-z_][\w:<>\*&,\s]*\s+)?(?:[A-Za-z_]\w*::)*[A-Za-z_~]\w*\s*\(",
        s) is not None


def function_regions(code_lines):
    """Return [(start_idx, end_idx, name)] for function bodies (brace matched)."""
    n = len(code_lines)
    regions = []
    i = 0
    while i < n:
        s = code_lines[i].strip()
        if not looks_definition(s):
            i += 1
            continue
        brace_line = None
        j = i
        while j < min(n, i + 12):
            t = code_lines[j].strip()
            if "{" in t:
                brace_line = j
                break
            if t.endswith(";") or (j > i and ";" in t):
                break
            j += 1
        if brace_line is None:
            i += 1
            continue
        depth = 0
        end = None
        for k in range(brace_line, n):
            code = strip_code(code_lines[k])
            depth += code.count("{") - code.count("}")
            if depth <= 0:
                end = k
                break
        if end is None:
            i += 1
            continue
        regions.append((i, end, function_name_from_line(s)))
        i = end + 1
    return regions


def enclosing_functions(code_lines):
    """line index -> index of the enclosing function definition (or None)."""
    n = len(code_lines)
    enclosing = [None] * n
    for start, end, _name in function_regions(code_lines):
        for k in range(start, min(end + 1, n)):
            if enclosing[k] is None:
                enclosing[k] = start
    return enclosing


def strip_code(line: str) -> str:
    """Drop // comments and string literals (best effort, for brace counting)."""
    line = line.split("//", 1)[0]
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    return line


def strip_comments_text(text: str) -> str:
    """Remove // and /* */ comments while preserving line numbering."""
    def repl_block(m):
        return "\n" * m.group(0).count("\n")
    t = re.sub(r"/\*.*?\*/", repl_block, text, flags=re.S)
    t = re.sub(r"//[^\n]*", "", t)
    return t


def rest_after_template_args(rest: str) -> str:
    """Given text right after a consumed '<', return text after its matching '>'."""
    depth = 1
    for k, ch in enumerate(rest):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth == 0:
                return rest[k + 1:]
    return ""


def io_call_after_lock(lines, idx, max_lines=90):
    """Yield the first IO/tool call line inside the scope started at the lock line.

    Walks forward while brace depth allows, stops when the enclosing block closes
    or a new function definition begins.  Reduces the cross-function false
    positives of a plain line-window heuristic.
    """
    io_re = re.compile(r"https?://|curl|invokeAddonTool|callMcpTool|chatWith|runLocalOnnx|"
                       r"subprocess|spawnDetached|Spawn|generate\(|sendRequest|"
                       r"performHttp|tcpExchange|perform\s*\(")
    depth = 0
    started = False
    for j in range(idx + 1, min(len(lines), idx + max_lines)):
        raw = lines[j]
        s = raw.strip()
        if not s or s.startswith(("//", "*", "/*")):
            continue
        code = strip_code(raw)
        # a new function definition at depth 0 means we left the lock's function
        if depth == 0 and started and "(" in code and not s.endswith(";") and \
                ("::" in code or re.match(r"^[A-Za-z_][\w:<>\*&,\s]*\s+\w+\s*\(", s)):
            return None
        if io_re.search(code):
            return j
        depth += code.count("{") - code.count("}")
        if code.count("{"):
            started = True
        if depth < 0:
            return None
        if started and depth == 0:
            return None
    return None


def function_name_from_line(sig: str):
    m = re.search(r"(\w+)\s*\([^;{]*\)\s*(?:const)?\s*\{?\s*$", sig)
    if m:
        return m.group(1)
    # multi-line signature: last qualified identifier before '('
    m = re.search(r"(?:[A-Za-z_]\w*::)*([A-Za-z_]\w*)\s*\(", sig)
    return m.group(1) if m else None


def call_sites_locked(code_lines, enclosing, sig_line_idx, func_name, lock_re):
    """True if every call site of func_name sits in a function that holds a lock.

    Covers the documented 'caller holds the lock' pattern (e.g. pruneCache is
    only called from remember() under cacheMu_).  Returns None when there is no
    call site to judge.
    """
    if not func_name:
        return None
    call_re = re.compile(r"\b%s\s*\(" % re.escape(func_name))
    calls = []
    for j, line in enumerate(code_lines):
        if j == sig_line_idx:
            continue
        s = line.strip()
        if not s or s.startswith(("//", "*", "/*")):
            continue
        if not call_re.search(line):
            continue
        if looks_definition(s):
            continue
        fs = enclosing[j] if j < len(enclosing) else None
        start = fs if fs is not None else max(0, j - 80)
        region = "\n".join(code_lines[start:j + 1])
        calls.append(bool(lock_re.search(region)))
    if not calls:
        return None
    return all(calls)


@rule("R3.2", "R3", "error",
      "Container member written under a lock but read unlocked (same class)",
      "Postmortem class of bug: mcpActionMap_/cache_/sessions_-style containers are "
      "written under a mutex in one method and read without it elsewhere; the window is "
      "small and only fails under concurrent missions.  Heuristic: the nearest enclosing "
      "function is scanned backwards for a lock token; a site with no lock in its own body "
      "is only accepted as unlocked if its callers are not all locked either (caller-locked "
      "helpers and explicit 'caller holds the lock' comments are exempt).")
def r32_unlocked_member(ctx: Ctx):
    findings, stats = [], {}
    members = ["mcpActionMap_", "cache_", "sessions_"]
    lock_re = re.compile(r"lock_guard|unique_lock|scoped_lock|shared_lock|\.lock\(\)|"
                         r"mu_\.lock|mutex_?\.lock|lock\(", re.I)
    per_member = {}
    hits = {m: {"guarded": [], "unguarded": [], "unknown": []} for m in members}
    for p in product_source_files(ctx):
        text = read_text(p)
        if not text or not any(m in text for m in members):
            continue
        code_lines = strip_comments_text(text).splitlines()
        raw_lines = text.splitlines()
        enclosing = enclosing_functions(code_lines)
        for i, code in enumerate(code_lines):
            s = code.strip()
            if not s:
                continue
            for member in members:
                # only count *uses* (member access), never declarations like
                # `std::map<...> cache_;` or constructor init lists
                if not re.search(r"\b%s\s*[.\[]" % re.escape(member), code):
                    continue
                fs = enclosing[i] if i < len(enclosing) else None
                start = fs if fs is not None else max(0, i - 80)
                region = "\n".join(code_lines[start:i + 1])
                guarded = bool(lock_re.search(region))
                if not guarded:
                    # explicit contract comment: "Caller holds cacheMu_" etc.
                    pre = "\n".join(raw_lines[max(0, start - 4):start + 1]).lower()
                    if "caller holds" in pre or "caller must hold" in pre:
                        guarded = True
                if not guarded and fs is not None:
                    fn = function_name_from_line(code_lines[fs])
                    locked_callers = call_sites_locked(code_lines, enclosing, fs, fn, lock_re)
                    if locked_callers:
                        guarded = True
                raw = raw_lines[i] if i < len(raw_lines) else code
                if fs is None and not guarded:
                    hits[member]["unknown"].append((p, i + 1, raw))
                else:
                    bucket = "guarded" if guarded else "unguarded"
                    hits[member][bucket].append((p, i + 1, raw))
    for member in members:
        guarded = hits[member]["guarded"]
        unguarded = hits[member]["unguarded"]
        unknown = hits[member]["unknown"]
        per_member[member] = {"guarded": len(guarded), "unguarded": len(unguarded),
                              "unknown_enclosing_fn": len(unknown)}
        if guarded and unguarded:
            findings.append(dict(
                severity="error",
                message="%s: %d guarded site(s) but %d unguarded occurrence(s) in the same tree"
                        % (member, len(guarded), len(unguarded)),
                evidence=[evidence(ctx, p, i, line) for p, i, line in unguarded[:8]]))
    stats["members"] = per_member
    return findings, stats


@rule("R3.3", "R3", "error",
      "fromJson/importState restoring enabled-like switches",
      "v8.3 incident: exportState/importState round-trips contained enabled switches, so a "
      "restored runtime_store silently re-enabled modules the config had disabled (4+ found). "
      "Rule: assignments to enabled-like fields inside fromJson/importState/deserialize.",
      )
def r33_fromjson_switches(ctx: Ctx):
    findings, stats = [], {}
    func_re = re.compile(r"\b(?:\w+::)?(fromJson|importState|loadState|restoreState|deserialize)\s*\(")
    assign_re = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*[Ee]nabled\s*=|"
                           r"\benabled\s*=|"
                           r"""\[\s*["'][^"']*enabled[^"']*["']\s*\]\s*=""")
    hits = []
    scanned_funcs = 0
    for p in product_source_files(ctx):
        text = read_text(p)
        if not text or "enabled" not in text:
            continue
        code_lines = strip_comments_text(text).splitlines()
        raw_lines = text.splitlines()
        i = 0
        while i < len(code_lines):
            line = code_lines[i]
            if not func_re.search(line):
                i += 1
                continue
            scanned_funcs += 1
            depth = 0
            started = False
            j = i
            while j < len(code_lines) and j < i + 600:
                code = strip_code(code_lines[j])
                if started and assign_re.search(code):
                    raw = raw_lines[j] if j < len(raw_lines) else code_lines[j]
                    hits.append((p, j + 1, raw))
                depth += code.count("{") - code.count("}")
                if code.count("{"):
                    started = True
                if started and depth <= 0:
                    break
                j += 1
            i = max(i + 1, j)
    stats["hits"] = len(hits)
    stats["restore_functions_scanned"] = scanned_funcs
    stats["files"] = len({str(h[0]) for h in hits})
    if hits:
        findings.append(dict(
            severity="error",
            message="%d assignment(s) of enabled-like switches inside restore/deserialize function bodies "
                    "(%d files)" % (len(hits), stats["files"]),
            evidence=[evidence(ctx, p, i, line) for p, i, line in hits[:25]]))
    return findings, stats


@rule("R3.4", "R3", "warn",
      "#if 0 / commented-out includes in main.cpp / if(false) disabled paths",
      "Disabled code is invisible to build-time checks and to reviewers; main.cpp has "
      "commented-out hub includes 'due to pre-existing compilation errors'.  Each one is a "
      "latent feature gap or a stale fix that never landed (v8.0 debug audit lesson).")
def r34_disabled_code(ctx: Ctx):
    findings, stats = [], {}
    if0, commented_inc, false_if = [], [], []
    for p, i, line in iter_lines(ctx):
        s = line.strip()
        if re.match(r"#\s*if\s+0\b", s):
            if0.append((p, i, line))
        if re.match(r"//\s*#\s*include\b", s):
            commented_inc.append((p, i, line))
        if re.search(r"\bif\s*\(\s*false\s*\)", line) or re.search(r"\bif\s*\(\s*0\s*\)", line):
            false_if.append((p, i, line))
    stats.update(if0=len(if0), commented_includes=len(commented_inc), if_false=len(false_if))
    if commented_inc:
        findings.append(dict(
            severity="warn",
            message="%d commented-out #include line(s) (disabled hub fragments)" % len(commented_inc),
            evidence=[evidence(ctx, p, i, line) for p, i, line in commented_inc[:15]]))
    if if0:
        findings.append(dict(
            severity="warn",
            message="%d '#if 0' block(s) in product sources" % len(if0),
            evidence=[evidence(ctx, p, i, line) for p, i, line in if0[:15]]))
    if false_if:
        findings.append(dict(
            severity="warn",
            message="%d 'if (false)' / 'if (0)' branch(es)" % len(false_if),
            evidence=[evidence(ctx, p, i, line) for p, i, line in false_if[:15]]))
    return findings, stats


@rule("R3.5", "R3", "warn",
      "Mutex held across a network/tool/LLM call (heuristic)",
      "Coarse locks around IO are how mission ticks serialised the whole gateway; the "
      "unified-pipeline refactor had to move file IO outside mutexes explicitly.  Holding a "
      "lock over an HTTP/LLM/tool call risks priority inversion and deadlock during shutdown.")
def r35_lock_across_io(ctx: Ctx):
    findings, stats = [], {}
    lock_re = re.compile(r"lock_guard|unique_lock|scoped_lock|shared_lock|\.lock\(\)")
    hits = []
    for p in product_source_files(ctx):
        text = read_text(p)
        if not text or ("lock" not in text and "mutex" not in text):
            continue
        code_lines = strip_comments_text(text).splitlines()
        raw_lines = text.splitlines()
        for i, line in enumerate(code_lines):
            if not lock_re.search(line):
                continue
            j = io_call_after_lock(code_lines, i)
            if j is not None:
                raw_lock = raw_lines[i] if i < len(raw_lines) else line
                raw_call = raw_lines[j] if j < len(raw_lines) else code_lines[j]
                hits.append((p, i + 1, raw_lock, j + 1, raw_call))
    stats["hits"] = len(hits)
    if hits:
        findings.append(dict(
            severity="warn",
            message="%d lock site(s) with an IO/tool/LLM call inside the same brace scope" % len(hits),
            evidence=["%s:%d lock `%s` -> %s:%d `%s`"
                      % (rel(ctx, p), li, " ".join(lines_li.strip().split())[:80],
                         rel(ctx, p), lj, " ".join(lines_lj.strip().split())[:100])
                      for p, li, lines_li, lj, lines_lj in hits[:20]]))
    return findings, stats


# --------------------------------------------------------------------------
# runner / aggregation / output
# --------------------------------------------------------------------------
def run_rules(ctx: Ctx, only=None):
    results = []
    for r in RULES:
        if only and r["group"] not in only:
            continue
        t0 = time.monotonic()
        try:
            findings, stats = r["fn"](ctx)
            if findings is None:
                findings = []
            if stats is None:
                stats = {}
            skipped = isinstance(stats, dict) and ("skipped" in stats)
        except Exception as exc:  # noqa: BLE001 - a rule crash is a rule failure, not a tool crash
            findings = [dict(severity="error", message="rule crashed: %s: %s" % (type(exc).__name__, exc),
                             evidence=[])]
            stats = {}
            skipped = False
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        if skipped:
            status = "SKIP"
        elif any(f.get("severity") == "error" for f in findings) or (findings and r["severity"] == "error"):
            status = "FAIL"
        elif findings:
            status = "WARN"
        else:
            status = "PASS"
        # normalise finding severity
        for f in findings:
            f.setdefault("severity", r["severity"])
        results.append(dict(
            id=r["id"], group=r["group"], title=r["title"], severity=r["severity"],
            rationale=r["rationale"], status=status, elapsed_ms=elapsed,
            stats=stats, findings=findings,
        ))
    return results


def aggregate(results):
    groups = {}
    for r in results:
        g = groups.setdefault(r["group"], dict(status="PASS", rules=0, PASS=0, WARN=0,
                                                FAIL=0, SKIP=0, findings=0))
        g["rules"] += 1
        g[r["status"]] = g.get(r["status"], 0) + 1
        g["findings"] += len(r["findings"])
        if r["status"] == "FAIL":
            g["status"] = "FAIL"
        elif r["status"] == "WARN" and g["status"] != "FAIL":
            g["status"] = "WARN"
    totals = dict(PASS=sum(g["PASS"] for g in groups.values()),
                  WARN=sum(g["WARN"] for g in groups.values()),
                  FAIL=sum(g["FAIL"] for g in groups.values()),
                  SKIP=sum(g["SKIP"] for g in groups.values()),
                  findings=sum(g["findings"] for g in groups.values()),
                  rules=len(results))
    return groups, totals


SEV_RANK = {"error": 0, "warn": 1, "info": 2}


def top_findings(results, n=5):
    allf = []
    for r in results:
        for f in r["findings"]:
            allf.append((SEV_RANK.get(f.get("severity"), 3), r["id"], r["group"], f, r))
    allf.sort(key=lambda x: (x[0], x[1]))
    return allf[:n]


def build_markdown(ctx: Ctx, results, groups, totals, elapsed_s, extra):
    lines = []
    def w(s=""):
        lines.append(s)

    w("# Phoenix audit check - R1 version / R2 dead code-data / R3 lesson gates")
    w()
    w("- Generated: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    w("- Root: `%s`" % ctx.root)
    w("- Tool: `tools/audit_check.py` v%s (stdlib only, read-only on product sources)"
      % TOOL_VERSION)
    w("- Elapsed: %.1fs" % elapsed_s)
    w("- Rule groups: %s" % ", ".join(sorted(groups.keys())))
    w("- Subprocess probes: %s" % ("disabled (--no-subprocess)" if ctx.args.no_subprocess else "enabled"))
    if extra.get("git_head"):
        w("- git HEAD: `%s`" % extra["git_head"])
    w()
    w("## Group summary (PASS/FAIL)")
    w()
    w("| Group | Status | Rules | PASS | WARN | FAIL | SKIP | Findings |")
    w("|---|---|---|---|---|---|---|---|")
    for g in sorted(groups):
        d = groups[g]
        icon = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[d["status"]]
        w("| %s | **%s** | %d | %d | %d | %d | %d | %d |"
          % (g, icon, d["rules"], d["PASS"], d["WARN"], d["FAIL"], d["SKIP"], d["findings"]))
    w()
    w("Totals: %d rules, %d PASS, %d WARN, %d FAIL, %d SKIP, %d findings."
      % (totals["rules"], totals["PASS"], totals["WARN"], totals["FAIL"],
         totals["SKIP"], totals["findings"]))
    w()
    w("## Top findings")
    w()
    top = top_findings(results, n=extra.get("top", 5))
    if not top:
        w("_No findings._")
    for i, (rank, rid, group, f, r) in enumerate(top, 1):
        w("%d. **%s %s** (%s, severity=%s) - %s" % (i, rid, r["title"], group,
                                                    f.get("severity"), f.get("message")))
        for e in (f.get("evidence") or [])[:4]:
            w("   - `%s`" % str(e).replace("`", "'"))
        w()
    for group in sorted(groups):
        rs = [r for r in results if r["group"] == group]
        w("## %s" % group)
        w()
        w("| Rule | Status | Title | Findings |")
        w("|---|---|---|---|")
        for r in rs:
            w("| %s | %s | %s | %d |" % (r["id"], r["status"], r["title"], len(r["findings"])))
        w()
        for r in rs:
            w("### %s [%s] %s" % (r["id"], r["status"], r["title"]))
            w()
            w("_Incident encoded:_ %s" % r["rationale"])
            w()
            if r["stats"]:
                w("Stats: `%s`" % json.dumps(r["stats"], ensure_ascii=False)[:1200])
                w()
            if not r["findings"]:
                w("No findings.")
            for f in r["findings"]:
                w("- **%s** %s" % (f.get("severity"), f.get("message")))
                for e in f.get("evidence") or []:
                    w("  - `%s`" % str(e).replace("`", "'"))
            w()
    w("## Rerun")
    w()
    w("```powershell")
    w("cd %s" % ctx.root)
    w("python tools\\audit_check.py            # writes build\\tmp\\audit_check.{json,md}")
    w("python tools\\audit_check.py --only R1  # single group")
    w("python tools\\audit_check.py --no-subprocess --quiet")
    w("```")
    return "\n".join(lines) + "\n"


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Phoenix static audit checker (R1 versions, R2 dead code/data, R3 lesson gates)")
    ap.add_argument("--root", default=str(PROJECT_ROOT), help="project root (default: parent of tools/)")
    ap.add_argument("--json", default=None, help="JSON output path (default build/tmp/audit_check.json)")
    ap.add_argument("--md", default=None, help="markdown output path (default build/tmp/audit_check.md)")
    ap.add_argument("--only", default=None, help="comma-separated groups, e.g. R1,R3")
    ap.add_argument("--top", type=int, default=5, help="top findings in markdown (default 5)")
    ap.add_argument("--no-subprocess", action="store_true", help="skip git/nvcc/g++/nvidia-smi probes")
    ap.add_argument("--quiet", action="store_true", help="print one summary line only")
    ap.add_argument("--fail-on", choices=("error", "warn", "none"), default="error",
                    help="exit nonzero on FAIL (default), WARN, or never")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not (root / "compile.bat").exists() and not (root / "main.cpp").exists():
        sys.exit("not a Phoenix root: %s (expected compile.bat or main.cpp)" % root)
    json_path = Path(args.json) if args.json else root / "build" / "tmp" / "audit_check.json"
    md_path = Path(args.md) if args.md else root / "build" / "tmp" / "audit_check.md"
    only = None
    if args.only:
        only = {g.strip().upper() for g in args.only.split(",") if g.strip()}

    ctx = Ctx(root, args)
    t0 = time.monotonic()
    results = run_rules(ctx, only)
    elapsed = round(time.monotonic() - t0, 2)
    groups, totals = aggregate(results)

    extra = {"top": args.top}
    if not ctx.args.no_subprocess:
        res = run(ctx, ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], timeout=10)
        if res:
            rc, out, _err = res
            if rc == 0 and out.strip():
                extra["git_head"] = out.strip()

    payload = dict(
        tool="audit_check", tool_version=TOOL_VERSION,
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
        root=str(root), git_head=extra.get("git_head"),
        elapsed_s=elapsed, groups=groups, summary=totals,
        top_findings=[dict(id=rid, group=g, severity=f.get("severity"),
                           message=f.get("message"), evidence=(f.get("evidence") or [])[:5])
                      for _r, rid, g, f, _rule in top_findings(results, args.top)],
        rules=results,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(build_markdown(ctx, results, groups, totals, elapsed, extra))

    if args.quiet:
        print("audit_check: groups=%s rules=%d PASS=%d WARN=%d FAIL=%d SKIP=%d findings=%d elapsed=%.1fs"
              % (",".join("%s:%s" % (g, groups[g]["status"]) for g in sorted(groups)),
                 totals["rules"], totals["PASS"], totals["WARN"], totals["FAIL"],
                 totals["SKIP"], totals["findings"], elapsed))
    else:
        print("Phoenix audit check")
        for g in sorted(groups):
            d = groups[g]
            print("  %s: %-4s  rules=%d pass=%d warn=%d fail=%d skip=%d findings=%d"
                  % (g, d["status"], d["rules"], d["PASS"], d["WARN"], d["FAIL"],
                     d["SKIP"], d["findings"]))
        print("  totals: rules=%d PASS=%d WARN=%d FAIL=%d SKIP=%d findings=%d (%.1fs)"
              % (totals["rules"], totals["PASS"], totals["WARN"], totals["FAIL"],
                 totals["SKIP"], totals["findings"], elapsed))
        print("  json: %s" % json_path)
        print("  md:   %s" % md_path)
        for i, (_r, rid, g, f, _rule) in enumerate(top_findings(results, args.top), 1):
            print("  TOP%d [%s] %s" % (i, rid, f.get("message")))
    if args.fail_on == "none":
        return 0
    if args.fail_on == "warn":
        return 1 if (totals["WARN"] or totals["FAIL"]) else 0
    return 1 if totals["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
