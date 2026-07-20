#!/usr/bin/env python3
"""Incremental GTest runner build. Compiles changed source files to .o in
build/obj_gtest and links gtest_runner.exe. Reuses object files to avoid
recompiling unchanged heavy sources such as frontend_server.cpp."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OBJ_DIR = ROOT / "build" / "obj_gtest"
CONAN_CFLAGS_FILE = ROOT / "build" / "conan_cflags_gtest.txt"
CONAN_LIBS_FILE = ROOT / "build" / "conan_libs_gtest.txt"

COMMON_SOURCES = [
    "transformer_main.cpp",
    "transformer_ollama_fine_tuning.cpp",
    "addon.cpp",
    r"addons\builtin_registry.cpp",
    r"addons\math_addon.cpp",
    r"addons\search_addon.cpp",
    r"addons\computer_shell_addon.cpp",
    "loggerCXX.cpp",
    "DATABASE_079.cpp",
    "frontend_server.cpp",
    "speak_io.cpp",
    "model_lifecycle.cpp",
    "autonomy_stack.cpp",
    "v51_runtime.cpp",
    "external_runtime.cpp",
    "edge_platform.cpp",
    "gguf_tensor_parser.cpp",
    "physics_world_runtime.cpp",
    "emotion_system.cpp",
    "llamacpp_emotion_adjuster.cpp",
    "plugin_system.cpp",
    "modern_context_system.cpp",
    r"module_overrides\adversarial_learner_advanced.cpp",
    r"module_overrides\gnn_ga_learner_advanced.cpp",
    r"module_overrides\reinforcement_learner_advanced.cpp",
]

BULLET3_SOURCES = [
    r"outsides\bullet3\src\btLinearMathAll.cpp",
    r"outsides\bullet3\src\btBulletCollisionAll.cpp",
    r"outsides\bullet3\src\btBulletDynamicsAll.cpp",
]


def find_gxx():
    gcc_bin = Path(r"D:\Scoop\apps\gcc\current\bin")
    if gcc_bin.exists():
        os.environ["PATH"] = str(gcc_bin) + os.pathsep + os.environ.get("PATH", "")
    gxx = shutil.which("g++")
    if not gxx:
        print("[ERROR] g++ not found")
        sys.exit(1)
    return Path(gxx)


def ensure_conan_flags():
    if not CONAN_CFLAGS_FILE.exists() or not CONAN_LIBS_FILE.exists():
        print("[ERROR] conan flags missing; run compile_gtest.bat first")
        sys.exit(1)


def collect_sources():
    sources = [ROOT / s for s in COMMON_SOURCES]
    for s in BULLET3_SOURCES:
        p = ROOT / s
        if p.exists():
            sources.append(p)
    # All gtest unit/module sources
    for p in (ROOT / "tests" / "gtest").rglob("*.cpp"):
        sources.append(p)
    return [s.resolve() for s in sources if s.exists()]


def obj_path(src: Path) -> Path:
    rel = src.relative_to(ROOT)
    return OBJ_DIR / rel.with_suffix(".o")


def needs_compile(src: Path, obj: Path, flags_file: Path) -> bool:
    if not obj.exists():
        return True
    if src.stat().st_mtime > obj.stat().st_mtime:
        return True
    if flags_file.exists() and flags_file.stat().st_mtime > obj.stat().st_mtime:
        return True
    return False


def build_compile_args(src: Path, obj: Path, gxx: Path) -> list:
    py_local_root = ROOT / "Python314"
    py_inc = py_local_root / "include"
    py_lib = py_local_root / "libs"
    py_link_name = None
    for name in ["python314", "python3"]:
        if (py_lib / f"{name}.lib").exists() or (py_lib / f"lib{name}.a").exists():
            py_link_name = name
            break

    outsides_cflags = []
    for d in [ROOT / "outsides" / "llamacpp" / "include",
              ROOT / "outsides" / "BitNet" / "include"]:
        if d.exists():
            outsides_cflags.append(f"-I{d}")
    bullet_hdr = ROOT / "outsides" / "bullet3" / "src" / "btBulletDynamicsCommon.h"
    if bullet_hdr.exists():
        outsides_cflags.append(f"-I{ROOT / 'outsides' / 'bullet3' / 'src'}")

    includes = [
        f"-I{ROOT}",
        f"-I{ROOT / 'poppler-25.12.0' / 'Library' / 'include'}",
        f"-I{py_inc}",
        f"-I{ROOT / 'tests' / 'gtest'}",
    ] + outsides_cflags

    cmd = [
        str(gxx),
        "-std=c++20",
        "-Wa,-mbig-obj",
        "-DAI_EXTERNAL_BACKEND_COMPAT=1",
        "-DAI_EXTERNAL_LEARNER_BRIDGE=1",
        "-DHAVE_SQLITE",
        "-O0",
        "-g",
        f"@{CONAN_CFLAGS_FILE}",
    ] + includes + ["-c", str(src), "-o", str(obj)]
    return cmd


def build_link_args(objects: list, gxx: Path) -> list:
    py_local_root = ROOT / "Python314"
    py_lib = py_local_root / "libs"
    py_link_name = None
    for name in ["python314", "python3"]:
        if (py_lib / f"{name}.lib").exists() or (py_lib / f"lib{name}.a").exists():
            py_link_name = name
            break
    if not py_link_name:
        py_link_name = "python3"

    poppler_lib = ROOT / "poppler-25.12.0" / "Library" / "lib"
    cmd = [str(gxx), "-o", str(ROOT / "gtest_runner.exe")]
    cmd += [str(o) for o in objects]
    cmd += [
        "-Wl,--start-group",
        f"@{CONAN_LIBS_FILE}",
        "-Wl,--end-group",
        str(poppler_lib / "poppler-cpp.lib"),
        str(poppler_lib / "poppler.lib"),
        f"-L{py_lib}",
        f"-l{py_link_name}",
        "-lws2_32",
    ]
    return cmd


def main():
    import shutil
    gxx = find_gxx()
    ensure_conan_flags()
    OBJ_DIR.mkdir(parents=True, exist_ok=True)

    sources = collect_sources()
    objects = []
    compile_count = 0
    skip_count = 0

    for src in sources:
        obj = obj_path(src)
        obj.parent.mkdir(parents=True, exist_ok=True)
        if needs_compile(src, obj, CONAN_CFLAGS_FILE):
            print(f"[COMPILE] {src.relative_to(ROOT)}")
            cmd = build_compile_args(src, obj, gxx)
            try:
                subprocess.run(cmd, cwd=ROOT, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to compile {src}: {e}")
                sys.exit(1)
            compile_count += 1
        else:
            skip_count += 1
        objects.append(obj)

    exe = ROOT / "gtest_runner.exe"
    exe_mtime = exe.stat().st_mtime if exe.exists() else 0
    need_link = not exe.exists() or any(o.stat().st_mtime > exe_mtime for o in objects)
    if need_link:
        print("[LINK] gtest_runner.exe")
        cmd = build_link_args(objects, gxx)
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Link failed: {e}")
            sys.exit(1)
    else:
        print("[SKIP] gtest_runner.exe up to date")

    print(f"[SUCCESS] compiled {compile_count}, skipped {skip_count}")


if __name__ == "__main__":
    main()
