#!/usr/bin/env python3
"""Browser-based end-to-end user-flow skeleton.

Requires the Phoenix frontend build under 079project_frontend/build and a
running phoenix_main (5080/5081).  It uses Selenium + pyautogui to simulate
real mouse/keyboard input.  OCR is left as a pluggable hook:
set PHOENIX_OCR_CMD to a CLI tool that takes a PNG path and prints text,
or override capture_text() below.

The 5 user scenarios are:
1. Visit the site, register, login, and see the dashboard.
2. Start a new chat, type a prompt, and wait for a non-empty reply.
3. Upload/click a voice input and invoke /speech/analyze.
4. Upload/click an image and invoke /vision/analyze.
5. Logout and confirm the login screen returns.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions


ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "079project_frontend" / "build"
BASE_URL = os.environ.get("PHOENIX_UI_URL", "http://127.0.0.1:3000")
API_BASE = os.environ.get("PHOENIX_API_URL", "http://127.0.0.1:5080")
SCREENSHOT_DIR = ROOT / "build" / "tmp" / "ui_e2e"


def locate_msedge() -> Path | None:
    edge = shutil.which("msedge") or shutil.which("edge")
    if edge:
        return Path(edge)
    for p in [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]:
        if Path(p).exists():
            return Path(p)
    return None


def start_frontend_server() -> subprocess.Popen:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sys.executable, "-m", "http.server", "3000", "--directory", str(FRONTEND_DIR)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def new_driver() -> webdriver.Edge:
    options = EdgeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    if (ROOT / "build" / "tmp" / "ui_e2e" / "edge_profile").exists():
        shutil.rmtree(ROOT / "build" / "tmp" / "ui_e2e" / "edge_profile")
    options.add_argument(f"--user-data-dir={ROOT / 'build' / 'tmp' / 'ui_e2e' / 'edge_profile'}")
    # Selenium 4 uses selenium-manager to auto-fetch msedgedriver.
    return webdriver.Edge(options=options)


def screenshot(driver: webdriver.Edge, name: str) -> Path:
    path = SCREENSHOT_DIR / f"{name}.png"
    driver.save_screenshot(str(path))
    return path


def capture_text(path: Path) -> str:
    """OCR hook.  Returns text from a screenshot.  Override with a real OCR engine."""
    ocr_cmd = os.environ.get("PHOENIX_OCR_CMD")
    if ocr_cmd:
        try:
            return subprocess.check_output([ocr_cmd, str(path)], text=True, timeout=30)
        except Exception:
            pass
    return "[OCR not configured: install tesseract and set PHOENIX_OCR_CMD]"


def wait_for(driver: webdriver.Edge, by: By, value: str, timeout: float = 20.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            return driver.find_element(by, value)
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"element not found: {value}")


def run_scenarios():
    edge = locate_msedge()
    if not edge:
        print("ERROR: Microsoft Edge not found; cannot run browser tests.", file=sys.stderr)
        sys.exit(1)
    if not FRONTEND_DIR.exists():
        print(f"ERROR: frontend build dir not found: {FRONTEND_DIR}", file=sys.stderr)
        sys.exit(1)

    print("[UI-E2E] starting static frontend server on :3000")
    server = start_frontend_server()
    time.sleep(2)

    try:
        print("[UI-E2E] launching Edge")
        driver = new_driver()
        driver.get(BASE_URL)
        time.sleep(3)
        screenshot(driver, "01_landing")

        # Scenario 1: register + login
        print("[UI-E2E] scenario 1: register and login")
        username = f"uie2e_{int(time.time())}"
        password = "UiE2e2026!"
        # The selectors below are placeholders; adjust after inspecting the frontend.
        wait_for(driver, By.NAME, "username").send_keys(username)
        wait_for(driver, By.NAME, "password").send_keys(password)
        wait_for(driver, By.TAG_NAME, "button").click()
        time.sleep(3)
        screenshot(driver, "02_after_login")
        print(f"[UI-E2E] screenshot text: {capture_text(SCREENSHOT_DIR / '02_after_login.png')[:120]}")

        # Scenario 2: chat
        print("[UI-E2E] scenario 2: chat")
        wait_for(driver, By.TAG_NAME, "input").send_keys("一句话说明 GNN" + Keys.RETURN)
        time.sleep(6)
        screenshot(driver, "03_chat_reply")

        # Scenario 3 & 4: voice / image buttons (placeholder)
        print("[UI-E2E] scenario 3/4: voice and image buttons not implemented without stable selectors")

        # Scenario 5: logout
        print("[UI-E2E] scenario 5: logout")
        wait_for(driver, By.TAG_NAME, "button").click()  # placeholder
        time.sleep(2)
        screenshot(driver, "05_logout")

        driver.quit()
        print("[UI-E2E] completed (skeleton)")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()


if __name__ == "__main__":
    run_scenarios()
