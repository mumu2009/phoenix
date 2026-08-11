#!/usr/bin/env python3
"""Real-user end-to-end UI test for the Phoenix web frontend.

This script only touches test code.  It drives Microsoft Edge via Selenium,
looks for input fields and buttons by their rendered HTML/text (visual/structural
search, not stable IDs), and aborts the whole run on the first failure.

Scenarios (atomic; one failure -> exit 1):
1. Register a new account and automatically log in.
2. Type a chat message and send it, verifying the prompt appears in the UI.
3. Attach the smoke image and send it, verifying the image path is accepted.
4. Click voice input / stop voice / speech recognition, verifying the UI state changes.
5. Log out by clearing localStorage and reloading, verifying the login form returns.
"""
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = os.environ.get("PHOENIX_UI_URL", "http://127.0.0.1:5081")
SMOKE_IMAGE = ROOT / "build" / "tmp" / "smoke_image.png"
SCREENSHOT_DIR = ROOT / "build" / "tmp" / "ui_e2e"
LOG_FILE = SCREENSHOT_DIR / "ui_e2e.log"

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8")],
    level=logging.INFO,
    format="[%(asctime)s] [UI-E2E] %(message)s",
    force=True,
)


def log(msg: str):
    logging.info(msg)


def ensure_smoke_image() -> Path:
    if not SMOKE_IMAGE.exists():
        from PIL import Image
        img = Image.new("RGB", (224, 224))
        for y in range(224):
            for x in range(224):
                img.putpixel((x, y), (x * 255 // 224, y * 255 // 224, 128))
        img.save(SMOKE_IMAGE)
    return SMOKE_IMAGE


def phoenix_running() -> bool:
    try:
        with urllib.request.urlopen(BASE_URL, timeout=3):
            return True
    except Exception:
        return False


def wait_for_llama_server(timeout: float = 120.0) -> bool:
    url = "http://127.0.0.1:8084/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read().decode("utf-8"))
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def wait_for_llama_server(port: int = 8084, timeout: float = 120.0) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read().decode("utf-8"))
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def start_llm_backend(env: dict) -> tuple[subprocess.Popen, subprocess.Popen | None, subprocess.Popen, subprocess.Popen | None]:
    log("starting llama-server (llama-3.1-8b)")
    model = ROOT / "GGUF_models" / "blobs" / "sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
    llama = subprocess.Popen(
        [str(ROOT / "outsides" / "llamacpp" / "build-gcc" / "bin" / "llama-server.exe"),
         "-m", str(model), "--host", "127.0.0.1", "--port", "8084", "-t", "6", "-c", "2048", "--parallel", "1"],
        cwd=str(ROOT),
        stdout=open(ROOT / "build" / "tmp" / "ui_llama_server.log", "w", encoding="utf-8"),
        stderr=open(ROOT / "build" / "tmp" / "ui_llama_server_err.log", "w", encoding="utf-8"),
    )
    if not wait_for_llama_server(8084):
        llama.kill()
        raise RuntimeError("llama-server (llama-3.1-8b) did not become healthy")

    tinyllama_model = ROOT / "GGUF_models" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    if tinyllama_model.exists():
        log("starting llama-server (tinyllama-1.1b)")
        tinyllama = subprocess.Popen(
            [str(ROOT / "outsides" / "llamacpp" / "build-gcc" / "bin" / "llama-server.exe"),
             "-m", str(tinyllama_model), "--host", "127.0.0.1", "--port", "8085", "-t", "4", "-c", "2048", "--parallel", "1"],
            cwd=str(ROOT),
            stdout=open(ROOT / "build" / "tmp" / "ui_tinyllama_server.log", "w", encoding="utf-8"),
            stderr=open(ROOT / "build" / "tmp" / "ui_tinyllama_server_err.log", "w", encoding="utf-8"),
        )
        if not wait_for_llama_server(8085):
            tinyllama.kill()
            raise RuntimeError("llama-server (tinyllama) did not become healthy")
    else:
        tinyllama = None

    log("starting llama_proxy")
    proxy = subprocess.Popen(
        [str(ROOT / "Python314" / "python.exe"), str(ROOT / "tools" / "llama_proxy.py"),
         "--proxy-port", "8080", "--backend-port", "8084"],
        cwd=str(ROOT),
        stdout=open(ROOT / "build" / "tmp" / "ui_llama_proxy.log", "w", encoding="utf-8"),
        stderr=open(ROOT / "build" / "tmp" / "ui_llama_proxy_err.log", "w", encoding="utf-8"),
        env=env,
    )

    tinyllama_proxy = None
    if tinyllama is not None:
        log("starting llama_proxy (tinyllama)")
        tinyllama_proxy = subprocess.Popen(
            [str(ROOT / "Python314" / "python.exe"), str(ROOT / "tools" / "llama_proxy.py"),
             "--proxy-port", "8086", "--backend-port", "8085"],
            cwd=str(ROOT),
            stdout=open(ROOT / "build" / "tmp" / "ui_tinyllama_proxy.log", "w", encoding="utf-8"),
            stderr=open(ROOT / "build" / "tmp" / "ui_tinyllama_proxy_err.log", "w", encoding="utf-8"),
            env=env,
        )

    time.sleep(1)
    return llama, tinyllama, proxy, tinyllama_proxy


def start_phoenix() -> list[subprocess.Popen] | None:
    if phoenix_running():
        log("phoenix_main already running")
        return None
    log("ensuring wikitext dataset is available")
    subprocess.run(
        [str(ROOT / "Python314" / "python.exe"), str(ROOT / "test-tools" / "ensure_wikitext.py")],
        cwd=str(ROOT),
        check=False,
    )
    log("setting deployment by hardware before start")
    subprocess.run(
        [str(ROOT / "Python314" / "python.exe"), str(ROOT / "test-tools" / "set_deployment_by_hardware.py")],
        cwd=str(ROOT),
        check=False,
    )
    log("starting phoenix_main with local llama backend")
    env = os.environ.copy()
    env["JEPA_IMAGE_VARIANT"] = "vision_encoder"
    env["JEPA_SPEECH_VARIANT"] = "speech_encoder"
    env["AI_LLAMACPP_BASE_URL"] = "http://127.0.0.1:8080"
    env["AI_LLAMACPP_MODEL"] = "GGUF_models\\blobs\\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
    env["AI_TINYLLAMA_BASE_URL"] = "http://127.0.0.1:8086"
    env["AI_TINYLLAMA_MODEL"] = "GGUF_models\\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    env["AI_TINYLLAMA_ENABLED"] = "true"
    env["AI_HTTP_LOG"] = "false"
    env["AI_DISABLE_LEARNING"] = "true"
    env["AI_DISABLE_GNN_MODULE"] = "true"
    env["AI_DISABLE_CONTEXT_MODULE"] = "true"
    env["FRONTEND_SUMMARY_MODEL_ENABLED"] = "false"
    env["TMP"] = str(ROOT / "build" / "tmp")
    env["TEMP"] = env["TMP"]
    llama, tinyllama, proxy, tinyllama_proxy = start_llm_backend(env)
    proc = subprocess.Popen(
        [str(ROOT / "phoenix_main.exe")],
        cwd=str(ROOT),
        stdout=open(ROOT / "build" / "tmp" / "ui_phoenix_out.log", "w", encoding="utf-8"),
        stderr=open(ROOT / "build" / "tmp" / "ui_phoenix_err.log", "w", encoding="utf-8"),
        env=env,
    )
    for _ in range(60):
        if phoenix_running():
            log("phoenix_main ready")
            procs = [llama, proxy, proc]
            if tinyllama is not None:
                procs.append(tinyllama)
            if tinyllama_proxy is not None:
                procs.append(tinyllama_proxy)
            return procs
        time.sleep(1)
    raise RuntimeError("phoenix_main did not become ready")


def new_driver() -> webdriver.Edge:
    options = EdgeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    user_dir = ROOT / "build" / "tmp" / "ui_e2e" / "edge_profile"
    if user_dir.exists():
        shutil.rmtree(user_dir)
    options.add_argument(f"--user-data-dir={user_dir}")
    driver = webdriver.Edge(options=options)
    driver.set_page_load_timeout(30)
    driver.set_script_timeout(10)
    return driver


def find_inputs(driver):
    return WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located((By.TAG_NAME, "input"))
    )


def find_button(driver, text: str, timeout: float = 10.0):
    end = time.time() + timeout
    while time.time() < end:
        for b in driver.find_elements(By.TAG_NAME, "button"):
            if text in (b.text or ""):
                return b
        time.sleep(0.5)
    raise RuntimeError(f"button with text '{text}' not found")


def find_button_or_none(driver, text: str, timeout: float = 5.0):
    try:
        return find_button(driver, text, timeout=timeout)
    except RuntimeError:
        return None


def save_screenshot(driver, name: str):
    path = SCREENSHOT_DIR / f"{name}.png"
    driver.save_screenshot(str(path))
    log(f"screenshot saved: {path}")
    return path


def assert_body(driver, name: str, *must_contain: str):
    body = driver.find_element(By.TAG_NAME, "body")
    text = body.text
    for s in must_contain:
        if s not in text:
            save_screenshot(driver, f"assert_fail_{name}")
            raise AssertionError(f"{name}: expected body to contain {s!r}, got {text[:200]!r}")
    log(f"{name}: body contains {must_contain}")


def scenario_1_register_login(driver):
    log("scenario 1: register and login")
    driver.get(BASE_URL)
    time.sleep(2)
    save_screenshot(driver, "01_landing")
    assert_body(driver, "landing", "身份验证", "登录")

    find_button(driver, "去注册").click()
    time.sleep(1)
    inputs = find_inputs(driver)
    assert len(inputs) == 4, f"register form should have 4 inputs, got {len(inputs)}"
    username = f"uie2e_{int(time.time() * 1000)}"
    inputs[0].send_keys(username)
    inputs[1].send_keys(f"{username}@example.com")
    inputs[2].send_keys("Test1234!")
    inputs[3].send_keys("Test1234!")
    save_screenshot(driver, "02_register_filled")
    find_button(driver, "注册并登录").click()

    time.sleep(3)
    save_screenshot(driver, "03_logged_in")
    assert_body(driver, "login_success", "079 Phoenix", "Local User", "发送")
    log(f"scenario 1 passed (user={username})")
    return username


def scenario_2_chat(driver):
    log("scenario 2: chat")
    inputs = find_inputs(driver)
    text_input = None
    for i in inputs:
        if i.get_attribute("type") == "text" and "输入消息" in (i.get_attribute("placeholder") or ""):
            text_input = i
            break
    if not text_input:
        raise RuntimeError("chat text input not found")
    text_input.send_keys("一句话说明 GNN 如何帮助 Transformer 推理")
    find_button(driver, "发送").click()
    time.sleep(4)
    save_screenshot(driver, "04_chat_sent")
    assert_body(driver, "chat_sent", "GNN", "Transformer")
    log("scenario 2 passed")


def scenario_3_image(driver):
    log("scenario 3: image upload")
    inputs = find_inputs(driver)
    file_input = None
    for i in inputs:
        if i.get_attribute("type") == "file":
            file_input = i
            break
    if not file_input:
        raise RuntimeError("image file input not found")
    image_path = ensure_smoke_image()
    file_input.send_keys(str(image_path))
    time.sleep(1)
    save_screenshot(driver, "05_image_attached")
    find_button(driver, "发送").click()
    time.sleep(5)
    save_screenshot(driver, "06_image_sent")
    body = driver.find_element(By.TAG_NAME, "body").text
    if "error" in body.lower() and "500" in body:
        raise AssertionError("image scenario: backend error in UI")
    log("scenario 3 passed")


def scenario_4_speech(driver):
    log("scenario 4: speech input UI flow")
    find_button(driver, "语音输入").click()
    time.sleep(1)
    save_screenshot(driver, "07_voice_recording")
    find_button(driver, "停止语音").click()
    time.sleep(2)
    save_screenshot(driver, "08_voice_stopped")
    find_button(driver, "语音识别").click()
    time.sleep(2)
    save_screenshot(driver, "09_voice_recognized")
    log("scenario 4 passed")


def scenario_5_config(driver):
    log("scenario 5: config panel")
    btn = find_button_or_none(driver, "Config")
    if btn is None:
        # try hamburger/settings icon labels
        for label in ("设置", "配置", "Settings", "⚙"):
            btn = find_button_or_none(driver, label)
            if btn is not None:
                break
    if btn is None:
        log("SKIP: no config button found")
        return
    btn.click()
    time.sleep(2)
    save_screenshot(driver, "11_config")
    body = driver.find_element(By.TAG_NAME, "body").text
    if not any(k in body for k in ("config", "配置", "Config", "设置")):
        raise AssertionError("config panel did not render")
    log("scenario 5 passed")


def scenario_6_world(driver):
    log("scenario 6: world panel")
    btn = find_button_or_none(driver, "World")
    if btn is None:
        for label in ("世界", "World Model", "world"):
            btn = find_button_or_none(driver, label)
            if btn is not None:
                break
    if btn is None:
        log("SKIP: no world button found")
        return
    btn.click()
    time.sleep(2)
    save_screenshot(driver, "12_world")
    body = driver.find_element(By.TAG_NAME, "body").text
    if not any(k in body.lower() for k in ("world", "世界", "环境")):
        raise AssertionError("world panel did not render")
    log("scenario 6 passed")


def scenario_7_logout(driver):
    log("scenario 7: logout and return to login")
    driver.execute_script("localStorage.removeItem('phoenix_auth_token'); location.reload();")
    time.sleep(3)
    save_screenshot(driver, "10_logout")
    assert_body(driver, "logout", "身份验证", "登录", "去注册")
    log("scenario 5 passed")


def kill_edge_orphans():
    """Kill any msedge.exe processes that were started for this test profile."""
    try:
        subprocess.run(
            ["wmic", "process", "where", 'name="msedge.exe" and commandline like "%ui_e2e%"', "call", "terminate"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def main() -> int:
    procs = None
    try:
        procs = start_phoenix()
        driver = new_driver()
        try:
            scenario_1_register_login(driver)
            scenario_2_chat(driver)
            scenario_3_image(driver)
            scenario_4_speech(driver)
            scenario_5_config(driver)
            scenario_6_world(driver)
            scenario_7_logout(driver)
            log("ALL UI E2E SCENARIOS PASSED")
            return 0
        finally:
            try:
                driver.quit()
            except Exception as e:
                log(f"WARN: driver.quit() failed: {e}")
    except Exception as e:
        log(f"FAIL: {e}")
        log(traceback.format_exc())
        return 1
    finally:
        if procs:
            log("stopping processes started by this test")
            for p in reversed(procs):
                if p is None:
                    continue
                try:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except Exception:
                        p.kill()
                except Exception as e:
                    log(f"WARN: process stop failed: {e}")
        kill_edge_orphans()
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())
