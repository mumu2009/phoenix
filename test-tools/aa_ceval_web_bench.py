#!/usr/bin/env python3
"""C-Eval web UI benchmark for Phoenix + llama-3.1-8b.

Input flow goes through the web frontend; output is read from the rendered
page.  This tests the full Phoenix augmentation (GNN, context, etc.), not the
raw llama-server.
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from selenium.webdriver.common.action_chains import ActionChains

# Reuse the E2E launch/driver helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_e2e_test as ui

ROOT = Path(__file__).resolve().parent.parent
CEVAL_DIR = ROOT / "build" / "tmp" / "ceval" / "val"
RE_ANSWER = re.compile(r"(?<![A-Za-z0-9])[A-D](?![A-Za-z0-9])")


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except OSError:
        try:
            print(msg.encode("utf-8", errors="replace").decode("gbk", errors="replace"), flush=True)
        except OSError:
            pass
    ui.log(msg)


def load_ceval_questions(val_dir: Path, per_subject: int = 1):
    questions = []
    for csv_path in sorted(val_dir.glob("*_val.csv")):
        subject = csv_path.stem.replace("_val", "")
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        end = per_subject if per_subject > 0 else len(rows)
        for row in rows[:end]:
            questions.append(
                {
                    "subject": subject,
                    "id": row["id"],
                    "question": row["question"],
                    "A": row["A"],
                    "B": row["B"],
                    "C": row["C"],
                    "D": row["D"],
                    "answer": row["answer"].strip().upper(),
                }
            )
    return questions


def build_prompt(q: dict) -> str:
    return (
        f"问题：{q['question']}\n"
        f"A. {q['A']}\n"
        f"B. {q['B']}\n"
        f"C. {q['C']}\n"
        f"D. {q['D']}\n"
        "请只输出正确选项的一个字母（A/B/C/D），不要解释。"
    )


def find_chat_input(driver):
    return ui.WebDriverWait(driver, 20).until(
        ui.EC.presence_of_element_located((ui.By.CSS_SELECTOR, "input.composer-input"))
    )


def get_last_assistant_text(driver):
    msgs = driver.find_elements(ui.By.CSS_SELECTOR, ".msg.assistant")
    if not msgs:
        return ""
    try:
        return msgs[-1].find_element(ui.By.CSS_SELECTOR, ".msg-text").text
    except Exception:
        return ""


def send_chat(driver, prompt: str, timeout: float = 300.0) -> str:
    """Type a prompt into the web chat and return the assistant's reply text."""
    text_input = find_chat_input(driver)

    # Use the native value setter so React's controlled state actually updates.
    driver.execute_script(
        """
        const el = arguments[0];
        const value = arguments[1];
        const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
        valueSetter.call(el, '');
        valueSetter.call(el, value);
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        el.focus();
        """,
        text_input,
        prompt,
    )

    # Wait for the send button to become clickable, then click.
    send_btn = ui.WebDriverWait(driver, 20).until(
        ui.EC.element_to_be_clickable((ui.By.XPATH, "//button[contains(text(), '发送') and not(@disabled)]"))
    )
    send_btn.click()

    # Wait for a new assistant bubble to appear.
    before_count = len(driver.find_elements(ui.By.CSS_SELECTOR, ".msg.assistant"))
    try:
        ui.WebDriverWait(driver, timeout).until(
            lambda d: len(d.find_elements(ui.By.CSS_SELECTOR, ".msg.assistant")) > before_count
        )
    except Exception as e:
        ui.save_screenshot(driver, "timeout_no_reply")
        log(f"DEBUG body text: {driver.find_element(ui.By.TAG_NAME, 'body').text[:500]!r}")
        raise RuntimeError("assistant did not reply within timeout") from e

    # Wait for the bubble text to settle.
    for _ in range(5):
        txt1 = get_last_assistant_text(driver)
        time.sleep(0.5)
        txt2 = get_last_assistant_text(driver)
        if txt1 == txt2:
            break
    return get_last_assistant_text(driver)


def extract_choice(text: str) -> str | None:
    """Extract the first standalone A-D from the assistant's reply text."""
    text = text or ""
    m = RE_ANSWER.search(text)
    if not m:
        return None
    return m.group(0).upper()


def _reset_phoenix_storage():
    """Remove runtime DBs so each benchmark run starts with a clean world/context state."""
    for path in (
        ROOT / "runtime_store" / "ai_store.sqlite",
        ROOT / "runtime_store" / "frontend_world_model.sqlite",
        ROOT / "lmdb" / "frontend_world_model",
    ):
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()
        except Exception as e:
            log(f"WARN: could not reset {path}: {e}")


def run_benchmark(questions: list[dict], per_subject: int) -> dict:
    if not CEVAL_DIR.exists():
        raise FileNotFoundError(f"C-Eval val dir not found: {CEVAL_DIR}")

    _reset_phoenix_storage()
    procs = ui.start_phoenix()
    if procs is None:
        raise RuntimeError("phoenix_main is already running or failed to start")

    driver = ui.new_driver()
    correct = 0
    total = 0
    results = []

    try:
        ui.scenario_1_register_login(driver)
        for idx, q in enumerate(questions, 1):
            log(f"[{idx}/{len(questions)}] {q['subject']} id={q['id']}")
            if idx > 1:
                # Force a fresh browser session/chat context for every question.
                # This keeps per-turn world/context state from accumulating and
                # mirrors the clean state the API benchmark gets per request.
                driver.get(ui.BASE_URL)
            prompt = build_prompt(q)
            try:
                page_text = send_chat(driver, prompt)
            except Exception as e:
                log(f"  SEND CHAT FAILED: {e}")
                ui.save_screenshot(driver, f"q{idx:03d}_fail")
                raise
            choice = extract_choice(page_text)
            ok = choice == q["answer"]
            correct += int(ok)
            total += 1
            results.append(
                {
                    "subject": q["subject"],
                    "id": q["id"],
                    "expected": q["answer"],
                    "predicted": choice or "",
                    "correct": ok,
                }
            )
            if choice is None:
                log(f"  DEBUG assistant text: {page_text[:200]!r}")
                ui.save_screenshot(driver, f"q{idx:03d}_no_choice")
            log(f"  expected={q['answer']} predicted={choice or 'NONE'} correct={ok}")
    finally:
        try:
            driver.quit()
        except Exception as e:
            log(f"WARN: driver.quit() failed: {e}")
        ui.kill_edge_orphans()
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

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="C-Eval web benchmark for Phoenix")
    parser.add_argument("--gnn", choices=["on", "off"], default="on", help="GNN module state")
    parser.add_argument("--per-subject", type=int, default=1, help="questions per C-Eval subject (0 = all)")
    parser.add_argument("--max-questions", type=int, default=0, help="limit total questions (0 = all)")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "tmp" / "ceval_bench.json")
    args = parser.parse_args()

    if args.gnn == "off":
        os.environ["AI_DISABLE_GNN_MODULE"] = "1"
    elif args.gnn == "on":
        os.environ.pop("AI_DISABLE_GNN_MODULE", None)

    questions = load_ceval_questions(CEVAL_DIR, per_subject=args.per_subject)
    if args.max_questions > 0:
        questions = questions[:args.max_questions]
    log(f"GNN={args.gnn}, total questions={len(questions)}")

    stats = run_benchmark(questions, args.per_subject)

    args.output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"C-Eval accuracy (GNN {args.gnn}): {stats['correct']}/{stats['total']} = {stats['accuracy']:.4%}")
    log(f"results written to {args.output}")
    return 0 if stats["total"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
