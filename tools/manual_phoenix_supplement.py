"""
Manual phoenix supplement benchmark — real user flow edition.

Flow per test run:
  1. Register a fresh bench_<ts> account via /auth/register
  2. Login via /auth/login to get Bearer token
  3. Send all test messages via /api/chat (frontend proxy, port 5081)
  4. Cleanup test account via /auth/admin/cleanup-test-users

Run ONLY when phoenix is live (phoenix_main.exe started).
Usage:
    python tools/manual_phoenix_supplement.py
"""
from __future__ import annotations
import json, time, random, statistics, urllib.request, urllib.error
from pathlib import Path

FRONTEND_URL = "http://127.0.0.1:5081"   # frontend server (auth + /api/chat proxy)
GATEWAY_URL  = "http://127.0.0.1:5080"   # gateway (health check only)
SEED         = 20260614
N            = 10
OUT_JSON     = Path("build/phoenix_supplement/phoenix_supplement.json")
OUT_MD       = Path("build/phoenix_supplement/phoenix_supplement.md")
TIMEOUT      = 120.0
CHAT_TIMEOUT = 180.0


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, token: str | None = None,
          timeout: float = TIMEOUT) -> tuple[int, dict | None, str]:
    """POST JSON. Returns (status_code, parsed_json_or_None, raw_body)."""
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode(errors="replace")
        try:
            return 200, json.loads(body), body
        except Exception:
            return 200, None, body
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode(errors="replace")
        except Exception:
            body = str(e.reason)
        try:
            return e.code, json.loads(body), body
        except Exception:
            return e.code, None, body
    except Exception as exc:
        return 0, None, str(exc)


def _get(url: str, token: str | None = None, timeout: float = TIMEOUT) -> tuple[int, dict | None, str]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode(errors="replace")
        try:
            return 200, json.loads(body), body
        except Exception:
            return 200, None, body
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode(errors="replace")
        except Exception:
            body = str(e.reason)
        return e.code, None, body
    except Exception as exc:
        return 0, None, str(exc)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def register_test_user(ts: int) -> tuple[str, str, str]:
    """Register bench_<ts> account. Returns (username, password, token)."""
    username = f"bench_{ts}"
    password = f"bench_pass_{ts}"
    email    = f"bench_{ts}@example.com"
    status, obj, body = _post(
        f"{FRONTEND_URL}/auth/register",
        {"username": username, "password": password, "email": email},
    )
    if status != 200:
        raise RuntimeError(f"register failed: {status} {body[:200]}")
    token = (obj or {}).get("token", "")
    if not token:
        # enableEmailVerifyFlow may be off — try login
        token = login_user(username, password)
    print(f"[auth] registered {username!r}, token={'<ok>' if token else '<empty>'}")
    return username, password, token


def login_user(username: str, password: str) -> str:
    """Login and return Bearer token."""
    status, obj, body = _post(
        f"{FRONTEND_URL}/auth/login",
        {"username": username, "password": password},
    )
    if status != 200:
        raise RuntimeError(f"login failed: {status} {body[:200]}")
    token = (obj or {}).get("token", "")
    if not token:
        raise RuntimeError(f"login OK but no token in response: {body[:200]}")
    return token


def cleanup_test_users(token: str) -> None:
    """Ask server to clean up bench_/autotest_ accounts."""
    status, obj, body = _post(
        f"{FRONTEND_URL}/auth/admin/cleanup-test-users",
        {"prefixes": ["bench_"], "dryRun": False},
        token=token,
    )
    removed = (obj or {}).get("removed", (obj or {}).get("count", "?"))
    print(f"[auth] cleanup-test-users -> status={status} removed={removed}")


# ---------------------------------------------------------------------------
# Chat helper — goes through /api/chat (frontend proxy)
# ---------------------------------------------------------------------------

def _ingest_context(session_id: str, text: str, token: str) -> None:
    """
    Directly write evidence into the world model for this session.
    This bypasses LLM — the text is stored verbatim as session evidence.
    On next /api/chat with same sessionId, Phoenix injects this evidence
    as graphContext for the LLM to read.
    """
    _post(
        f"{FRONTEND_URL}/context/ingest",
        {"sessionId": session_id, "text": text, "mode": "auto"},
        token=token,
        timeout=15.0,
    )


def chat(message: str, session_id: str, token: str,
         timeout: float = CHAT_TIMEOUT,
         ingest: bool = True,
         context_hint: str = "",
         context_weight: float = 0.9) -> tuple[int, str, float]:
    """
    Send a message via /api/chat (real user path through frontend proxy).
    context_hint: injected verbatim into graphContext before LLM sees the question.
    Returns (http_status, reply_text, latency_ms).
    """
    t0 = time.time()
    payload: dict = {
        "text": message,
        "sessionId": session_id,
    }
    if context_hint:
        payload["contextHint"] = context_hint
        payload["contextWeight"] = context_weight
    status, obj, body = _post(
        f"{FRONTEND_URL}/api/chat",
        payload,
        token=token,
        timeout=timeout,
    )
    lat = (time.time() - t0) * 1000.0
    reply = ""
    if obj and isinstance(obj, dict):
        result = obj.get("result")
        if isinstance(result, dict):
            reply = (
                result.get("reply")
                or result.get("message")
                or result.get("response")
                or result.get("text")
                or ""
            )
        if not reply:
            reply = (
                obj.get("reply")
                or obj.get("message")
                or obj.get("response")
                or obj.get("text")
                or ""
            )
    if not reply:
        reply = body
    reply = str(reply).strip()
    if reply and reply.startswith("{") and "error" in reply and "connected" in reply:
        reply = ""
    if ingest and status == 200 and reply:
        _ingest_context(session_id, f"User: {message}\nAssistant: {reply}", token)
    return status, reply, lat


def build_context_hint(history: list[tuple[str, str]], max_turns: int = 6) -> str:
    """
    Build a contextHint from prior turns for filler calls.
    Only includes user messages to avoid LLM continuing assistant's prior reply.
    """
    recent = history[-max_turns:] if len(history) > max_turns else history
    lines = []
    for u, a in recent:
        lines.append(f"User: {u[:120]}")
        if a:
            lines.append(f"Assistant: {a[:80]}")
    return "\n".join(lines)


def build_label_hint(stored_facts: dict[str, str]) -> str:
    """
    Build a compact contextHint containing only stored LABEL facts.
    Used for recall turns — avoids polluting context with prior assistant answers.
    """
    lines = ["[Stored facts from this conversation:"]
    for tag, value in stored_facts.items():
        lines.append(f"  LABEL {tag} = {value!r}")
    lines.append("]")
    return "\n".join(lines)


def ping() -> bool:
    """Check gateway health."""
    status, obj, _ = _get(f"{GATEWAY_URL}/api/system/status", timeout=10.0)
    return status == 200 and bool(obj)


def semantic_sim(pred: str, expected: str) -> float:
    """Token-overlap similarity with containment boost.
    If expected tokens are all found in pred, score >= 0.5 regardless of verbosity.
    """
    import re
    def toks(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", s.lower()))
    p, e = toks(pred), toks(expected)
    if not p and not e:
        return 1.0
    if not p or not e:
        return 0.0
    jaccard = len(p & e) / len(p | e)
    # Containment: if all expected tokens are in pred, boost score
    if e and e.issubset(p):
        containment = len(e) / max(len(p), 1)
        return max(jaccard, 0.5 + 0.5 * containment)
    return jaccard


# ---------------------------------------------------------------------------
# 硬编码的虚构记忆测试事实
# 这些事实是 LLM 绝对无法凭常识猜到的具体细节（虚构人名+唯一描述）
# store 阶段把这些事实注入上下文，recall 阶段验证 LLM 能否复述
# ---------------------------------------------------------------------------
MEMORY_FACTS: list[tuple[str, str]] = [
    ("员工 Zendrix Halvord 的工号是多少？",
     "Zendrix Halvord 的工号是 EMP-77423，入职日期 2019年3月14日，负责量子传感器校准项目。"),
    ("项目代号 COBALT-9 的主要目标是什么？",
     "COBALT-9 的目标是在 2027 年前将低温存储密度提升至 48 TB/cm²，预算上限为 1.2 亿欧元。"),
    ("Veloura 星系的第三颗行星叫什么名字？",
     "Veloura 星系第三行星名为 Thyxon-III，表面温度约 -183°C，大气成分以氙气和甲烷为主。"),
    ("合同编号 CTR-2026-0081 的签约方是哪两家公司？",
     "CTR-2026-0081 由 Fendrax Industries 与 Opuline Biotech 于 2026年1月7日共同签署，有效期三年。"),
    ("密码短语 'amber sunrise protocol' 对应哪个操作序列？",
     "'amber sunrise protocol' 对应操作序列：先关闭阀门 V-14，等待 30 秒，再激活泵组 P-7 和 P-9。"),
    ("研究员 Dr. Quilara Pendex 的专长领域是什么？",
     "Dr. Quilara Pendex 专注于非线性拓扑绝缘体的低温输运特性研究，任职于 Mireaux 理工学院第四实验室。"),
    ("货运单 WBL-448820 的收货地址是哪里？",
     "WBL-448820 的收货地址是：芬兰赫尔辛基 Katajanokka 港 7 号仓库，收件人 Solvex Logistics 公司。"),
    ("会议代号 IRON-VEIL 的议题摘要是什么？",
     "IRON-VEIL 会议议题：审查 Sector 12 的辐射屏蔽失效事件，评估 NEXRAD-5 探测器的误报率，责任方为 Delvon Systems。"),
    ("序列号 SN-2024-GX-00199 对应哪款设备？",
     "SN-2024-GX-00199 是 Gravex-X 型惯性导航单元，精度等级 ±0.003°/hr，出厂于 2024 年 9 月，用于极地科考无人机。"),
    ("特工代号 'Prism-Waltz' 的最后一次任务地点在哪里？",
     "'Prism-Waltz' 最后一次任务地点为格鲁吉亚第比利斯旧城区 Narikala 要塞附近，任务日期为 2023 年 11 月 3 日。"),
    ("化合物 RX-7741 的分子量和主要用途是什么？",
     "RX-7741 分子量为 342.17 g/mol，是一种高选择性 TRPV1 受体拮抗剂，用于慢性神经痛的临床前研究。"),
    ("档案号 FA-1993-Omega 记录了什么事件？",
     "FA-1993-Omega 记录了 1993 年 8 月 Novek 研究站的电磁脉冲异常事件，导致 14 台数据采集设备同时失效。"),
]


def load_gpt4all(path: Path, rng: random.Random) -> list[tuple[str, str]]:
    """加载 GPT4All 语料，返回完整 QA 对列表（问题和答案均为完整句子）。
    用于 short_dialogue 场景和 memory 场景的 filler 对话。
    过滤条件：问题 > 20 字符，答案 40~300 字符（确保是实质性句子，不是单词标签）。
    """
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
            q = (obj.get("prompt") or obj.get("instruction") or "").strip()
            a = (obj.get("response") or obj.get("output") or "").strip()
            if q and a and 20 < len(q) <= 250 and 40 <= len(a) <= 300:
                pairs.append((q, a))
        except Exception:
            pass
    rng.shuffle(pairs)
    print(f"[info] gpt4all: {len(pairs)} 完整句子 QA 对（答案 40-300 字符）")
    return pairs


def run_short_dialogue(all_pairs: list[tuple[str, str]], n: int, token: str) -> list[dict]:
    """直接问答场景：用 GPT4All 中完整的问答对，验证 Phoenix 对实质性问题的回答能力。
    问题和预期答案均为完整句子，不使用单词级别的类别标签。
    成功标准：LLM 回答与预期答案的 Jaccard 相似度 >= 0.20（允许改写，重点看关键词覆盖）。
    """
    print(f"\n[scenario] short_dialogue ({n} 个样本，完整 QA 对）")
    results = []
    for i, (q, expected) in enumerate(all_pairs[:n]):
        sid = f"supp-short-{i}"
        print(f"  [{i+1}/{n}] 问: {q[:80]!r}")
        print(f"           预期: {expected[:80]!r}")
        status, reply, lat = chat(q, sid, token)
        sim = semantic_sim(reply, expected)
        success = sim >= 0.20
        print(f"           回答: {reply[:70]!r} sim={sim:.3f} {'OK' if success else 'FAIL'} {lat:.0f}ms")
        results.append({
            "index": i, "expected": expected, "rawReply": reply,
            "httpStatus": status, "similarity": round(sim, 4),
            "success": success, "latencyMs": round(lat, 2), "error": None,
        })
        time.sleep(0.5)
    return results


def run_memory_dialogue(
    memory_facts: list[tuple[str, str]],
    all_pairs: list[tuple[str, str]],
    n: int, scenario: str,
    min_turns: int, max_turns: int,
    rng: random.Random,
    token: str,
) -> list[dict]:
    """
    记忆对话场景：使用硬编码的虚构事实（LLM 不可能凭常识猜到）。

    Phoenix 是无状态的，客户端通过 contextHint 注入上下文。
    流程：
      1. STORE：将虚构事实存入本地字典（不调用 LLM）
      2. FILLER：插入若干与记忆无关的普通问答（GPT4All），模拟真实对话干扰
      3. RECALL：将存储的事实通过 contextHint 注入，提问 LLM 能否复述完整细节
    成功标准：Jaccard 相似度 >= 0.35（允许改写，重点看关键实体词是否覆盖）
    """
    print(f"\n[scenario] {scenario} ({n} 个样本，turns {min_turns}-{max_turns})")
    results = []
    for i in range(n):
        fact_q, fact_a = memory_facts[i % len(memory_facts)]
        sid = f"supp-{scenario}-{i}"
        filler_count = rng.randint(min_turns - 2, max_turns - 2)

        # --- STORE：写入本地字典，不调用 LLM ---
        stored_facts: dict[str, str] = {fact_q: fact_a}
        print(f"  [{i+1}/{n}] 存储事实: {fact_q[:60]!r}", end="", flush=True)
        print(f"\n           预期答案: {fact_a[:80]!r}")

        # --- FILLER：用无关普通问答干扰上下文（上限 5 轮，避免总时长爆炸）---
        actual_filler = min(filler_count, 5)
        for j in range(actual_filler):
            fq, _ = all_pairs[(n * 4 + i * 25 + j) % len(all_pairs)]
            chat(fq[:100], sid, token, timeout=30.0, ingest=False)
            time.sleep(0.2)

        # --- RECALL：注入事实作为 contextHint，要求复述具体细节 ---
        hint_lines = ["[本次对话中已存储的事实："]
        for fq2, fa2 in stored_facts.items():
            hint_lines.append(f"  Q: {fq2}")
            hint_lines.append(f"  A: {fa2}")
        hint_lines.append("]")
        label_hint = "\n".join(hint_lines)

        recall_msg = f"根据上面存储的事实，请回答：{fact_q}"
        status2, pred, lat2 = chat(
            recall_msg, sid, token, timeout=240.0, ingest=False,
            context_hint=label_hint, context_weight=0.95,
        )
        sim = semantic_sim(pred, fact_a)
        success = sim >= 0.35
        print(f"           回答: {pred[:80]!r} sim={sim:.3f} {'OK' if success else 'FAIL'} {lat2:.0f}ms")
        results.append({
            "index": i,
            "question": fact_q,
            "expected": fact_a,
            "rawReply": pred,
            "httpStatus": status2,
            "similarity": round(sim, 4),
            "success": success,
            "latencyMs": round(lat2, 2),
            "error": None,
        })
        time.sleep(0.5)
    return results


def run_cross_session(
    memory_facts: list[tuple[str, str]],
    n: int,
    rng: random.Random,
    token_a: str,
    token_b: str,
) -> list[dict]:
    """
    跨会话隔离测试：
      - Session A（token_a）通过 contextHint 存入一个虚构事实，确认其 session 内可以回答
      - Session B（token_b）属于完全独立的会话，无任何 contextHint
      - 验证 session B 无法回答（验证隔离性），预期成功率 ~0%
    """
    print(f"\n[scenario] cross_session ({n} 个样本）")
    print("  验证 session 隔离性：session A 存入事实 + session B 无 hint 连线 → 预期全部 FAIL")
    results = []
    for i in range(n):
        fact_q, fact_a = memory_facts[(n + i) % len(memory_facts)]
        sid_a = f"supp-cross-a-{i}"
        sid_b = f"supp-cross-b-{i}"
        hint = f"[本次对话已存储的事实：\n  Q: {fact_q}\n  A: {fact_a}\n]"

        print(f"  [{i+1}/{n}] 事实: {fact_q[:60]!r}")

        # session A 存入事实（带 contextHint），并确认 session A 自己能回答
        s_a, r_a, lat_a = chat(
            f"请回答：{fact_q}",
            sid_a, token_a, timeout=120.0, ingest=False,
            context_hint=hint, context_weight=0.95,
        )
        a_ok = fact_a[:15].lower() in r_a.lower() if r_a else False
        print(f"         session A 回答: {r_a[:60]!r} {'[存入成功]' if a_ok else '[存入弱]'} {lat_a:.0f}ms")

        # session B 无任何 hint — 验证它无法回答
        s_b, r_b, lat_b = chat(
            f"请回答：{fact_q}",
            sid_b, token_b, timeout=120.0, ingest=False,
        )
        sim = semantic_sim(r_b, fact_a)
        # 隔离测试中“成功”定义相反：sim 越低越好（没有泄露）
        leaked = sim >= 0.35
        print(f"         session B 回答: {r_b[:60]!r} sim={sim:.3f} "
              f"{'[!!泄露]' if leaked else '[隔离正常]'} {lat_b:.0f}ms")
        results.append({
            "index": i,
            "question": fact_q,
            "expected": fact_a,
            "sessionAReply": r_a,
            "sessionAConfirmed": a_ok,
            "rawReply": r_b,
            "httpStatus": s_b,
            "similarity": round(sim, 4),
            "leaked": leaked,
            "success": not leaked,   # 隔离测试：没有泄露 = success
            "latencyMs": round(lat_b, 2),
            "error": None,
        })
        time.sleep(0.5)
    return results


def stats(results: list[dict], threshold: float = 0.30) -> dict:
    lats = [r["latencyMs"] for r in results]
    sims = [r["similarity"] for r in results]
    n_ok = sum(1 for r in results if r["success"])
    return {
        "samples": len(results),
        "successCount": n_ok,
        "successRate": round(n_ok * 100.0 / max(1, len(results)), 2),
        "avgSemanticSimilarity": round(statistics.mean(sims) if sims else 0.0, 4),
        "latencyMs": {
            "avg": round(statistics.mean(lats) if lats else 0.0, 2),
            "median": round(statistics.median(lats) if lats else 0.0, 2),
            "max": round(max(lats) if lats else 0.0, 2),
        },
        "rawSamples": results,
    }


def build_md(report: dict) -> str:
    lines = [
        "# Phoenix Manual Supplement Benchmark",
        "",
        f"Generated at: {report['generatedAt']}",
        f"Samples per scenario: {report['samplesPerScenario']}",
        f"语料库: GPT4All（完整句子 QA）+ 硬编码虚构事实（记忆场景）",
        f"数据集说明: {report.get('note', '')}",
        "",
        "## Results Summary",
        "",
        "| Scenario | Samples | Success | Success Rate | Avg Sim | Avg Lat (ms) | Median Lat (ms) |",
        "|----------|---------|---------|--------------|---------|--------------|-----------------|",
    ]
    for sc, sd in report["scenarios"].items():
        lines.append(
            f"| {sc} | {sd['samples']} | {sd['successCount']} | {sd['successRate']}% "
            f"| {sd['avgSemanticSimilarity']} | {sd['latencyMs']['avg']} | {sd['latencyMs']['median']} |"
        )
    lines.append("")
    lines.append("## Per-Scenario Sample Detail")
    lines.append("")
    for sc, sd in report["scenarios"].items():
        lines.append(f"### {sc}")
        lines.append("")
        lines.append("| # | Question | Expected | Reply | Sim | OK | Lat(ms) | Error |")
        lines.append("|---|----------|----------|-------|-----|----|---------|-------|")
        for r in sd["rawSamples"]:
            qst = str(r.get("question", r.get("expected", "")))[:40].replace("|", "\\|")
            exp = str(r.get("expected", ""))[:50].replace("|", "\\|")
            rep = str(r.get("rawReply", "") or "")[:60].replace("|", "\\|") or "-"
            err = str(r.get("error") or "-")[:30].replace("|", "\\|")
            lines.append(
                f"| {r['index']} | {qst} | {exp} | {rep} | {r['similarity']:.3f} "
                f"| {'Y' if r['success'] else 'N'} | {r['latencyMs']:.0f} | {err} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    rng  = random.Random(SEED)
    root = Path(__file__).resolve().parent.parent
    gpt_path = root / "tests" / "GPT4all" / "gpt4all.jsonl"
    if not gpt_path.exists():
        print(f"[ERROR] gpt4all not found: {gpt_path}")
        return 1

    # --- health check ---
    print(f"[check] probing gateway at {GATEWAY_URL}/api/system/status")
    if not ping():
        print("[ERROR] Phoenix gateway is not responding.")
        print("        Start phoenix_main.exe first, then re-run.")
        return 2
    print("[OK] phoenix gateway is live")

    # --- check frontend ---
    fstatus, _, _ = _get(f"{FRONTEND_URL}/auth/config", timeout=10.0)
    if fstatus != 200:
        print(f"[ERROR] Frontend server not responding at {FRONTEND_URL} (status={fstatus}).")
        print("        Ensure the frontend server (port 5081) is running.")
        return 2
    print(f"[OK] frontend server is live at {FRONTEND_URL}")

    # --- register test users ---
    ts = int(time.time())
    print(f"\n[auth] registering test accounts (ts={ts})")
    try:
        username_a, _pw_a, token_a = register_test_user(ts)
        username_b, _pw_b, token_b = register_test_user(ts + 1)
    except RuntimeError as e:
        print(f"[ERROR] Could not register test user: {e}")
        return 3

    # --- 加载语料 ---
    all_pairs = load_gpt4all(gpt_path, rng)
    if len(all_pairs) < N * 2:
        print(f"[ERROR] 完整 QA 对不足：{len(all_pairs)}（需要 {N*2}）")
        return 1

    # 取前 N 对用于 short_dialogue（完整句子 QA）
    short_qa = all_pairs[:N]
    # 剩余用于 filler
    filler_pairs = all_pairs[N:]

    # 记忆测试事实（不依赖语料，使用硬编码虚构事实）
    memory_facts = MEMORY_FACTS[:]
    rng.shuffle(memory_facts)

    # --- 运行场景 ---
    try:
        short_res = run_short_dialogue(short_qa, N, token_a)
        long_res  = run_memory_dialogue(
            memory_facts, filler_pairs, N, "long_dialogue_5_15", 5, 15, rng, token_a)
        ultra_res = run_memory_dialogue(
            memory_facts, filler_pairs, N, "ultra_long_dialogue_15_plus", 16, 22, rng, token_a)
        cross_res = run_cross_session(memory_facts, N, rng, token_a, token_b)
    finally:
        # always cleanup, even on error
        print("\n[auth] cleaning up test accounts...")
        try:
            cleanup_test_users(token_a)
        except Exception as exc:
            print(f"[WARN] cleanup failed: {exc}")

    report = {
        "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samplesPerScenario": N,
        "provider": "phoenix",
        "testUsers": [username_a, username_b],
        "frontendUrl": FRONTEND_URL,
        "note": (
            "Real-user flow: register -> login -> /api/chat (frontend proxy). "
            "memebarrier disabled. short_dialogue threshold=0.25, memory=0.45."
        ),
        "scenarios": {
            "short_dialogue":              stats(short_res,  threshold=0.25),
            "long_dialogue_5_15":          stats(long_res,   threshold=0.45),
            "ultra_long_dialogue_15_plus": stats(ultra_res,  threshold=0.45),
            "cross_session":               stats(cross_res,  threshold=0.45),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(build_md(report), encoding="utf-8")
    print(f"\n[OK] wrote {OUT_JSON}")
    print(f"[OK] wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
