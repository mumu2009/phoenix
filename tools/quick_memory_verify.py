"""
快速记忆验证脚本 — 在运行完整 benchmark 前先手动验证。

测试内容：
  1. Session 内记忆：session1 先被告知 abc=xxx，然后问 abc 是什么
  2. 跨 session 隔离：session2 完全没有 session1 的上下文，问同一个问题

使用方法：
    python tools/quick_memory_verify.py

需要 Phoenix 正在运行（gateway:5080 frontend:5081）。
"""
from __future__ import annotations
import json, time, sys, urllib.request, urllib.error, logging, os

# 配置日志直接写文件，避免 stdout 重定向问题
os.makedirs("build", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("build/quick_memory_verify_latest.log", mode="w", encoding="utf-8", errors="replace")
    ]
)
log = logging

FRONTEND_URL = "http://127.0.0.1:5081"
GATEWAY_URL  = "http://127.0.0.1:5080"


def _post(url, payload, token=None, timeout=120.0):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode(errors="replace")
        try:
            return r.status, json.loads(body), body
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


def _get(url, token=None, timeout=10.0):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode(errors="replace")
        try:
            return r.status, json.loads(body), body
        except Exception:
            return 200, None, body
    except urllib.error.HTTPError as e:
        return e.code, None, str(e.reason)
    except Exception as exc:
        return 0, None, str(exc)


def context_status(session_id: str, token: str = ""):
    """查询 /context/status，返回 (mode, messageCount, shortWindowItems)。"""
    import urllib.parse
    url = f"{FRONTEND_URL}/context/status?sessionId={urllib.parse.quote(session_id)}"
    status, obj, _ = _get(url, token=token, timeout=10.0)
    if status == 200 and isinstance(obj, dict) and obj.get("exists"):
        return obj.get("lastMode", "?"), obj.get("messageCount", 0), obj.get("shortWindowItems", 0)
    return "(none)", 0, 0


def knowledge_status(token: str = ""):
    """查询 /context/knowledge，返回跨 session 学习全局状态 dict。"""
    url = f"{FRONTEND_URL}/context/knowledge"
    status, obj, _ = _get(url, token=token, timeout=10.0)
    if status == 200 and isinstance(obj, dict):
        return obj
    return {}


def reset_session(session_id: str, token: str = ""):
    """触发 /context/reset（前端直连处理器，body 传 sessionId），归档并启动跨 session 学习。"""
    url = f"{FRONTEND_URL}/context/reset"
    return _post(url, {"sessionId": session_id}, token=token, timeout=30.0)


def wait_for_learning(prev_learned: int, timeout_s: float = 180.0, token: str = ""):
    """轮询 /context/knowledge，等待后台探测完成（learnInFlight 归零）且 learnedFactCount 增长。"""
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        st = knowledge_status(token)
        last = st
        in_flight = st.get("learnInFlight", 0)
        learned = st.get("learnedFactCount", 0)
        if in_flight == 0 and learned > prev_learned:
            return st
        time.sleep(2.0)
    return last


def register(ts_suffix: int):
    username = f"bench_{ts_suffix}"
    status, obj, body = _post(f"{FRONTEND_URL}/auth/register", {
        "username": username,
        "password": f"pass_{ts_suffix}",
        "email": f"{username}@example.com",
    })
    if status != 200:
        raise RuntimeError(f"注册失败 {status}: {body[:200]}")
    token = (obj or {}).get("token", "")
    if not token:
        # 尝试 login
        status2, obj2, body2 = _post(f"{FRONTEND_URL}/auth/login", {
            "username": username, "password": f"pass_{ts_suffix}"
        })
        token = (obj2 or {}).get("token", "")
    if not token:
        raise RuntimeError("注册/登录均未返回 token")
    log.info(f"  [auth] 注册 {username!r} 成功，token 已获取")
    return username, token


def chat(message: str, session_id: str, token: str, context_hint: str = "", timeout=300.0):
    payload = {"text": message, "sessionId": session_id, "maxTokens": 500, "temperature": 0.7}
    if context_hint:
        payload["contextHint"] = context_hint
        payload["contextWeight"] = 0.95
    t0 = time.time()
    status, obj, body = _post(f"{FRONTEND_URL}/api/chat", payload, token=token, timeout=timeout)
    lat = (time.time() - t0) * 1000
    reply = ""
    if obj and isinstance(obj, dict):
        result = obj.get("result", {})
        if isinstance(result, dict):
            reply = (result.get("reply") or result.get("message") or
                     result.get("response") or result.get("text") or "")
        if not reply:
            reply = (obj.get("reply") or obj.get("message") or obj.get("text") or "")
    if not reply:
        reply = body
    reply = str(reply).strip()
    # 过滤错误 JSON 响应
    if reply.startswith("{") and ("error" in reply):
        reply = f"[ERR:{status}] {reply[:80]}"
    return status, reply, lat


def sep(title=""):
    log.info("\n" + "="*60)
    if title:
        log.info(f"  {title}")
        log.info("="*60)


def main():
    # ── 健康检查 ──────────────────────────────────────────────
    sep("健康检查")
    gs, _, _ = _get(f"{GATEWAY_URL}/api/system/status")
    fs, _, _ = _get(f"{FRONTEND_URL}/auth/config")
    log.info(f"  Gateway  {GATEWAY_URL} → {gs}")
    log.info(f"  Frontend {FRONTEND_URL} → {fs}")
    if gs != 200 or fs != 200:
        log.info("\n[STOP] Phoenix 未运行，请先启动 phoenix_main.exe")
        return

    ts = int(time.time())
    user1, tok1 = register(ts)
    user2, tok2 = register(ts + 1)

    # ── 测试 1：Session 内记忆（同一 session，contextHint 注入）─────
    sep("测试 1：Session 内记忆（contextHint 注入）")
    sid1 = f"mem-test-{ts}-A"

    # 先告知事实（通过 contextHint）
    fact    = "Zendrix Halvord 的工号是 EMP-77423，负责量子传感器校准项目。"
    store_q = "请记住以下信息：" + fact
    hint    = f"[本次对话已记录的事实：\n  {fact}\n]"

    log.info(f"\n  [1a] 发送存储请求（内嵌 contextHint）...")
    s, r, lat = chat(store_q, sid1, tok1, context_hint=hint)
    log.info(f"       状态={s}  延迟={lat:.0f}ms")
    log.info(f"       回复: {r[:120]!r}")

    log.info(f"\n  [1b] 在同一 session 内提问（携带同样的 contextHint）...")
    q_recall = "Zendrix Halvord 的工号是多少？"
    s2, r2, lat2 = chat(q_recall, sid1, tok1, context_hint=hint)
    log.info(f"       状态={s2}  延迟={lat2:.0f}ms")
    log.info(f"       回复: {r2[:200]!r}")
    ok1 = "EMP-77423" in r2 or "emp-77423" in r2.lower()
    log.info(f"       ✅ 包含正确工号 EMP-77423" if ok1 else f"       ❌ 未包含正确工号（实际：{r2[:80]!r}）")

    # ── 测试 2：Session 内记忆（无 contextHint，仅凭 LLM 上下文窗口）
    sep("测试 2：Session 内记忆（无 contextHint，仅靠 LLM 对话历史）")
    sid2 = f"mem-test-{ts}-B"

    log.info(f"\n  [2a] 发送存储请求（无 contextHint）...")
    s3, r3, lat3 = chat(f"请记住：工号 EMP-99001 属于员工 Mirova Qeltran。", sid2, tok1)
    log.info(f"       状态={s3}  延迟={lat3:.0f}ms  回复: {r3[:80]!r}")

    log.info(f"\n  [2b] 同一 session 内提问（无 contextHint）...")
    s4, r4, lat4 = chat("Mirova Qeltran 的工号是多少？", sid2, tok1)
    log.info(f"       状态={s4}  延迟={lat4:.0f}ms")
    log.info(f"       回复: {r4[:200]!r}")
    ok2 = "EMP-99001" in r4 or "99001" in r4
    log.info(f"       ✅ 包含正确工号 EMP-99001" if ok2 else f"       ❌ 未包含（实际：{r4[:80]!r}）")
    m2, mc2, sw2 = context_status(sid2)
    log.info(f"       → 路由状态: mode={m2} messageCount={mc2} windowItems={sw2}")
    log.info(f"       → 预期 PASS：前端代理已自动从 ContextService 注入会话历史（无需手动 contextHint）")

    # ── 测试 3：跨 Session 隔离（session B 无任何上下文）───────────
    sep("测试 3：跨 Session 隔离（不同用户/session，无 contextHint）")
    log.info(f"\n  session A ({user1}) 存储事实后...")
    log.info(f"  session B ({user2}) 独立询问同一问题（无 contextHint）")

    sid_b = f"mem-test-{ts}-CROSS-B"
    s5, r5, lat5 = chat("Zendrix Halvord 的工号是多少？", sid_b, tok2)
    log.info(f"       状态={s5}  延迟={lat5:.0f}ms")
    log.info(f"       回复: {r5[:200]!r}")
    ok3 = "EMP-77423" in r5
    log.info(f"       ⚠️  session B 意外知道了工号（可能模型训练数据含此内容）" if ok3
          else f"       ✅ session B 不知道工号（符合预期：跨 session 无记忆泄露）")

    # ── 测试 4：RNN 档多轮多事实记忆（验证分层路由 + 长距离会话记忆）──
    sep("测试 4：RNN 档多轮多事实记忆（concat→rnn 分层路由，无 contextHint）")
    sid4 = f"mem-test-{ts}-RNN"
    facts = [
        "请记住：工号 EMP-10001 属于员工 Alpha。",
        "请记住：工号 EMP-10002 属于员工 Beta。",
        "请记住：工号 EMP-10003 属于员工 Gamma。",
    ]
    tier_ok = True
    for i, fact in enumerate(facts):
        s, r, lat = chat(fact, sid4, tok1, timeout=600.0)
        log.info(f"  [4-{i+1}] 存储事实 {i+1}: 状态={s} 延迟={lat:.0f}ms")
        if s != 200:
            tier_ok = False
    # 依次查询每个工号
    for i, name in enumerate(["Alpha", "Beta", "Gamma"]):
        q = f"{name} 的工号是多少？"
        s, r, lat = chat(q, sid4, tok1, timeout=600.0)
        log.info(f"  [4-{i+4}] 查询 {name}: 状态={s} 延迟={lat:.0f}ms 回复: {r[:100]!r}")
        if s != 200 or f"EMP-1000{i+1}" not in r:
            tier_ok = False
    ok4 = tier_ok
    log.info(f"  {'✅ PASS - RNN 档多轮多事实记忆正常' if ok4 else '❌ FAIL - RNN 档测试失败'}")

    # ── 测试 5：跨 Session 学习（学习未知 / 隔离已知）──────────────
    sep("测试 5：跨 Session 学习（探测基座 LLM：未知→学习持久化，已知→隔离）")
    learn0 = knowledge_status(tok1)
    learned_before = learn0.get("learnedFactCount", 0)
    log.info(f"  学习启用={learn0.get('crossSessionLearnEnabled')} 阈值={learn0.get('knownSimThreshold')} 初始已学习事实={learned_before}")

    # [5a] Session A 存储一个基座模型几乎不可能知道的虚构事实（≥2 条消息以触发归档）
    #      使用非敏感虚构知识（虚构星球属性），避免触发模型对 PII/工号的安全拒答。
    sidA = f"mem-test-{ts}-LEARN-A"
    unknown_fact = "请记住：虚构星球 Zorvex-9 的表面平均温度是零下188摄氏度，大气主要由氙晶气体组成。"
    s, r, _ = chat(unknown_fact, sidA, tok1, timeout=600.0)
    log.info(f"  [5a] Session A 存储未知事实: 状态={s} 回复: {r[:60]!r}")
    s, r, _ = chat("好的，谢谢你记住了。", sidA, tok1, timeout=600.0)
    log.info(f"       Session A 追加一句以满足归档条件(messageCount>=2)")

    # [5b] reset 触发后台跨 session 学习探测
    st, obj, _ = reset_session(sidA, tok1)
    log.info(f"  [5b] reset Session A → 状态={st} removed={(obj or {}).get('removed')}，等待后台探测...")
    learn1 = wait_for_learning(learned_before, timeout_s=240.0, token=tok1)
    learned_after = learn1.get("learnedFactCount", learned_before)
    log.info(f"       探测完成: learnInFlight={learn1.get('learnInFlight')} learnedFactCount={learned_after} episodicEntries={learn1.get('episodicEntries')}")
    learned_ok = learned_after > learned_before
    log.info(f"       {'✅ 已将未知事实学习并持久化' if learned_ok else '⚠️  未学习到新事实（可能基座模型自信回答或探测失败）'}")

    # [5c] 新 session B 询问该未知事实，应能从跨 session 记忆检索到
    sidB = f"mem-test-{ts}-LEARN-B"
    s5b, r5b, lat5b = chat("虚构星球 Zorvex-9 的表面平均温度是多少？", sidB, tok1, timeout=600.0)
    log.info(f"  [5c] Session B 查询: 状态={s5b} 延迟={lat5b:.0f}ms 回复: {r5b[:200]!r}")
    ok5 = ("188" in r5b)
    log.info(f"       {'✅ PASS - 新 session 成功检索到已学习的跨 session 未知事实' if ok5 else '❌ FAIL - 新 session 未能检索到学习内容'}")

    # [5d] 软验证“隔离已知”：存储一条常识事实，reset 后不应被持久化
    sidC = f"mem-test-{ts}-KNOWN-C"
    learned_pre_known = knowledge_status(tok1).get("learnedFactCount", 0)
    chat("请记住：中华人民共和国的首都是北京。", sidC, tok1, timeout=600.0)
    chat("好的。", sidC, tok1, timeout=600.0)
    reset_session(sidC, tok1)
    # 等待后台探测（可能不增长）
    time.sleep(2.0)
    kstat = knowledge_status(tok1)
    # 轮询直到 inFlight 归零
    _dl = time.time() + 180.0
    while kstat.get("learnInFlight", 0) != 0 and time.time() < _dl:
        time.sleep(2.0)
        kstat = knowledge_status(tok1)
    learned_post_known = kstat.get("learnedFactCount", learned_pre_known)
    isolate_ok = learned_post_known == learned_pre_known
    log.info(f"  [5d] 隔离已知: 学习计数 {learned_pre_known}→{learned_post_known} "
             f"{'✅ 已知常识被隔离（未持久化）' if isolate_ok else '🟡 已知常识也被学习（探测判定为未知）'}")

    # ── 汇总 ─────────────────────────────────────────────────
    sep("汇总")
    log.info(f"  测试1 session 内记忆（手动 contextHint）:      {'✅ PASS' if ok1 else '❌ FAIL'}")
    log.info(f"  测试2 session 内记忆（无hint，自动注入）:       {'✅ PASS' if ok2 else '❌ FAIL'}")
    log.info(f"  测试3 跨 session 隔离（无 reset，不泄露）:       {'✅ PASS（隔离正常）' if not ok3 else '⚠️  session 隔离失效'}")
    log.info(f"  测试4 RNN档多轮多事实:                         {'✅ PASS' if ok4 else '❌ FAIL'}")
    log.info(f"  测试5 跨 session 学习未知事实:                  {'✅ PASS' if ok5 else '❌ FAIL'}")
    log.info(f"  测试5d 隔离已知常识:                            {'✅ PASS' if isolate_ok else '🟡 部分'}")
    log.info("")
    log.info("  结论：")
    if ok2:
        log.info("  → ✅ 前端代理已接入 ContextService：无需手动 contextHint 即可在 session 内记忆。")
    else:
        log.info("  → ❌ 自动注入未生效，检查 frontend_server.cpp proxyApiCall 的 prepareChatContext 集成。")
    if ok5:
        log.info("  → ✅ 跨 session 学习生效：探测基座 LLM 判定为未知的事实被学习并可在新 session 检索。")
    else:
        log.info("  → ❌ 跨 session 学习未生效，检查 probeBaseModelKnows / learnUnknownFacts / retrieveEpisodicMemory。")
    if isolate_ok:
        log.info("  → ✅ 已知常识被正确隔离，未污染跨 session 知识库。")
    if ok3:
        log.info("  → ⚠️ 无 reset 时跨 session 出现记忆泄露，需检查会话隔离。")


if __name__ == "__main__":
    main()
