#!/usr/bin/env python3
"""
最小化记忆验证脚本 - 仅验证 llamacpp 后端可用性和基本记忆功能
"""
import json
import logging
import time
import requests
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

FRONTEND_URL = "http://127.0.0.1:5081"
GATEWAY_URL = "http://127.0.0.1:5080"
LLAMACPP_URL = "http://127.0.0.1:8082"

def sep(title: str):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)

def _post(url: str, data: dict, token: str, timeout: float = 30.0):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(url, json=data, headers=headers, timeout=timeout)
    return resp

def chat(msg: str, session_id: str, token: str, timeout: float = 120.0):
    data = {
        "message": msg,
        "sessionId": session_id,
        "maxTokens": 500,
        "temperature": 0.7
    }
    start = time.time()
    resp = _post(f"{FRONTEND_URL}/api/chat", data, token, timeout=timeout)
    lat = (time.time() - start) * 1000
    if resp.status_code == 200:
        result = resp.json()
        reply = result.get("result", {}).get("reply", "")
        return resp.status_code, reply, lat
    else:
        return resp.status_code, f"[ERR:{resp.status_code}] {resp.text[:200]}", lat

def context_status(session_id: str):
    resp = requests.get(f"{FRONTEND_URL}/api/context/status?sessionId={session_id}", timeout=10.0)
    if resp.status_code == 200:
        data = resp.json()
        return data.get("mode", ""), data.get("messageCount", 0), data.get("windowItems", 0)
    return "", 0, 0

def main():
    ts = int(time.time())
    sep("最小化记忆验证 - llamacpp 后端")

    # 健康检查
    log.info("  健康检查")
    try:
        resp = requests.get(f"{FRONTEND_URL}/", timeout=5.0)
        log.info(f"  Frontend {FRONTEND_URL} → {resp.status_code}")
    except Exception as e:
        log.error(f"  Frontend 健康检查失败: {e}")
        return 1

    try:
        resp = requests.get(f"{LLAMACPP_URL}/health", timeout=5.0)
        log.info(f"  llamacpp {LLAMACPP_URL} → {resp.status_code}")
    except Exception as e:
        log.error(f"  llamacpp 健康检查失败: {e}")
        return 1

    # 注册用户
    sep("用户注册")
    user1 = f"bench_{ts}"
    reg_data = {"username": user1, "password": "test123", "email": f"{user1}@test.local"}
    resp = requests.post(f"{FRONTEND_URL}/auth/register", json=reg_data, timeout=30.0)
    if resp.status_code == 200:
        token1 = resp.json().get("token", "")
        log.info(f"  [auth] 注册 '{user1}' 成功，token 已获取")
    else:
        log.error(f"  [auth] 注册失败: {resp.status_code} {resp.text[:200]}")
        return 1

    # 测试1：基本聊天功能
    sep("测试 1：基本聊天功能（验证 llamacpp 后端可用）")
    sid1 = f"mem-test-{ts}-BASIC"
    s1, r1, lat1 = chat("你好，请简短回复。", sid1, token1, timeout=120.0)
    log.info(f"  状态={s1} 延迟={lat1:.0f}ms")
    log.info(f"  回复: {r1[:200]!r}")
    ok1 = (s1 == 200 and len(r1) > 0)
    log.info(f"  {'✅ PASS' if ok1 else '❌ FAIL'} - llamacpp 后端可用")

    # 测试2：Session 内记忆
    sep("测试 2：Session 内记忆（无 contextHint）")
    sid2 = f"mem-test-{ts}-MEMORY"
    fact = "请记住：苹果是红色的。"
    s2a, r2a, lat2a = chat(fact, sid2, token1, timeout=120.0)
    log.info(f"  [2a] 存储事实: 状态={s2a} 延迟={lat2a:.0f}ms")
    s2b, r2b, lat2b = chat("苹果是什么颜色的？", sid2, token1, timeout=120.0)
    log.info(f"  [2b] 回忆事实: 状态={s2b} 延迟={lat2b:.0f}ms")
    log.info(f"  回复: {r2b[:200]!r}")
    ok2 = (s2b == 200 and "红色" in r2b)
    log.info(f"  {'✅ PASS' if ok2 else '❌ FAIL'} - Session 内记忆工作正常")

    # 测试3：跨 Session 隔离
    sep("测试 3：跨 Session 隔离")
    user2 = f"bench_{ts+1}"
    reg_data2 = {"username": user2, "password": "test123", "email": f"{user2}@test.local"}
    resp2 = requests.post(f"{FRONTEND_URL}/auth/register", json=reg_data2, timeout=30.0)
    if resp2.status_code == 200:
        token2 = resp2.json().get("token", "")
    else:
        log.error(f"  [auth] 注册用户2失败")
        return 1

    sid3 = f"mem-test-{ts}-CROSS"
    s3, r3, lat3 = chat("苹果是什么颜色的？", sid3, token2, timeout=120.0)
    log.info(f"  状态={s3} 延迟={lat3:.0f}ms")
    log.info(f"  回复: {r3[:200]!r}")
    ok3 = (s3 == 200 and "红色" not in r3)
    log.info(f"  {'✅ PASS' if ok3 else '⚠️ 可能泄露'} - 跨 Session 隔离正常")

    # 汇总
    sep("汇总")
    log.info(f"  测试1 基本聊天功能:         {'✅ PASS' if ok1 else '❌ FAIL'}")
    log.info(f"  测试2 Session 内记忆:      {'✅ PASS' if ok2 else '❌ FAIL'}")
    log.info(f"  测试3 跨 Session 隔离:      {'✅ PASS' if ok3 else '⚠️ 可能泄露'}")
    log.info("")
    log.info("  结论：")
    if ok1:
        log.info("  → ✅ llamacpp 后端可用，基本聊天功能正常")
    else:
        log.info("  → ❌ llamacpp 后端不可用，检查 llama-server 启动状态")
    if ok2:
        log.info("  → ✅ Session 内记忆工作正常")
    else:
        log.info("  → ❌ Session 内记忆未生效，检查 ContextService 集成")
    if ok3:
        log.info("  → ✅ 跨 Session 隔离正常")
    else:
        log.info("  → ⚠️ 跨 Session 可能出现记忆泄露")

    return 0 if (ok1 and ok2) else 1

if __name__ == "__main__":
    sys.exit(main())
