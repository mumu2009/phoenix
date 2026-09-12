import importlib.util
import pathlib
import time
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tools" / "rdk_gateway_product_soak.py"
SPEC = importlib.util.spec_from_file_location("rdk_gateway_product_soak", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ClassifySoakContractTests(unittest.TestCase):
    def test_chat_timeout_is_not_down(self) -> None:
        kind = MODULE.classify(
            0,
            {"ok": False, "error": "The read operation timed out"},
            "/api/chat",
            300.1,
        )
        self.assertEqual(kind, "timeout")

    def test_chat_urlopen_timed_out_is_timeout(self) -> None:
        kind = MODULE.classify(
            0,
            {"ok": False, "error": "timed out"},
            "http://192.168.1.107:5081/api/chat",
            301.0,
        )
        self.assertEqual(kind, "timeout")

    def test_chat_slow_success_is_slow_not_down(self) -> None:
        kind = MODULE.classify(200, {"ok": True, "reply": "OK"}, "/api/chat", 156.0)
        self.assertEqual(kind, "slow")

    def test_chat_fast_success_is_ok(self) -> None:
        kind = MODULE.classify(200, {"ok": True, "reply": "OK"}, "/api/chat", 37.0)
        self.assertEqual(kind, "ok")

    def test_chat_busy_429(self) -> None:
        self.assertEqual(MODULE.classify(429, {"error": "chat-busy"}, "/api/chat"), "busy")
        self.assertEqual(MODULE.classify(200, {"error": "chat-busy"}, "/api/chat"), "busy")

    def test_connection_refused_is_down(self) -> None:
        kind = MODULE.classify(
            0,
            {"ok": False, "error": "urlopen error [WinError 10061] No connection could be made"},
            "/api/health",
            0.2,
        )
        self.assertEqual(kind, "down")

    def test_port_refused_chat_is_still_down(self) -> None:
        kind = MODULE.classify(
            0,
            {"ok": False, "error": "Connection refused"},
            "/api/chat",
            0.1,
        )
        self.assertEqual(kind, "down")

    def test_health_timeout_is_timeout_not_down(self) -> None:
        kind = MODULE.classify(
            0,
            {"ok": False, "error": "timed out"},
            "/api/health",
            15.0,
        )
        self.assertEqual(kind, "timeout")

    def test_last_ok_age_ignores_chat(self) -> None:
        stats = MODULE.Stats()
        stats.add("http://x/api/health", "ok", 0.1)
        first = stats.last_ok
        time.sleep(0.05)
        stats.add("/api/chat", "ok", 40.0)
        stats.add("/api/chat", "timeout", 300.0)
        self.assertEqual(stats.last_ok, first)
        stats.add("http://x/api/system/status", "ok", 0.2)
        self.assertGreater(stats.last_ok, first)

    def test_chat_timeout_seconds_at_least_300(self) -> None:
        self.assertGreaterEqual(MODULE.CHAT_TIMEOUT_SEC, 300.0)


if __name__ == "__main__":
    unittest.main()
