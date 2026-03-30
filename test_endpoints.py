import httpx, sys, json

BASE = "http://localhost:7860"
passed = 0
failed = 0


def test(name, method, path, body=None, expect_keys=None):
    global passed, failed
    try:
        if method == "GET":
            r = httpx.get(BASE + path, timeout=10)
        else:
            r = httpx.post(BASE + path, json=body or {}, timeout=10)
        assert r.status_code == 200, f"Status {r.status_code}"
        data = r.json()
        if expect_keys:
            for k in expect_keys:
                assert k in data, f"Missing key: {k}"
        print(f"  PASS: {name}")
        passed += 1
        return data
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        failed += 1
        return None


print("Testing TrafficSignalEnv endpoints...")
test("Health check", "GET", "/health", expect_keys=["status"])
test("Reset default", "POST", "/reset", {}, expect_keys=["intersections", "step_count"])
test("Reset task1", "POST", "/reset", {"task_id": "task1"}, expect_keys=["intersections"])
test("Reset task2", "POST", "/reset", {"task_id": "task2"})
test("Reset task3", "POST", "/reset", {"task_id": "task3"})
test(
    "Step switch_phase",
    "POST",
    "/step",
    {"intersection_id": 0, "command": "switch_phase"},
    expect_keys=["observation", "reward", "done", "info"],
)
test(
    "Step extend_green",
    "POST",
    "/step",
    {"intersection_id": 0, "command": "extend_green", "duration": 10},
)
test("State", "GET", "/state", expect_keys=["intersections"])
test("Tasks list", "GET", "/tasks", expect_keys=["tasks"])

print(f"Results: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
