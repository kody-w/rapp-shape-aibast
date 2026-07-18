"""Behavioral pins for the self-heal core and the wave-2 runtime hardening.

Review found call_copilot's resilience paths (401 invalidate-retry, the
400/429/5xx model-fallback loop, multi-choice merge) had ZERO coverage — they
run exactly when Copilot misbehaves in production, and any regression shipped
green. Also pins: auth-chain precedence, the SSE heartbeat helper, the pip
typosquat gate, and the agent class cache's fresh-instance guarantee.
"""
import json
import os
import shutil
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

import requests

import brainstem as bs


class FakeResp:
    def __init__(self, status_code, body=None, text=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text if text is not None else json.dumps(self._body)
        self.encoding = None

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code), response=self)


def _ok_body(content="pong"):
    return {"choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}]}


class _CallCopilotBase(unittest.TestCase):
    """Patch the network + token layer around call_copilot."""

    def _run(self, post_responses, model="claude-x", models=None, tokens=None):
        posts = []
        responses = list(post_responses)

        def fake_post(url, headers=None, json=None, timeout=None, **kw):
            posts.append({"url": url, "headers": dict(headers or {}), "body": dict(json or {})})
            return responses.pop(0)

        token_seq = list(tokens or [("tok1", "https://ep/v1")])

        def fake_token():
            return token_seq.pop(0) if len(token_seq) > 1 else token_seq[0]

        models = models if models is not None else [
            {"id": model, "available": True},
            {"id": bs._SAFETY_NET_MODEL, "available": True},
        ]
        with mock.patch.object(bs, "MODEL", model), \
             mock.patch.object(bs, "AVAILABLE_MODELS", models), \
             mock.patch.object(bs, "get_copilot_token", side_effect=fake_token) as _, \
             mock.patch.object(bs, "_invalidate_copilot_token") as invalidate, \
             mock.patch.object(bs.requests, "post", side_effect=fake_post):
            result = bs.call_copilot([{"role": "user", "content": "hi"}])
        return result, posts, invalidate


class TestCallCopilot401SelfHeal(_CallCopilotBase):
    def test_401_invalidates_reexchanges_and_retries_once(self):
        (result, responded_model), posts, invalidate = self._run(
            [FakeResp(401, text="unauthorized"), FakeResp(200, _ok_body())],
            tokens=[("tok1", "https://ep/v1"), ("tok2", "https://ep/v1")],
        )
        self.assertEqual(len(posts), 2)
        invalidate.assert_called_once()
        self.assertEqual(posts[0]["headers"]["Authorization"], "Bearer tok1")
        self.assertEqual(posts[1]["headers"]["Authorization"], "Bearer tok2")
        self.assertEqual(result["choices"][0]["message"]["content"], "pong")
        self.assertEqual(responded_model, "claude-x")


class TestCallCopilotModelFallback(_CallCopilotBase):
    def test_400_falls_back_to_safety_net_and_reports_substitution(self):
        (result, responded_model), posts, _ = self._run(
            [FakeResp(400, text="bad model"), FakeResp(200, _ok_body())])
        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0]["body"]["model"], "claude-x")
        self.assertEqual(posts[1]["body"]["model"], bs._SAFETY_NET_MODEL)
        # The silent substitution must be surfaced to the caller.
        self.assertEqual(responded_model, bs._SAFETY_NET_MODEL)

    def test_fallback_sweep_is_capped(self):
        models = [{"id": "claude-x", "available": True}] + [
            {"id": f"m{i}", "available": True} for i in range(8)]
        failures = [FakeResp(500, text="down")] * (1 + bs._FALLBACK_ATTEMPT_CAP + 5)
        with self.assertRaises(requests.exceptions.HTTPError):
            self._run(failures, models=models)
        # Re-run capturing posts (assertRaises swallowed the return path).
        posts_seen = []
        responses = [FakeResp(500, text="down")] * (1 + bs._FALLBACK_ATTEMPT_CAP + 5)

        def fake_post(url, headers=None, json=None, timeout=None, **kw):
            posts_seen.append(dict(json or {}))
            return responses.pop(0)

        with mock.patch.object(bs, "MODEL", "claude-x"), \
             mock.patch.object(bs, "AVAILABLE_MODELS", models), \
             mock.patch.object(bs, "get_copilot_token", return_value=("t", "https://ep/v1")), \
             mock.patch.object(bs.requests, "post", side_effect=fake_post):
            with self.assertRaises(requests.exceptions.HTTPError):
                bs.call_copilot([{"role": "user", "content": "hi"}])
        self.assertEqual(len(posts_seen), 1 + bs._FALLBACK_ATTEMPT_CAP)


class TestCallCopilotResponseNormalization(_CallCopilotBase):
    def test_multi_choice_split_text_and_tools_are_merged(self):
        body = {"choices": [
            {"message": {"role": "assistant", "content": "thinking..."},
             "finish_reason": "stop"},
            {"message": {"role": "assistant",
                         "tool_calls": [{"id": "c1", "function": {"name": "F", "arguments": "{}"}}]},
             "finish_reason": "tool_calls"},
        ]}
        (result, _), posts, _ = self._run([FakeResp(200, body)])
        self.assertEqual(len(result["choices"]), 1)
        merged = result["choices"][0]["message"]
        self.assertEqual(merged["content"], "thinking...")
        self.assertEqual(len(merged["tool_calls"]), 1)
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")

    def test_empty_choices_raises_descriptive_error(self):
        with self.assertRaisesRegex(RuntimeError, "no choices"):
            self._run([FakeResp(200, {"choices": []})])


class TestGithubTokenPrecedence(unittest.TestCase):
    def test_env_var_wins(self):
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "env_tok"}), \
             mock.patch.object(bs, "_read_token_file",
                               return_value={"access_token": "file_tok"}):
            self.assertEqual(bs.get_github_token(), "env_tok")

    def test_token_file_beats_gh_cli(self):
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": ""}), \
             mock.patch.object(bs, "_read_token_file",
                               return_value={"access_token": "file_tok"}), \
             mock.patch.object(bs.subprocess, "run",
                               side_effect=AssertionError("gh must not be consulted")):
            self.assertEqual(bs.get_github_token(), "file_tok")

    def test_gh_cli_gho_token_is_rejected(self):
        gh = mock.Mock()
        gh.stdout = "gho_not_copilot_compatible\n"
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": ""}), \
             mock.patch.object(bs, "_read_token_file", return_value=None), \
             mock.patch.object(bs.subprocess, "run", return_value=gh):
            self.assertIsNone(bs.get_github_token())

    def test_gh_cli_compatible_token_is_used_last(self):
        gh = mock.Mock()
        gh.stdout = "ghu_from_cli\n"
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": ""}), \
             mock.patch.object(bs, "_read_token_file", return_value=None), \
             mock.patch.object(bs.subprocess, "run", return_value=gh):
            self.assertEqual(bs.get_github_token(), "ghu_from_cli")


class TestStreamHeartbeat(unittest.TestCase):
    @staticmethod
    def _drive(gen):
        yielded = []
        while True:
            try:
                yielded.append(next(gen))
            except StopIteration as stop:
                return yielded, stop.value

    def test_pings_flow_while_call_blocks_and_result_is_returned(self):
        def slow():
            time.sleep(0.08)
            return ("result", "model-x")

        with mock.patch.object(bs, "_STREAM_HEARTBEAT_SECS", 0.01):
            pings, value = self._drive(bs._blocking_call_with_heartbeat(slow))
        self.assertGreaterEqual(len(pings), 1)
        self.assertTrue(all(p == ": ping\n\n" for p in pings))
        self.assertEqual(value, ("result", "model-x"))

    def test_exceptions_propagate(self):
        def broken():
            raise RuntimeError("upstream fell over")

        with mock.patch.object(bs, "_STREAM_HEARTBEAT_SECS", 0.01):
            with self.assertRaisesRegex(RuntimeError, "upstream fell over"):
                self._drive(bs._blocking_call_with_heartbeat(broken))

    def test_stream_fallback_emits_heartbeat_comments(self):
        """End-to-end: /chat/stream in non-streaming fallback keeps bytes moving."""
        def fake_stream(*a, **k):
            raise bs.StreamingUnsupported(400, "no stream", "claude-x")
            yield  # pragma: no cover

        def slow_call(messages, tools=None):
            time.sleep(0.05)
            return ({"choices": [{"message": {"role": "assistant", "content": "hi"},
                                  "finish_reason": "stop"}]}, "claude-x")

        bs.app.testing = True
        client = bs.app.test_client()
        with mock.patch.object(bs, "_STREAM_HEARTBEAT_SECS", 0.01), \
             mock.patch.object(bs, "load_soul", return_value="SOUL"), \
             mock.patch.object(bs, "load_agents", return_value={}), \
             mock.patch.object(bs, "call_copilot_stream", side_effect=fake_stream), \
             mock.patch.object(bs, "call_copilot", side_effect=slow_call):
            resp = client.post("/chat/stream", json={"user_input": "hi"})
            raw = resp.get_data(as_text=True)
        self.assertIn(": ping\n\n", raw)
        self.assertIn('"type": "done"', raw.replace('", "', '", "'))


class TestPipTyposquatGate(unittest.TestCase):
    def setUp(self):
        self._failed = set(bs._failed_installs)
        self._refused = set(bs._refused_installs)
        bs._failed_installs.clear()
        bs._refused_installs.clear()

    def tearDown(self):
        bs._failed_installs.clear()
        bs._failed_installs.update(self._failed)
        bs._refused_installs.clear()
        bs._refused_installs.update(self._refused)

    def test_undeclared_package_is_refused_without_running_pip(self):
        with mock.patch.object(bs.subprocess, "run",
                               side_effect=AssertionError("pip must not run")):
            self.assertFalse(bs._auto_install("requets"))

    def test_curated_package_is_allowed(self):
        ok = mock.Mock()
        ok.returncode = 0
        with mock.patch.object(bs.subprocess, "run", return_value=ok) as run:
            self.assertTrue(bs._auto_install("beautifulsoup4"))
        run.assert_called_once()

    def test_declared_package_is_allowed(self):
        ok = mock.Mock()
        ok.returncode = 0
        with mock.patch.object(bs.subprocess, "run", return_value=ok) as run:
            self.assertTrue(bs._auto_install("somenichepkg",
                                             declared=frozenset({"somenichepkg"})))
        run.assert_called_once()

    def test_requires_declaration_parsing(self):
        tmp = tempfile.mkdtemp(prefix="reqdecl-")
        try:
            path = os.path.join(tmp, "x_agent.py")
            with open(path, "w") as f:
                f.write("# requires: feedparser, Beautifulsoup4\n"
                        "#requires: extra-pkg\n"
                        "import feedparser\n")
            declared = bs._declared_requirements(path)
            self.assertEqual(declared,
                             frozenset({"feedparser", "beautifulsoup4", "extra-pkg"}))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestAgentClassCache(unittest.TestCase):
    AGENT_TEMPLATE = textwrap.dedent('''
        from agents.basic_agent import BasicAgent
        with open({counter!r}, "a") as f:
            f.write("x")

        class {cls}(BasicAgent):
            def __init__(self):
                self.name = {name!r}
                self.metadata = {{
                    "name": self.name,
                    "description": "cache test agent",
                    "parameters": {{"type": "object", "properties": {{}}, "required": []}},
                }}
                super().__init__(name=self.name, metadata=self.metadata)

            def perform(self, **kwargs):
                return "ok"
    ''')

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="agentcache-")
        self.counter = os.path.join(self.tmp, "exec_count")
        with bs._agent_class_cache_lock:
            bs._agent_class_cache.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        with bs._agent_class_cache_lock:
            bs._agent_class_cache.clear()

    def _execs(self):
        try:
            with open(self.counter) as f:
                return len(f.read())
        except OSError:
            return 0

    def _write(self, cls="CacheAgent", name="CacheAgent"):
        path = os.path.join(self.tmp, "cache_test_agent.py")
        with open(path, "w") as f:
            f.write(self.AGENT_TEMPLATE.format(counter=self.counter, cls=cls, name=name))
        return path

    def test_unchanged_file_execs_once_but_instances_stay_fresh(self):
        path = self._write()
        first = bs._load_agent_from_file(path)
        second = bs._load_agent_from_file(path)
        self.assertEqual(self._execs(), 1)  # module exec'd once, not per request
        self.assertIn("CacheAgent", first)
        self.assertIn("CacheAgent", second)
        # Fresh instance per load — no state can leak between requests.
        self.assertIsNot(first["CacheAgent"], second["CacheAgent"])
        self.assertIs(type(first["CacheAgent"]), type(second["CacheAgent"]))

    def test_edited_file_reloads(self):
        path = self._write()
        bs._load_agent_from_file(path)
        # Rewrite with a different agent; ensure the mtime tick moves.
        time.sleep(0.01)
        with open(path, "w") as f:
            f.write(self.AGENT_TEMPLATE.format(
                counter=self.counter, cls="EditedAgent", name="EditedAgent"))
        os.utime(path)
        agents = bs._load_agent_from_file(path)
        self.assertIn("EditedAgent", agents)
        self.assertNotIn("CacheAgent", agents)
        self.assertEqual(self._execs(), 2)

    def test_one_failing_constructor_does_not_drop_siblings(self):
        path = os.path.join(self.tmp, "mixed_agent.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent('''
                from agents.basic_agent import BasicAgent

                class BrokenAgent(BasicAgent):
                    def __init__(self):
                        raise RuntimeError("boom")

                    def perform(self, **kwargs):
                        return "never"

                class HealthyAgent(BasicAgent):
                    def __init__(self):
                        self.name = "Healthy"
                        self.metadata = {
                            "name": self.name,
                            "description": "survives sibling failure",
                            "parameters": {"type": "object", "properties": {}, "required": []},
                        }
                        super().__init__(name=self.name, metadata=self.metadata)

                    def perform(self, **kwargs):
                        return "ok"
            '''))
        agents = bs._load_agent_from_file(path)
        self.assertIn("Healthy", agents)
        with bs._quarantine_lock:
            entry = bs._quarantined_agents.get(path)
        self.assertIsNotNone(entry)


if __name__ == "__main__":
    unittest.main()
