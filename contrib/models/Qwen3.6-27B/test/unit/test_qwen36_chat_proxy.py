# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[5]
_PROXY_PATH = (
    _REPO_ROOT / "contrib" / "models" / "Qwen3.6-27B" / "vllm" / "qwen36_chat_proxy.py"
)
_SPEC = importlib.util.spec_from_file_location("qwen36_chat_proxy_under_test", _PROXY_PATH)
_PROXY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROXY)


class TestQwen36ChatProxy(unittest.TestCase):
    def test_default_policy_disables_thinking(self):
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        enabled = _PROXY._apply_thinking_policy(payload, allow_thinking=True)

        self.assertFalse(enabled)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_default_thinking_enables_when_no_request_toggle(self):
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        enabled = _PROXY._apply_thinking_policy(
            payload,
            allow_thinking=True,
            default_thinking=True,
        )

        self.assertTrue(enabled)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})

    def test_default_thinking_allows_explicit_disable(self):
        payload = {"enable_thinking": False}

        enabled = _PROXY._apply_thinking_policy(
            payload,
            allow_thinking=True,
            default_thinking=True,
        )

        self.assertFalse(enabled)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("enable_thinking", payload)

    def test_force_disabled_policy_overrides_request(self):
        payload = {
            "enable_thinking": True,
            "chat_template_kwargs": {"enable_thinking": True, "foo": "bar"},
        }

        enabled = _PROXY._apply_thinking_policy(payload, allow_thinking=False)

        self.assertFalse(enabled)
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": False, "foo": "bar"},
        )
        self.assertNotIn("enable_thinking", payload)

    def test_allow_thinking_accepts_top_level_toggle(self):
        payload = {"enable_thinking": True, "chat_template_kwargs": {"foo": "bar"}}

        enabled = _PROXY._apply_thinking_policy(payload, allow_thinking=True)

        self.assertTrue(enabled)
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": True, "foo": "bar"},
        )
        self.assertNotIn("enable_thinking", payload)

    def test_allow_thinking_accepts_native_chat_template_kwargs(self):
        payload = {"chat_template_kwargs": {"enable_thinking": "true"}}

        enabled = _PROXY._apply_thinking_policy(payload, allow_thinking=True)

        self.assertTrue(enabled)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})

    def test_allow_thinking_accepts_reasoning_effort(self):
        payload = {"reasoning_effort": "low"}

        enabled = _PROXY._apply_thinking_policy(payload, allow_thinking=True)

        self.assertTrue(enabled)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("reasoning_effort", payload)

    def test_reasoning_effort_none_disables_thinking(self):
        payload = {"reasoning_effort": "none"}

        enabled = _PROXY._apply_thinking_policy(payload, allow_thinking=True)

        self.assertFalse(enabled)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning_effort", payload)

    def test_thinking_budget_tokens_can_toggle(self):
        payload = {"thinking": {"budget_tokens": 128}}

        enabled = _PROXY._apply_thinking_policy(payload, allow_thinking=True)

        self.assertTrue(enabled)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("thinking", payload)

    def test_system_and_developer_messages_are_hoisted(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "system", "content": "sys"},
            {"role": "developer", "content": [{"type": "text", "text": "dev"}]},
            {"role": "assistant", "content": "ok"},
        ]

        normalized = _PROXY._normalize_messages_for_qwen(messages)

        self.assertEqual(normalized[0], {"role": "system", "content": "sys\n\ndev"})
        self.assertEqual([message["role"] for message in normalized], ["system", "user", "assistant"])

    def test_chat_path_allows_trailing_slash_and_query(self):
        self.assertEqual(_PROXY._request_path("/v1/chat/completions"), "/v1/chat/completions")
        self.assertEqual(_PROXY._request_path("/v1/chat/completions/"), "/v1/chat/completions")
        self.assertEqual(
            _PROXY._request_path("/v1/chat/completions?api-version=1"),
            "/v1/chat/completions",
        )

    def test_streaming_thinking_start_is_injected_when_missing(self):
        event = (
            b'data: {"choices":[{"delta":{"content":"Here is the thought"}}]}\n\n'
        )

        patched, decided, changed = _PROXY._prepend_think_start_to_sse_event(event)

        self.assertTrue(decided)
        self.assertTrue(changed)
        self.assertIn(b'"content":"<think>\\nHere is the thought"', patched)

    def test_streaming_thinking_start_is_not_duplicated(self):
        event = b'data: {"choices":[{"delta":{"content":"\\n\\n<think>\\nThought"}}]}\n\n'

        patched, decided, changed = _PROXY._prepend_think_start_to_sse_event(event)

        self.assertTrue(decided)
        self.assertFalse(changed)
        self.assertEqual(patched, event)

    def test_streaming_usage_chunk_does_not_decide_thinking_start(self):
        event = b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'

        patched, decided, changed = _PROXY._prepend_think_start_to_sse_event(event)

        self.assertFalse(decided)
        self.assertFalse(changed)
        self.assertEqual(patched, event)


if __name__ == "__main__":
    unittest.main()
