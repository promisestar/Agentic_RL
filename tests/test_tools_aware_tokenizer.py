"""Tests for wrap_tokenizer_with_tools (方案一: dynamic subclass)."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

unsloth_stub = types.ModuleType("unsloth")
unsloth_stub.FastLanguageModel = MagicMock()
sys.modules["unsloth"] = unsloth_stub

trl_stub = types.ModuleType("trl")
trl_stub.GRPOTrainer = MagicMock()
trl_stub.GRPOConfig = MagicMock()
sys.modules["trl"] = trl_stub

datasets_stub = types.ModuleType("datasets")
datasets_stub.Dataset = MagicMock()
sys.modules["datasets"] = datasets_stub

import train_openenv as toe  # noqa: E402


class _FakeBaseTok:
    """Stand-in for PreTrainedTokenizerBase / TokenizersBackend."""

    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        tools = kwargs.get("tools") or []
        names = [t.get("function", {}).get("name", "") for t in tools]
        return f"<tools>{','.join(names)}</tools>"

    def encode(self, text):
        return [1, 2, 3]


class WrapTokenizerWithToolsTests(unittest.TestCase):
    def test_isinstance_preserved_via_subclass(self) -> None:
        tok = _FakeBaseTok()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "catalog.search",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        wrapped = toe.wrap_tokenizer_with_tools(tok, tools)
        # Same instance, new class that subclasses the original.
        self.assertIs(wrapped, tok)
        self.assertTrue(issubclass(type(wrapped), _FakeBaseTok))
        self.assertIsInstance(wrapped, _FakeBaseTok)
        self.assertTrue(getattr(wrapped, "_ecom_rlve_tools_wrapped", False))

    def test_apply_chat_template_injects_tools(self) -> None:
        tok = _FakeBaseTok()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "catalog.search",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        wrapped = toe.wrap_tokenizer_with_tools(tok, tools)
        out = wrapped.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertIn("catalog.search", out)

        # Explicit tools= from caller must not be overwritten.
        other = [{"type": "function", "function": {"name": "other"}}]
        out2 = wrapped.apply_chat_template([], tools=other)
        self.assertIn("other", out2)
        self.assertNotIn("catalog.search", out2)

    def test_getattr_and_encode_still_work(self) -> None:
        tok = _FakeBaseTok()
        wrapped = toe.wrap_tokenizer_with_tools(tok, [])
        self.assertEqual(wrapped.pad_token_id, 0)
        self.assertEqual(wrapped.encode("abc"), [1, 2, 3])

    def test_tools_aware_tokenizer_shim(self) -> None:
        tok = _FakeBaseTok()
        tools = [{"type": "function", "function": {"name": "cart.view"}}]
        wrapped = toe.ToolsAwareTokenizer(tok, tools)
        self.assertIs(wrapped, tok)
        self.assertIn("cart.view", wrapped.apply_chat_template([]))

    def test_double_wrap_refreshes_tools(self) -> None:
        tok = _FakeBaseTok()
        t1 = [{"type": "function", "function": {"name": "a"}}]
        t2 = [{"type": "function", "function": {"name": "b"}}]
        w1 = toe.wrap_tokenizer_with_tools(tok, t1)
        w2 = toe.wrap_tokenizer_with_tools(tok, t2)
        self.assertIs(w1, w2)
        self.assertIn("b", w2.apply_chat_template([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
