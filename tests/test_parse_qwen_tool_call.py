"""Tests for Qwen XML tool_call parsing and dual-format parse_action."""

from __future__ import annotations

import json
import unittest

from ecom_rlve.server.state import parse_action, parse_qwen_tool_calls


class ParseQwenToolCallTests(unittest.TestCase):
    def test_parse_qwen_tool_calls_with_json_filters(self) -> None:
        text = """I'll search the catalog for you.

<tool_call>
<function=catalog.search>
<parameter=query>
electronics mobile tablets
</parameter>
<parameter=filters>
{"price_max": 35.14, "cat": "electronics/mobile/tablets"}
</parameter>
<parameter=top_k>
20
</parameter>
</function>
</tool_call>
"""
        calls = parse_qwen_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "catalog.search")
        self.assertEqual(calls[0].args["query"], "electronics mobile tablets")
        self.assertEqual(calls[0].args["filters"]["price_max"], 35.14)
        self.assertEqual(
            calls[0].args["filters"]["cat"], "electronics/mobile/tablets"
        )
        self.assertEqual(calls[0].args["top_k"], 20)

    def test_parse_action_xml_path(self) -> None:
        text = (
            "Searching now.\n"
            "<tool_call>\n"
            "<function=catalog.search>\n"
            "<parameter=query>\ntablets\n</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
        )
        action, valid = parse_action(text)
        self.assertTrue(valid)
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.assistant_message, "Searching now.")
        self.assertEqual(len(action.tool_calls), 1)
        self.assertEqual(action.tool_calls[0].name, "catalog.search")
        self.assertIsNone(action.answer)

    def test_parse_action_xml_empty_preamble(self) -> None:
        text = (
            "<tool_call>\n"
            "<function=cart.view>\n"
            "</function>\n"
            "</tool_call>\n"
        )
        action, valid = parse_action(text)
        self.assertTrue(valid)
        assert action is not None
        self.assertEqual(action.assistant_message, "Using tools.")
        self.assertEqual(action.tool_calls[0].name, "cart.view")

    def test_parse_action_terminal_json_still_works(self) -> None:
        payload = {
            "assistant_message": "Here are my recommendations.",
            "tool_calls": [],
            "answer": {
                "env": "PD",
                "recommended_product_ids": ["syn_000032"],
                "done": True,
            },
        }
        action, valid = parse_action(json.dumps(payload))
        self.assertTrue(valid)
        assert action is not None
        self.assertTrue(action.answer is not None and action.answer.done)
        self.assertEqual(action.answer.recommended_product_ids, ["syn_000032"])

    def test_parse_action_invalid(self) -> None:
        action, valid = parse_action("not a tool call and not json")
        self.assertFalse(valid)
        self.assertIsNone(action)


if __name__ == "__main__":
    unittest.main(verbosity=2)
