"""Tests for ToolRegistry.to_openai_tools()."""

from __future__ import annotations

import unittest

from ecom_rlve.tools.cart import register_cart_tools
from ecom_rlve.tools.catalog import CatalogSearchArgs, register_catalog_tools
from ecom_rlve.tools.orders import register_order_tools
from ecom_rlve.tools.policy import register_policy_tools
from ecom_rlve.tools.registry import ToolRegistry
from ecom_rlve.tools.returns import register_return_tools


class OpenAIToolsExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        register_catalog_tools(self.registry)
        register_cart_tools(self.registry)
        register_order_tools(self.registry)
        register_return_tools(self.registry)
        register_policy_tools(self.registry)

    def test_to_openai_tools_shape(self) -> None:
        tools = self.registry.to_openai_tools()
        self.assertGreater(len(tools), 0)
        for t in tools:
            self.assertEqual(t["type"], "function")
            self.assertIn("function", t)
            fn = t["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            self.assertIn("parameters", fn)
            self.assertIsInstance(fn["parameters"], dict)

    def test_catalog_search_schema_includes_filters_constraints(self) -> None:
        tools = self.registry.to_openai_tools()
        by_name = {t["function"]["name"]: t for t in tools}
        self.assertIn("catalog.search", by_name)
        params = by_name["catalog.search"]["function"]["parameters"]
        props = params.get("properties") or {}
        self.assertIn("query", props)
        self.assertIn("filters", props)
        self.assertIn("top_k", props)
        filters_desc = props["filters"].get("description", "")
        self.assertIn("price_max", filters_desc)
        self.assertIn("cat", filters_desc)
        self.assertIn("electronics/mobile/tablets", filters_desc)
        self.assertIn("max_price", filters_desc)  # warned against in description
        field = CatalogSearchArgs.model_fields["filters"]
        self.assertIn("price_max", field.description or "")
        self.assertIn("Do NOT invent keys", field.description or "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
