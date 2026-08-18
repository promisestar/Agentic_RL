"""Tests that ProductCard exposes ``cat`` for post-search verification."""

from __future__ import annotations

import unittest

from ecom_rlve.data.schema import Product, ProductCard, product_to_card


class ProductCardCatTests(unittest.TestCase):
    def test_product_to_card_includes_cat(self) -> None:
        product = Product(
            id="syn_000032",
            title="BrightPath E-Reader",
            desc="An e-reader.",
            cat="electronics/mobile/tablets",
            brand="BrightPath",
            attrs={"color": "orange"},
            price=29.35,
            rating=4.5,
            ship_days=1,
            stock_qty=19,
        )
        card = product_to_card(product)
        self.assertIsInstance(card, ProductCard)
        self.assertEqual(card.cat, "electronics/mobile/tablets")
        dumped = card.model_dump()
        self.assertEqual(dumped["cat"], "electronics/mobile/tablets")
        self.assertEqual(dumped["product_id"], "syn_000032")
        self.assertEqual(dumped["price"], 29.35)

    def test_product_card_requires_cat(self) -> None:
        with self.assertRaises(Exception):
            ProductCard(
                product_id="p1",
                title="t",
                price=1.0,
                rating=1.0,
                ship_days=1,
                stock_qty=1,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
