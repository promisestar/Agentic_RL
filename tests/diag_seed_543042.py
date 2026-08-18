"""Diagnostic: replay the same episode as seed=543042 and inspect
catalog.search results for the candidate agent queries.

Run with the project virtualenv:

    .venv/bin/python tests/diag_seed_543042.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import os

# Force offline + debug embedding off (use real embeddings).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ecom_rlve.server.openenv import EcomRLVEEnv  # noqa: E402


def main() -> None:
    env = EcomRLVEEnv(
        collection="C1",
        seed=42,
        config={
            "embedding_debug": False,
            "embedding_model": "/data_160TB/2024/hanshuaiteng/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/",
        },
    )
    env.dump_dir = ""
    env.trace_episodes = False

    obs = env.reset(env_id="PD", seed=543042)
    state = env._state
    hidden_goal = state.hidden_goal
    target_ids = list(hidden_goal.target_product_ids)
    print(f"hidden_goal.target_product_ids = {target_ids}")
    print(f"hidden_goal.constraints       = {hidden_goal.constraints}")
    print(f"hidden_goal.extra             = {hidden_goal.extra}")
    print()
    print(f"User message : {obs.conversation[-1]['content']!r}")
    print()

    # Verify the target product exists & passes the constraint.
    for tid in target_ids:
        p = state.products_by_id.get(tid)
        if p is None:
            print(f"!! target {tid} not in catalog")
            continue
        print(f"target product: id={p.id} cat={p.cat!r} price={p.price} "
              f"brand={p.brand} rating={p.rating} ship_days={p.ship_days}")
        # Check the constraint manually.
        for c in hidden_goal.constraints:
            attr, op, value = c["attr"], c["op"], c["value"]
            actual = getattr(p, attr, None) or p.attrs.get(attr)
            print(f"   constraint {attr} {op} {value}  ->  actual={actual!r}")
    print()

    # Now mimic each agent query from the dump and inspect the result.
    queries = [
        {"query": "electronics mobile tablets",
         "filters": {"price_max": 35.14, "category": ["electronics", "mobile", "tablets"]},
         "top_k": 20},
        {"query": "tablets",
         "filters": {"max_price": 35.14, "category": "electronics"},
         "top_k": 10},
        {"query": "tablets",
         "filters": {"category": "electronics", "subcategory": "mobile", "max_price": 35.14},
         "top_k": 10},
        {"query": "mobile tablets",
         "filters": {"price_max": 35.14, "category": "electronics"},
         "top_k": 10},
    ]
    for i, q in enumerate(queries, 1):
        print(f"--- query #{i}: {q['query']!r} filters={q['filters']} ---")
        action = {
            "assistant_message": "searching",
            "tool_calls": [{"name": "catalog.search", "args": q}],
            "answer": None,
        }
        import json
        obs2, reward, done, info = env.step(json.dumps(action))
        # The tool result should be in info or in conversation.
        # Look at the most recent assistant message for tool_results.
        from ecom_rlve.tools.catalog import catalog_search
        # Re-run the search directly via the tool for clearer inspection.
        results = catalog_search(
            query=q["query"],
            filters=q["filters"],
            top_k=q["top_k"],
            state=env._state,
        )
        if not results:
            print("   -> NO RESULTS (filters eliminated everything)")
        else:
            for card in results[:5]:
                cid = card.get("id") or card.get("product_id")
                cat = card.get("cat") or card.get("category")
                price = card.get("price")
                in_target = cid in target_ids
                marker = " <-- TARGET" if in_target else ""
                print(f"   - id={cid} cat={cat!r} price={price}{marker}")
            print(f"   ... ({len(results)} total; target in result: "
                  f"{any((c.get('id') or c.get('product_id')) in target_ids for c in results)})")
        print()


if __name__ == "__main__":
    main()