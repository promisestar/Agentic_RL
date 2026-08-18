"""Smoke test for the --dump_prompts feature in train_openenv.py.

We don't instantiate a real ``EcomRLVEEnv`` here (that pulls in the
full catalog + embedding stack); instead we patch ``EcomRLVEOpenEnv``
methods with lightweight stand-ins and verify that the dump
machinery itself writes well-formed JSONL records.

The goal is narrow: catch regressions in the JSONL format that the
user will use to verify whether the environment is producing
sensible user messages and hidden goals.

Run with the project virtualenv:

    .venv/bin/python tests/test_dump_prompts.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Stub heavy imports so ``import train_openenv`` doesn't pull in
# Unsloth/TRL/the real torch spec.
unsloth_stub = types.ModuleType("unsloth")


class _FastLanguageModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return MagicMock(), MagicMock()


unsloth_stub.FastLanguageModel = _FastLanguageModel
sys.modules["unsloth"] = unsloth_stub

trl_stub = types.ModuleType("trl")
trl_stub.GRPOTrainer = MagicMock()
trl_stub.GRPOConfig = MagicMock()
sys.modules["trl"] = trl_stub

datasets_stub = types.ModuleType("datasets")
datasets_stub.Dataset = MagicMock()
sys.modules["datasets"] = datasets_stub

torch_stub = types.ModuleType("torch")
torch_stub.float32 = "float32"
torch_stub.bfloat16 = "bfloat16"
sys.modules["torch"] = torch_stub

try:
    import train_openenv as toe
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Failed to import train_openenv: {exc}")


class _FakeState:
    """Minimal stand-in for ``EpisodeState`` with just hidden_goal."""

    def __init__(self, hidden_goal: dict) -> None:
        self.hidden_goal = hidden_goal


class _FakeObservation:
    def __init__(self, conversation: list[dict[str, str]]) -> None:
        self.conversation = conversation


class DumpPromptsTests(unittest.TestCase):
    def _make_env(self, dump_path: Path) -> toe.EcomRLVEOpenEnv:
        """Build an EcomRLVEOpenEnv instance *without* running
        EcomRLVEEnv.__init__ (which needs torch + sentence-transformers)."""
        env = toe.EcomRLVEOpenEnv.__new__(toe.EcomRLVEOpenEnv)
        env.collection = "C1"
        env.env_ids = ["PD", "CD", "OD", "RD"]  # from get_collection("C1")
        env._episode_counter = 0
        # Stub the bits the new code path touches.
        env.env = MagicMock()
        env.env.reset.return_value = _FakeObservation(
            conversation=[
                {"role": "user",
                 "content": "I'm looking for running shoes under $100."},
            ]
        )
        env.env._state = _FakeState(
            hidden_goal={
                "env_id": "PD",
                "difficulty": 1,
                "seed": 1042,
                "target_product_ids": ["P000123"],
                "constraints": [{"type": "price", "op": "<=", "value": 100}],
                "extra": {},
            }
        )
        env.dump_path = dump_path
        return env

    def test_sample_prompt_writes_prompt_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dump_path = Path(tmp) / "prompts.jsonl"
            env = self._make_env(dump_path)

            tokenizer = MagicMock()
            messages, env_id, episode_seed = env.sample_prompt(tokenizer)

            self.assertTrue(dump_path.exists())
            with dump_path.open("r", encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(records), 1)

            prompt = records[0]
            self.assertEqual(prompt["kind"], "prompt")
            self.assertEqual(prompt["env_id"], env_id)
            self.assertEqual(prompt["episode_seed"], episode_seed)
            self.assertEqual(prompt["user_messages"],
                             ["I'm looking for running shoes under $100."])
            self.assertEqual(prompt["hidden_goal"]["target_product_ids"],
                             ["P000123"])
            self.assertEqual(prompt["hidden_goal"]["constraints"],
                             [{"type": "price", "op": "<=", "value": 100}])

    def test_dump_records_is_idempotent_and_appends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dump_path = Path(tmp) / "prompts.jsonl"
            env = self._make_env(dump_path)

            env.dump_records([
                {"kind": "prompt", "env_id": "PD", "episode_seed": 1},
            ])
            env.dump_records([
                {"kind": "result", "env_id": "PD", "episode_seed": 1,
                 "reward_raw": 0.5},
            ])

            with dump_path.open("r", encoding="utf-8") as fh:
                lines = [line for line in fh if line.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["kind"], "prompt")
            self.assertEqual(json.loads(lines[1])["kind"], "result")

    def test_dump_records_is_noop_when_path_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dump_path = Path(tmp) / "prompts.jsonl"
            env = self._make_env(dump_path)
            env.dump_path = None
            # Should not raise, should not write anything anywhere.
            env.dump_records([{"kind": "prompt"}])
            self.assertFalse(dump_path.exists())

    def test_serialize_problem_params_handles_dataclass_like(self) -> None:
        env = self._make_env(Path("/tmp/whatever.jsonl"))
        from types import SimpleNamespace
        params = SimpleNamespace(env_id="PD", target=["P1"], extra={"k": 1})
        out = env._serialize_problem_params(params)
        self.assertEqual(out["env_id"], "PD")
        self.assertEqual(out["target"], ["P1"])
        self.assertEqual(out["extra"], {"k": 1})

    def test_result_record_includes_completion_payload(self) -> None:
        """Verify the schema written by the env_reward dump path
        carries the raw completion string the model emitted, plus the
        JSON we extracted from it. Both are needed when debugging why
        the agent didn't find the right products.
        """
        with tempfile.TemporaryDirectory() as tmp:
            dump_path = Path(tmp) / "prompts.jsonl"
            env = self._make_env(dump_path)

            completion_raw = (
                "Here is my plan.\n"
                "```json\n"
                '{"assistant_message": "done", "tool_calls": [], '
                '"answer": {"env": "PD", "done": true, '
                '"recommended_product_ids": []}}\n'
                "```"
            )
            completion_extracted = (
                '{"assistant_message": "done", "tool_calls": [], '
                '"answer": {"env": "PD", "done": true, '
                '"recommended_product_ids": []}}'
            )

            env.dump_records([{
                "kind": "result",
                "env_id": "PD",
                "episode_seed": 7,
                "completion_index": 0,
                "completion_raw": completion_raw,
                "completion_extracted": completion_extracted,
                "reward_raw": -0.5,
                "reward_scaled": -2.5,
                "is_correct": False,
                "turn": 1,
                "termination_reason": "agent_done",
                "reward_breakdown": {"r_task": -0.9, "r_total": -0.5},
            }])

            with dump_path.open("r", encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec["kind"], "result")
            self.assertIn("completion_raw", rec)
            self.assertIn("completion_extracted", rec)
            self.assertEqual(rec["completion_raw"], completion_raw)
            self.assertEqual(rec["completion_extracted"], completion_extracted)
            # The extracted JSON must itself be parseable.
            json.loads(rec["completion_extracted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)