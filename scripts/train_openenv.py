#!/usr/bin/env python3
"""EcomRLVE-GYM OpenEnv RL Training Script.

Trains a language model to act as an e-commerce shopping assistant using
Reinforcement Learning (GRPO) with EcomRLVE-GYM environments as the
reward signal.  Follows the Unsloth + TRL GRPOTrainer pattern from the
OpenEnv 2048 notebook, adapted for multi-turn e-commerce conversation.

Usage:
    # Basic run with defaults (Qwen3-1.7B, C1, 300 steps)
    python scripts/train_openenv.py

    # Full options
    python scripts/train_openenv.py \
        --model Qwen/Qwen3-1.7B \
        --collection C1 \
        --max_steps 300 \
        --lora_rank 16 \
        --num_generations 4 \
        --load_in_4bit \
        --output_dir outputs/ecomrlve_grpo

    Each run writes under a timestamped subdirectory of --output_dir, e.g.
    outputs/ecomrlve_grpo/20260818_093012/{train.log,prompts.jsonl,final,...}.
    Pass --no_timestamp to use --output_dir as-is.

Requires (via uv; default index is Tsinghua):
    uv sync --extra train
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure ecom_rlve is importable (add src/ to path if needed)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ecom_rlve.server.openenv import EcomRLVEEnv
from ecom_rlve.server.state import parse_action
from ecom_rlve.training.collections import COLLECTIONS, get_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ecomrlve.train_openenv")


def _resolve_run_output_dir(base_dir: str, *, with_timestamp: bool) -> Path:
    """Resolve the per-run output directory and create it on disk.

    By default each training run gets its own subdirectory under ``base_dir``
    named ``YYYYMMDD_HHMMSS`` (local time), so ``train.log``, ``prompts.jsonl``,
    checkpoints and ``final/`` from different runs do not overwrite each other.

    When ``with_timestamp`` is False, ``base_dir`` itself is used (useful for
    resuming into a known path or for tests that expect a fixed location).
    """
    base = Path(base_dir).expanduser()
    if with_timestamp:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / stamp
    else:
        run_dir = base
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def _setup_log_file(log_path: Path) -> None:
    """Attach a FileHandler to the root logger so INFO+ goes to disk too.

    Stream handlers (console) are kept; we only add a file sink. Idempotent:
    if a previous handler for this path exists, leave it alone.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    # Avoid stacking handlers across re-runs in the same process.
    for existing in root.handlers:
        if (
            isinstance(existing, logging.FileHandler)
            and Path(getattr(existing, "baseFilename", "")) == log_path
        ):
            return
    root.addHandler(file_handler)


# ===================================================================
# System prompt (role + general workflow + terminal JSON protocol)
# Tool schemas are injected separately via apply_chat_template(tools=...).
# ===================================================================

SYSTEM_PROMPT = """\
You are a helpful e-commerce shopping assistant. Your goal is to help \
customers find products, manage carts and orders, handle returns, and \
answer store-policy questions.

Workflow (applies to every task and every tool):
1. Understand the user's goal and constraints.
2. Call tools from the available tool list when you need information or \
to change state. Arguments MUST strictly match each tool's JSON schema \
(types, required fields, and allowed parameter names). Do not invent \
parameter names.
3. Read tool results (including errors). Call additional tools if you \
still lack evidence.
4. Only when the evidence is sufficient, submit a final answer. Do not \
set done early.

When you are ready to finish (no further tool calls), reply with ONLY \
this JSON object — no <tool_call> blocks:

{
    "assistant_message": "your message to the user",
    "tool_calls": [],
    "answer": {"env": "<ENV_ID>", "recommended_product_ids": [], "done": true}
}

Set "done": true in the answer field. Include other answer fields \
required by the current environment when applicable.\
"""


# ===================================================================
# Environment wrapper — one persistent EcomRLVEEnv instance
# ===================================================================

class EcomRLVEOpenEnv:
    """Thin wrapper around EcomRLVEEnv that stores state for reward
    computation across the GRPO generate → reward pipeline.

    GRPOTrainer generates completions first, then calls reward functions.
    We must be able to reconstruct an episode for each (prompt, completion)
    pair.  Since each prompt is generated from a fresh env.reset(), we
    cache the Observation so the reward function can evaluate it.
    """

    def __init__(self, collection: str = "C1", seed: int = 42) -> None:
        self.env = EcomRLVEEnv(collection=collection, seed=seed, config={"embedding_debug": False, "embedding_model": "/data_160TB/2024/hanshuaiteng/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/",})
        self.env.dump_dir = ""       # Disable disk trace during training
        self.env.trace_episodes = False
        self.env.validate_rewards = True
        self.collection = collection
        self.env_ids = get_collection(collection)
        self._episode_counter = 0
        # Optional prompt/result sink for offline debugging. When set,
        # every sampled prompt and the matching reward result are
        # appended to the given path as JSONL. See ``_dump_records``.
        self.dump_path: Path | None = None

    # ------------------------------------------------------------------
    # Dump helpers (used only when --dump_prompts is set)
    # ------------------------------------------------------------------
    def _serialize_problem_params(self, params: Any) -> dict[str, Any]:
        """Best-effort JSON-safe copy of ProblemParams (or any mapping)."""
        if params is None:
            return {}

        # ``hasattr(params, "__dict__")`` is True for *any* object
        # including dicts, so handle plain mappings explicitly first.
        if isinstance(params, dict):
            out: dict[str, Any] = {}
            for key, value in params.items():
                if key.startswith("_"):
                    continue
                try:
                    json.dumps(value)
                    out[key] = value
                except (TypeError, ValueError):
                    out[key] = str(value)
            return out

        if hasattr(params, "__dict__"):
            out = {}
            for key, value in vars(params).items():
                if key.startswith("_"):
                    continue
                try:
                    json.dumps(value)
                    out[key] = value
                except (TypeError, ValueError):
                    out[key] = str(value)
            return out

        return {"raw": str(params)}

    def _dump_records(self, records: list[dict[str, Any]]) -> None:
        """Append ``records`` to ``self.dump_path`` as JSONL.

        We use a single-process trainer (single GPU / DataLoader worker=0)
        so a plain ``open(..., 'a')`` is safe — no extra locking needed.
        """
        if self.dump_path is None or not records:
            return
        try:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            with self.dump_path.open("a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as exc:
            # The dump is purely diagnostic — never fail training because
            # of a logging issue.
            logger.warning("[EcomRLVEOpenEnv] dump failed: %s", exc)

    # Public alias (handy for REPL / unit tests).
    dump_records = _dump_records

    def sample_prompt(self, tokenizer: Any) -> tuple[list[dict[str, str]], str, int]:
        """Reset the env and produce a chat-messages prompt.

        Returns:
            Tuple of (messages, env_id, episode_seed) where messages is
            a list of {"role": ..., "content": ...} dicts suitable for
            tokenizer.apply_chat_template(), and env_id + episode_seed
            can be used to deterministically re-create the exact same
            episode for reward evaluation.
        """
        # Cycle through envs uniformly
        env_id = self.env_ids[self._episode_counter % len(self.env_ids)]
        self._episode_counter += 1

        # Use a deterministic seed so the reward function can
        # reconstruct the exact same episode (same target products,
        # constraints, user message, etc.)
        episode_seed = self._episode_counter * 1000 + 42

        obs = self.env.reset(env_id=env_id, seed=episode_seed)

        # Build messages: [system, user]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        for msg in obs.conversation:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Optional debug dump: persist user message + hidden_goal so we
        # can sanity-check what the environment produced for each
        # training sample. See ``--dump_prompts``.
        if self.dump_path is not None:
            state = getattr(self.env, "_state", None)
            hidden_goal = getattr(state, "hidden_goal", None)
            n_tools = 0
            try:
                n_tools = len(self.env._tool_registry.get_tool_names())
            except Exception:
                n_tools = 0
            self._dump_records([{
                "kind": "prompt",
                "global_step": self._episode_counter,
                "env_id": env_id,
                "episode_seed": episode_seed,
                "messages": messages,
                "user_messages": [
                    m["content"] for m in obs.conversation
                    if m.get("role") == "user"
                ],
                "hidden_goal": self._serialize_problem_params(hidden_goal),
                "tools_injected": True,
                "n_tools": n_tools,
            }])

        return messages, env_id, episode_seed

    def evaluate_completion(
        self,
        completion: str,
        env_id: str,
        episode_seed: int,
    ) -> dict[str, Any]:
        """Run one episode: reset with the SAME env_id and seed that
        generated the prompt, then step with the completion.

        Because env.reset() is deterministic for a given (env_id, seed),
        this re-creates the exact same problem instance (same hidden
        goal, constraints, target products, user message, persona) that
        the model saw during generation.

        Args:
            completion:    The model's raw JSON action string.
            env_id:        Environment ID used when the prompt was created.
            episode_seed:  Seed used when the prompt was created.

        Returns:
            Dict with reward, is_correct, turn, termination_reason,
            and reward_breakdown.
        """
        # Re-create the exact same episode the model was prompted with
        obs = self.env.reset(env_id=env_id, seed=episode_seed)

        # Step with the model completion
        obs, reward, done, info = self.env.step(completion)

        # If the model didn't signal done, force a terminal answer
        if not done:
            action, valid = parse_action(completion)
            fallback_env = obs.env_id or env_id
            fallback_answer: dict[str, Any] = {"env": fallback_env, "done": True}
            if action and action.answer:
                fallback_answer = action.answer.model_dump()
                fallback_answer["done"] = True
            else:
                fallback_answer["recommended_product_ids"] = []

            done_action = json.dumps({
                "assistant_message": "Here is my final answer.",
                "tool_calls": [],
                "answer": fallback_answer,
            })
            obs, reward, done, info = self.env.step(done_action)

        return {
            "reward": reward,
            "is_correct": info.get("is_correct", False),
            "turn": info.get("turn", 0),
            "termination_reason": info.get("termination_reason", "unknown"),
            "reward_breakdown": info.get("reward_breakdown", {}),
        }


# ===================================================================
# Reward functions for GRPOTrainer
# ===================================================================

# Global env wrapper -- initialized in main()
_OPENENV: EcomRLVEOpenEnv | None = None
_PRINT_COUNTER: int = 0


def _extract_json_from_completion(text: str) -> str | None:
    """Extract the first JSON object from a completion string.

    Handles cases where the model wraps JSON in markdown code blocks
    or emits thinking tokens before the JSON.

    Returns None when the text looks like a Qwen XML tool_call (those
    should be parsed via ``parse_action`` on the raw string instead).
    """
    # Try direct JSON parse first
    text = text.strip()

    # Qwen native tool calls are not JSON — callers should pass raw text
    # to parse_action instead of using this helper.
    if "<tool_call>" in text.lower():
        return None

    # Strip markdown code fences if present
    if "```" in text:
        first = text.find("```") + 3
        second = text.find("```", first)
        if second > first:
            candidate = text[first:second].strip()
            candidate = candidate.removeprefix("json\n").removeprefix("json").strip()
            text = candidate

    # Find the first '{' and last '}'
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1 or end < start:
        return None

    candidate = text[start : end + 1]
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def _completion_for_env(response: str) -> str | None:
    """Normalize a model completion for ``env.step`` / ``parse_action``.

    Prefers the raw string when it contains Qwen ``<tool_call>`` XML
    (so the dual-format parser can see it). Otherwise extracts a JSON
    object for the legacy / terminal-answer path.
    """
    if "<tool_call>" in (response or "").lower():
        return response
    return _extract_json_from_completion(response)


def format_reward(completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
    """Reward: does the completion parse as a valid agent action?

    Accepts either Qwen-native XML ``<tool_call>`` blocks or the
    EcomRLVE terminal JSON protocol.

    +1.0  valid XML tool_call(s) or valid JSON with assistant_message
    -0.5  partially valid / missing required fields
    -2.0  unparseable
    """
    scores: list[float] = []
    for completion in completions:
        response = completion[0]["content"]
        action, valid = parse_action(response)
        if valid and action is not None:
            scores.append(1.0)
        elif _extract_json_from_completion(response) is not None:
            scores.append(-0.5)
        else:
            scores.append(-2.0)

    return scores


def tool_usage_reward(completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
    """Reward: does the completion use tools appropriately?

    +1.0  has well-formed tool_calls with valid tool names
    +0.5  has answer with done=true but no tool calls (acceptable for
          simple tasks)
    -0.5  has tool_calls but with invalid tool names
    -1.0  no parseable action
    """
    VALID_TOOL_PREFIXES = {
        "catalog.", "cart.", "order.", "return.", "policy.",
        "datetime.", "user.",
    }

    scores: list[float] = []
    for completion in completions:
        response = completion[0]["content"]
        action, valid = parse_action(response)
        if not valid or action is None:
            scores.append(-1.0)
            continue

        if action.tool_calls:
            all_valid = all(
                any(tc.name.startswith(prefix) for prefix in VALID_TOOL_PREFIXES)
                for tc in action.tool_calls
            )
            scores.append(1.0 if all_valid else -0.5)
        elif action.answer and action.answer.done:
            scores.append(0.5)
        else:
            # No tools and no done answer — unhelpful
            scores.append(-0.25)

    return scores


def env_reward(completions: list[list[dict[str, str]]], **kwargs: Any) -> list[float]:
    """Reward: run the completion through the EcomRLVE-GYM environment
    and return the environment's scalar reward.

    This is the core reward that evaluates actual e-commerce task
    performance: product recommendation quality, cart correctness,
    return handling, etc.

    TRL GRPOTrainer passes extra dataset columns through **kwargs,
    so we receive `env_id` and `episode_seed` lists that let us
    reconstruct the exact same episode the prompt came from.

    Reward range: [-1.0, 1.0] from the environment, scaled by 5.0
    to make it the dominant signal.
    """
    global _OPENENV, _PRINT_COUNTER
    assert _OPENENV is not None, "EcomRLVEOpenEnv not initialized"

    # Extract episode identifiers from kwargs (set in the dataset)
    env_ids = kwargs.get("env_id", [])
    episode_seeds = kwargs.get("episode_seed", [])

    scores: list[float] = []
    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        extracted = _completion_for_env(response)

        should_print = (_PRINT_COUNTER % 10 == 0)
        _PRINT_COUNTER += 1

        if extracted is None:
            if should_print:
                logger.info("[env_reward] No parseable action found in completion")
            scores.append(-1.0)
            continue

        # Recover the env_id and seed for this sample
        eid = env_ids[i] if i < len(env_ids) else _OPENENV.env_ids[0]
        eseed = episode_seeds[i] if i < len(episode_seeds) else 42

        try:
            result = _OPENENV.evaluate_completion(
                completion=extracted,
                env_id=eid,
                episode_seed=int(eseed),
            )
            env_score = result["reward"]

            # Persist the model's verdict for this episode alongside the
            # prompt we already wrote in ``sample_prompt``. Sharing the
            # ``episode_seed`` is enough to join the two rows offline.
            if _OPENENV.dump_path is not None:
                breakdown = result.get("reward_breakdown", {}) or {}
                # Keep the model output on disk too so you can see what
                # the agent actually emitted, not just the scalar reward.
                # We store the *raw* response (pre-extraction) so we
                # still have the markdown / thinking wrappers if any,
                # and we truncate to ~8KB to bound the file size.
                raw_response = response if isinstance(response, str) else str(response)
                truncated = raw_response[:8192]
                if len(raw_response) > 8192:
                    truncated += f"\n... [truncated, original length={len(raw_response)}]"
                _OPENENV._dump_records([{
                    "kind": "result",
                    "env_id": eid,
                    "episode_seed": int(eseed),
                    "completion_index": i,
                    "completion_raw": truncated,
                    "completion_extracted": extracted,
                    "reward_raw": float(env_score),
                    "reward_scaled": float(env_score) * 5.0,
                    "is_correct": bool(result["is_correct"]),
                    "turn": int(result.get("turn", 0)),
                    "termination_reason": result.get(
                        "termination_reason", "unknown"
                    ),
                    "reward_breakdown": {
                        k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in breakdown.items()
                    },
                }])

            if should_print:
                logger.info(
                    "[env_reward] env=%s seed=%d reward=%.4f correct=%s reason=%s breakdown=%s",
                    eid,
                    eseed,
                    env_score,
                    result["is_correct"],
                    result["termination_reason"],
                    {k: f"{v:.3f}" if isinstance(v, float) else v
                     for k, v in result.get("reward_breakdown", {}).items()
                     if k in ("r_task", "r_eff", "r_hall", "r_total")},
                )

            # Scale the env reward to dominate the combined signal
            scores.append(float(env_score) * 5.0)

        except Exception as exc:
            logger.warning("[env_reward] Exception: %s: %s", type(exc).__name__, exc)
            scores.append(-3.0)

    return scores


# ===================================================================
# Tokenizer / Processor compatibility
# ===================================================================


def wrap_tokenizer_with_tools(tokenizer: Any, tools: list[dict[str, Any]]) -> Any:
    """Rebind ``tokenizer`` so ``apply_chat_template`` always gets ``tools=``.

    TRL / Unsloth ``GRPOTrainer`` requires ``processing_class`` to be an
    instance of ``PreTrainedTokenizerBase`` or ``ProcessorMixin``
    (``isinstance`` check). A plain wrapper class fails that check.

    Strategy (方案一): dynamically subclass the *real* tokenizer class and
    reassign ``tokenizer.__class__``. The instance identity is preserved,
    so ``isinstance(tokenizer, PreTrainedTokenizerBase)`` still holds, while
    ``apply_chat_template`` injects the OpenAI-format tool schemas that
    Qwen chat templates render into ``<tools>...</tools>``.
    """
    # Already wrapped in a previous call — just refresh the tool list.
    if getattr(tokenizer, "_ecom_rlve_tools_wrapped", False):
        tokenizer._ecom_rlve_tools = tools
        return tokenizer

    base_cls = type(tokenizer)

    class ToolsAwareTokenizer(base_cls):  # type: ignore[misc,valid-type]
        """Subclass of the concrete tokenizer with tools= injection."""

        def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
            tool_list = getattr(self, "_ecom_rlve_tools", None)
            if tool_list is not None:
                kwargs.setdefault("tools", tool_list)
            return super().apply_chat_template(*args, **kwargs)

    tokenizer._ecom_rlve_tools = tools
    tokenizer._ecom_rlve_tools_wrapped = True
    tokenizer.__class__ = ToolsAwareTokenizer
    return tokenizer


# Backward-compatible name used by unit tests / callers that expect a class.
# Prefer ``wrap_tokenizer_with_tools`` for production use.
class ToolsAwareTokenizer:
    """Deprecated constructor shim — delegates to ``wrap_tokenizer_with_tools``.

    Kept so existing tests that do ``ToolsAwareTokenizer(tok, tools)`` still
    work. Prefer calling ``wrap_tokenizer_with_tools`` directly.
    """

    def __new__(cls, tokenizer: Any, tools: list[dict[str, Any]]) -> Any:
        return wrap_tokenizer_with_tools(tokenizer, tools)


def resolve_text_tokenizer(processing_or_tokenizer: Any) -> Any:
    """Return an object with text tokenizer APIs (encode / chat template).

    Unsloth may return a multimodal Processor for some checkpoints
    (e.g. Qwen3.5 → ``Qwen3VLProcessor``). Those expose ``.tokenizer``
    but not top-level ``.encode()``. Prefer the inner tokenizer when
    present so the rest of this script can assume a text tokenizer.
    """
    inner = getattr(processing_or_tokenizer, "tokenizer", None)
    if inner is not None and callable(getattr(inner, "encode", None)):
        logger.info(
            "Resolved text tokenizer from %s → %s",
            type(processing_or_tokenizer).__name__,
            type(inner).__name__,
        )
        return inner
    return processing_or_tokenizer


def apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render chat messages to a plain string via chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def count_tokens(tokenizer: Any, text: str) -> int:
    """Count tokens for a string across tokenizer / processor variants."""
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        return len(encode(text))

    # Fallback: some processors only tokenize via __call__
    encoded = tokenizer(text)
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if input_ids is None and hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    if input_ids is None:
        raise TypeError(
            f"Cannot count tokens with {type(tokenizer).__name__}: "
            "no encode() and no input_ids from __call__"
        )
    # input_ids may be nested for batched processor output
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        return len(input_ids[0])
    return len(input_ids)


# ===================================================================
# Dataset builder
# ===================================================================

def build_dataset(
    openenv: EcomRLVEOpenEnv,
    tokenizer: Any,
    n_prompts: int = 1000,
) -> "Dataset":
    """Build a HuggingFace Dataset of prompts sampled from EcomRLVE-GYM.

    Each row has:
        - 'prompt':        list of chat messages for apply_chat_template()
        - 'env_id':        environment ID (e.g. "PD", "SUB", ...)
        - 'episode_seed':  deterministic seed for env.reset()

    TRL GRPOTrainer passes extra columns through to reward functions
    as **kwargs, allowing the reward function to reconstruct the exact
    same episode that generated each prompt.
    """
    from datasets import Dataset

    rows: list[dict[str, Any]] = []
    for i in range(n_prompts):
        messages, env_id, episode_seed = openenv.sample_prompt(tokenizer)
        rows.append({
            "prompt": messages,
            "env_id": env_id,
            "episode_seed": episode_seed,
        })

    dataset = Dataset.from_list(rows)
    logger.info(
        "Built dataset with %d prompts (envs: %s)",
        len(dataset),
        ", ".join(sorted(set(r["env_id"] for r in rows))),
    )
    return dataset


# ===================================================================
# Main training loop
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EcomRLVE-GYM OpenEnv RL Training with GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B",
        help="HuggingFace model name or path (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", default=True,
        help="Load model in 4-bit quantization (default: True)",
    )
    parser.add_argument(
        "--load_in_16bit", action="store_true", default=False,
        help="Load model in 16-bit (overrides --load_in_4bit)",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--fast_inference", action="store_true", default=False,
        help=(
            "Enable Unsloth/vLLM fast inference for GRPO generation. "
            "Requires a separately installed vLLM compatible with the "
            "current transformers pin; default False (needed for Qwen3.5)."
        ),
    )

    # Environment
    parser.add_argument(
        "--collection", type=str, default="C1",
        choices=sorted(COLLECTIONS.keys()),
        help="Environment collection (default: C1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n_prompts", type=int, default=1000,
        help="Number of training prompts to generate (default: 1000)",
    )

    # Training
    parser.add_argument(
        "--max_steps", type=int, default=300,
        help="Maximum training steps (default: 300)",
    )
    parser.add_argument(
        "--num_generations", type=int, default=4,
        help="GRPO group size G (default: 4)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Per-device train batch size (default: 1)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=None,
        help="Max prompt length in tokens (auto-detected if None)",
    )
    parser.add_argument(
        "--max_completion_length", type=int, default=512,
        help="Max completion length in tokens (default: 512)",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="outputs/ecomrlve_grpo",
        help=(
            "Base directory for run artifacts (default: outputs/ecomrlve_grpo). "
            "Unless --no_timestamp is set, a YYYYMMDD_HHMMSS subdirectory is "
            "created under this path for each training run."
        ),
    )
    parser.add_argument(
        "--no_timestamp",
        action="store_true",
        default=False,
        help=(
            "Use --output_dir directly instead of creating a timestamped "
            "subdirectory. Useful when resuming into a known path."
        ),
    )
    parser.add_argument(
        "--save_steps", type=int, default=50,
        help="Save checkpoint every N steps (default: 50)",
    )
    parser.add_argument(
        "--report_to", type=str, default="none",
        choices=["none", "wandb", "tensorboard", "trackio"],
        help="Experiment tracker (default: none)",
    )
    parser.add_argument(
        "--log_to_file", action="store_true", default=False,
        help=(
            "Also tee training logs to a file. Default path is "
            "<run_output_dir>/train.log (created if missing)."
        ),
    )
    parser.add_argument(
        "--log_path", type=str, default=None,
        help=(
            "Explicit log file path. Overrides the default "
            "<run_output_dir>/train.log. Implies --log_to_file."
        ),
    )
    parser.add_argument(
        "--dump_prompts", action="store_true", default=False,
        help=(
            "Append every sampled prompt (messages + hidden_goal) and "
            "matching completion result to a JSONL file. Useful for "
            "offline verification that the environment is generating "
            "correct user messages and ground-truth targets."
        ),
    )
    parser.add_argument(
        "--dump_path", type=str, default=None,
        help=(
            "Path to the prompt/result dump JSONL file. Defaults to "
            "<run_output_dir>/prompts.jsonl when --dump_prompts is set. "
            "Existing files are truncated at start-up."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Materialize a per-run directory first so train.log / prompts.jsonl /
    # checkpoints / final all land in the same place and never clobber a
    # previous training run's artifacts.
    run_output_dir = _resolve_run_output_dir(
        args.output_dir,
        with_timestamp=not args.no_timestamp,
    )
    args.output_dir = str(run_output_dir)

    if args.log_path:
        log_file = Path(args.log_path).expanduser().resolve()
        _setup_log_file(log_file)
        logger.info("Logging to file: %s", log_file)
    elif args.log_to_file:
        log_file = run_output_dir / "train.log"
        _setup_log_file(log_file)
        logger.info("Logging to file: %s", log_file)

    logger.info("=" * 70)
    logger.info("EcomRLVE-GYM OpenEnv Training")
    logger.info("=" * 70)
    logger.info("Model:      %s", args.model)
    logger.info("Collection: %s -> %s", args.collection, get_collection(args.collection))
    logger.info("LoRA rank:  %d", args.lora_rank)
    logger.info("Max steps:  %d", args.max_steps)
    logger.info("Group size: %d (G)", args.num_generations)
    logger.info("LR:         %s", args.learning_rate)
    logger.info("Output:     %s", args.output_dir)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Initialize EcomRLVE-GYM environment
    # ------------------------------------------------------------------
    logger.info("Initializing EcomRLVE-GYM environment (collection=%s)...", args.collection)
    global _OPENENV
    _OPENENV = EcomRLVEOpenEnv(collection=args.collection, seed=args.seed)
    logger.info(
        "Environment ready: %d envs, catalog loaded",
        len(_OPENENV.env_ids),
    )

    # Configure the optional prompt/result dump. We truncate the file
    # at start-up so each run starts fresh — useful when iterating on
    # reward shaping or dataset fixes.
    if args.dump_prompts:
        dump_path = Path(
            args.dump_path if args.dump_path
            else Path(args.output_dir) / "prompts.jsonl"
        ).expanduser().resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate; subsequent appends are O(1) line writes.
        dump_path.write_text("", encoding="utf-8")
        _OPENENV.dump_path = dump_path
        logger.info(
            "Prompt/result dump enabled → %s (records per sample + reward result)",
            dump_path,
        )

    # ------------------------------------------------------------------
    # 2. Load model with Unsloth
    # ------------------------------------------------------------------
    logger.info("Loading model: %s ...", args.model)

    from unsloth import FastLanguageModel

    load_in_4bit = args.load_in_4bit and not args.load_in_16bit

    if args.fast_inference:
        logger.info("fast_inference=True (requires a compatible vLLM install)")
    else:
        logger.info(
            "fast_inference=False (default). Pass --fast_inference only if "
            "vLLM is installed and compatible with transformers<=5.5.0."
        )

    model, processing = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=args.fast_inference,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=0.6,  # Reduce if OOM
    )
    # Qwen3.5 etc. may return a VL Processor; unwrap to a text tokenizer.
    tokenizer = resolve_text_tokenizer(processing)

    # Inject OpenAI-format tool schemas into every apply_chat_template call
    # so GRPOTrainer (which does not pass tools= itself) still renders the
    # Qwen <tools>...</tools> block with full parameter constraints.
    # wrap_tokenizer_with_tools rebinds __class__ to a subclass of the real
    # tokenizer so isinstance(..., PreTrainedTokenizerBase) still passes.
    openai_tools = _OPENENV.env._tool_registry.to_openai_tools()
    tokenizer = wrap_tokenizer_with_tools(tokenizer, openai_tools)
    logger.info(
        "Injected %d tools into chat template (OpenAI tools protocol; class=%s)",
        len(openai_tools),
        type(tokenizer).__name__,
    )

    logger.info("Model loaded. Adding LoRA adapters (rank=%d)...", args.lora_rank)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ------------------------------------------------------------------
    # 3. Build the training dataset
    # ------------------------------------------------------------------
    logger.info("Building training dataset (%d prompts)...", args.n_prompts)
    dataset = build_dataset(_OPENENV, tokenizer, n_prompts=args.n_prompts)

    # Compute max_prompt_length from a sample if not specified
    if args.max_prompt_length is None:
        sample_text = apply_chat_template_text(
            tokenizer,
            dataset[0]["prompt"],
            add_generation_prompt=True,
        )
        sample_len = count_tokens(tokenizer, sample_text)
        # Add 20% headroom
        args.max_prompt_length = int(sample_len * 1.2) + 1
        logger.info(
            "Auto-detected max_prompt_length=%d (sample=%d tokens)",
            args.max_prompt_length, sample_len,
        )

    max_completion_length = min(
        args.max_completion_length,
        args.max_seq_length - args.max_prompt_length,
    )
    if max_completion_length <= 0:
        logger.error(
            "max_seq_length (%d) is too small for max_prompt_length (%d). "
            "Increase --max_seq_length or decrease --max_prompt_length.",
            args.max_seq_length, args.max_prompt_length,
        )
        sys.exit(1)

    logger.info(
        "Sequence budget: prompt=%d + completion=%d = %d / %d",
        args.max_prompt_length,
        max_completion_length,
        args.max_prompt_length + max_completion_length,
        args.max_seq_length,
    )

    # ------------------------------------------------------------------
    # 4. Configure GRPO training
    # ------------------------------------------------------------------
    logger.info("Configuring GRPOTrainer...")

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        # Generation
        temperature=args.temperature,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=args.num_generations,

        # Optimization
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",

        # Batching
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Schedule
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=args.save_steps,

        # Output
        output_dir=args.output_dir,
        report_to=args.report_to,

        # Misc
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    # ------------------------------------------------------------------
    # 5. Create trainer with EcomRLVE reward functions
    # ------------------------------------------------------------------
    logger.info("Creating GRPOTrainer with 3 reward functions...")
    logger.info("  1. format_reward:     valid XML tool_call or terminal JSON")
    logger.info("  2. tool_usage_reward: correct tool names & structure")
    logger.info("  3. env_reward:        EcomRLVE-GYM environment reward (×5)")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward,
            tool_usage_reward,
            env_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # ------------------------------------------------------------------
    # 6. Train!
    # ------------------------------------------------------------------
    logger.info("Starting GRPO training...")
    logger.info(
        "Expect slow initial progress — the model needs ~50-100 steps "
        "to learn JSON formatting before env rewards improve."
    )
    t0 = time.monotonic()

    trainer.train()

    elapsed = time.monotonic() - t0
    logger.info("Training completed in %.1f minutes (%d steps)", elapsed / 60, args.max_steps)

    # ------------------------------------------------------------------
    # 7. Save final model
    # ------------------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    logger.info("Saving final LoRA adapters to %s ...", final_dir)
    model.save_pretrained(final_dir)
    # Prefer saving the original Processor when Unsloth returned one;
    # otherwise save the text tokenizer.
    if processing is not tokenizer and hasattr(processing, "save_pretrained"):
        processing.save_pretrained(final_dir)
    else:
        tokenizer.save_pretrained(final_dir)

    # ------------------------------------------------------------------
    # 8. Quick inference test
    # ------------------------------------------------------------------
    logger.info("Running quick inference test...")
    test_messages, _, _ = _OPENENV.sample_prompt(tokenizer)
    text = apply_chat_template_text(
        tokenizer,
        test_messages,
        add_generation_prompt=True,
    )

    # Switch from training mode to inference mode
    FastLanguageModel.for_inference(model)

    from transformers import TextStreamer
    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        temperature=0.7,
        max_new_tokens=max_completion_length,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

    logger.info("=" * 70)
    logger.info("Training complete! Model saved to: %s", final_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
