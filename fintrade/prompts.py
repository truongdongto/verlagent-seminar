"""Prompt templates and action parsing for the trading agent.

The templates follow the ALFWorld / WebShop templates used in the HGPO paper
(Fig. 5-6). They live here, not only in agent_system/environments/prompts, so
that standalone evaluation (local checkpoints, Claude, Gemini) builds exactly
the same prompts as RL training.
"""

import re
from typing import List, Optional, Tuple

from .sim import POSITION_LEVELS

OBJECTIVES = {
    "log_return": "maximize the portfolio's total return",
    "risk_adjusted": "maximize the portfolio's total return while avoiding drawdowns",
    "excess_terminal": "end the episode with a higher return than simply buying and holding Stock A",
}

TASK_TEMPLATE = (
    "manage a portfolio made of cash and one US stock (called Stock A) for {window_len} trading days. "
    "At the close of each trading day you set the target position in Stock A, which is held until the next close. "
    "Your goal is to {objective}. Changing the position costs {cost_bps:g} basis points of the traded amount"
)

ADMISSIBLE_ACTIONS = ", ".join(f"'{p}'" for p in POSITION_LEVELS)

# --------------------- Trading --------------------- #
TRADING_TEMPLATE_NO_HIS = """
You are an expert trading agent operating in a daily stock trading environment.
Your task is to: {task_description}.
Your current observation is:
{current_observation}
Your admissible actions of the current situation are: [{admissible_actions}] (the percentage of capital to hold in Stock A).

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

TRADING_TEMPLATE = """
You are an expert trading agent operating in a daily stock trading environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is:
{current_observation}
Your admissible actions of the current situation are: [{admissible_actions}] (the percentage of capital to hold in Stock A).

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


def task_description(window_len: int, reward_type: str, cost_bps: float) -> str:
    return TASK_TEMPLATE.format(window_len=window_len, objective=OBJECTIVES[reward_type], cost_bps=cost_bps)


def build_prompt(
    current_observation: str,
    task: str,
    action_history: str = "",
    history_length: int = 0,
    step_count: int = 0,
) -> str:
    if history_length <= 0 or step_count == 0:
        return TRADING_TEMPLATE_NO_HIS.format(
            task_description=task,
            current_observation=current_observation,
            admissible_actions=ADMISSIBLE_ACTIONS,
        )
    return TRADING_TEMPLATE.format(
        task_description=task,
        step_count=step_count,
        history_length=history_length,
        action_history=action_history,
        current_step=step_count + 1,
        current_observation=current_observation,
        admissible_actions=ADMISSIBLE_ACTIONS,
    )


_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def parse_action(text: str) -> Tuple[Optional[int], int]:
    """Return (position level or None, valid flag). Requires both <think> and <action> blocks."""
    matches = _ACTION_RE.findall(text)
    if not matches:
        return None, 0
    numbers = _NUMBER_RE.findall(matches[-1])
    if len(numbers) != 1:
        return None, 0
    value = float(numbers[0])
    if value not in POSITION_LEVELS:
        return None, 0
    if "<think>" not in text or "</think>" not in text:
        return int(value), 0
    return int(value), 1


def parse_actions(texts: List[str]) -> Tuple[List[Optional[int]], List[int]]:
    parsed = [parse_action(t) for t in texts]
    return [p for p, _ in parsed], [v for _, v in parsed]
