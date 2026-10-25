from typing import List

from fintrade.prompts import parse_actions


def trading_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>0/50/100</action>
    Returns the target position level (None if it cannot be parsed, in which case the
    environment keeps the current position) and a validity flag. A missing <think> block
    keeps the parsed action but marks it invalid, as in sokoban_projection.
    """
    return parse_actions(actions)
