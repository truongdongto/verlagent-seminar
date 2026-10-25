# The trading templates are defined in fintrade.prompts so that standalone
# evaluation (local checkpoints, Claude, Gemini) builds identical prompts.
from fintrade.prompts import TRADING_TEMPLATE_NO_HIS, TRADING_TEMPLATE

__all__ = ["TRADING_TEMPLATE_NO_HIS", "TRADING_TEMPLATE"]
