# Single-stock trading environment and baselines for comparing agentic RL
# algorithms (PPO / RLOO / GRPO / DAPO / GSPO / GiGPO / HGPO) on financial
# decision-making. Kept free of torch / verl imports so that data preparation,
# classic DRL baselines and analysis can run on a CPU-only machine.
#
#   fintrade.data       download prices + technical features (python -m fintrade.data)
#   fintrade.sim        TradingSim shared by the LLM env and every baseline
#   fintrade.prompts    prompt templates and <action> parsing
#   fintrade.metrics    Sharpe / drawdown / win-rate metrics
#   fintrade.diagnose   environment sanity report (python -m fintrade.diagnose)
#   fintrade.baselines  rule-based, classic DRL (SB3) and API LLM baselines
