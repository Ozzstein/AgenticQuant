---
name: analyze-ticker
description: Run the full multi-agent analysis on a stock ticker using QuantAgentLab's TradingDesk
disable-model-invocation: true
---

Run the multi-agent investment analysis on a given ticker.

Parse the user's arguments to extract:
- First positional arg: ticker symbol (e.g., AAPL, NVDA, MSFT). Required.
- `--provider` or `-p`: LLM provider (anthropic, openai, ollama). Default: anthropic
- `--output` or `-o`: output format (rich, json, csv). Default: rich
- `--agents`: comma-separated list of agents to enable (fundamental,sentiment,technical,risk). Default: all

Then execute:
```
conda run -n aiquant python scripts/run_agents.py analyze <TICKER> --output <format>
```

Display the result as:
- **Decision**: BUY/SELL/HOLD/STRONG_BUY/STRONG_SELL with confidence %
- **Target Price** and **Stop Loss**
- **Position Size**: recommended % of portfolio
- **Time Horizon**: e.g., 1M, 3M, 6M
- **Reasoning**: key thesis in bullet points
- **Risk Flags**: list of identified risks
- **Catalysts**: upcoming catalysts

If the analysis fails due to missing API key, remind the user to set `ANTHROPIC_API_KEY` in `config/.env`.
