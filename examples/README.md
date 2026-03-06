# DopamineCore Examples

## basic_demo.py

No dependencies needed beyond `dopamine-core`. Shows how the engine adapts its context injection based on a sequence of wins and losses. Run it to see DopamineCore in action instantly.

```bash
pip install dopamine-core
python basic_demo.py
```

## openai_trader.py

Uses GPT-4 to make BTC UP/DOWN predictions with simulated outcomes. Shows how DopamineCore steers LLM behavior through subliminal context injection.

```bash
pip install dopamine-core openai
export OPENAI_API_KEY="sk-..."
python openai_trader.py
```

## moltrooms_bot.py

A complete autonomous trading bot for [moltrooms.ai](https://moltrooms.ai) — a BTC prediction arena on Base blockchain. Connects via WebSocket, places 1 USDC bets each round, and uses DopamineCore to develop adaptive trading behavior.

```bash
pip install dopamine-core openai web3 eth-account websockets aiohttp
export OPENAI_API_KEY="sk-..."
export MOLTROOMS_PRIVATE_KEY="0x..."
export MOLTROOMS_JWT="eyJ..."
python moltrooms_bot.py
```
