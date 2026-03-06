"""
DopamineCore + Moltrooms.ai — Autonomous BTC Prediction Bot

A complete trading bot for the Moltrooms prediction arena with
DopamineCore providing intrinsic motivation. The bot connects via
WebSocket, listens for rounds, and places 1 USDC bets on UP/DOWN.

Prerequisites:
    pip install dopamine-core openai web3 eth-account websockets aiohttp

    Set environment variables:
        OPENAI_API_KEY=sk-...
        MOLTROOMS_PRIVATE_KEY=0x...
        MOLTROOMS_JWT=eyJ...

Usage:
    python moltrooms_bot.py

The bot will:
1. Connect to Moltrooms WebSocket
2. On each round: inject DopamineCore context → ask LLM → place bet
3. On round result: update DopamineCore with win/loss
4. Adapt behavior naturally over time
"""

import asyncio
import json
import os
import time

import aiohttp
from openai import OpenAI
from web3 import Web3

from dopamine_core import DopamineEngine, DopamineConfig

# --- Configuration ---
API_BASE = "https://api.moltrooms.ai"
WS_URL = "wss://api.moltrooms.ai/ws"
BASE_RPC = "https://mainnet.base.org"
USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
PLATFORM_WALLET = "0xf7ac902d82BbC53206906af08113351936087E8a"

USDC_ABI = [
    {
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]


class MoltroomsBot:
    def __init__(self) -> None:
        self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.private_key = os.environ["MOLTROOMS_PRIVATE_KEY"]
        self.jwt = os.environ["MOLTROOMS_JWT"]

        self.w3 = Web3(Web3.HTTPProvider(BASE_RPC))
        self.account = self.w3.eth.account.from_key(self.private_key)

        # DopamineCore engine — this is where the magic happens
        config = DopamineConfig()
        config.injection.verbosity = "moderate"
        self.engine = DopamineEngine(config)

        # Track state
        self.last_response_text: str = ""
        self.last_decision: str = ""
        self.round_bets: dict[int, str] = {}  # round_id -> side

    async def run(self) -> None:
        """Main loop: connect to WebSocket and play forever."""
        print("Moltrooms Bot with DopamineCore starting...")
        print(f"Wallet: {self.account.address}")

        async with aiohttp.ClientSession() as session:
            self.session = session
            async with session.ws_connect(WS_URL) as ws:
                # Authenticate
                await ws.send_str(self.jwt)
                print("Connected and authenticated.")

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        event = json.loads(msg.data)
                        await self.handle_event(event)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"WebSocket error: {ws.exception()}")
                        break

    async def handle_event(self, event: dict) -> None:
        """Route WebSocket events."""
        event_type = event.get("type")
        data = event.get("data", {})

        if event_type == "round_created":
            await self.on_round_created(data)
        elif event_type == "round_settled":
            await self.on_round_settled(data)
        elif event_type == "payout_confirmed":
            self.on_payout(data)
        elif event_type == "refund_confirmed":
            print(f"  Refund received: ${float(data.get('amount', 0)):.2f}")

    async def on_round_created(self, data: dict) -> None:
        """New round opened — analyze, decide, and bet."""
        round_id = data["round_id"]
        open_price = data["open_price"]
        print(f"\n--- Round {round_id} | BTC: ${open_price} ---")

        # Check USDC balance
        usdc = self.w3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS),
            abi=[{
                "inputs": [{"name": "account", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function",
            }],
        )
        balance = usdc.functions.balanceOf(self.account.address).call() / 1_000_000
        if balance < 1.0:
            print(f"  Insufficient USDC ({balance:.2f}). Skipping.")
            return

        # Build prompt WITH DopamineCore motivation
        base_prompt = (
            f"BTC/USD is currently at ${open_price}. "
            f"This is a 1-minute prediction: will BTC close UP or DOWN?\n"
            f"Analyze and explain your reasoning. End with: UP or DOWN."
        )
        prompt = self.engine.inject_context(base_prompt)

        # Ask LLM
        response = self.openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        self.last_response_text = response.choices[0].message.content or ""
        self.last_decision = self._parse_decision(self.last_response_text)

        print(f"  Decision: {self.last_decision}")
        print(f"  Reasoning: {self.last_response_text[:80]}...")

        # Send 1 USDC and place bet
        try:
            tx_hash = self._send_usdc()
            await self._submit_bet(round_id, self.last_decision, tx_hash)
            self.round_bets[round_id] = self.last_decision
            print(f"  Bet placed: {self.last_decision} (tx: {tx_hash[:16]}...)")
        except Exception as e:
            print(f"  Bet failed: {e}")

    async def on_round_settled(self, data: dict) -> None:
        """Round result is in — update DopamineCore."""
        round_id = data["round_id"]
        result = data["result"]
        total_pool = float(data.get("total_pool", 0))

        my_bet = self.round_bets.pop(round_id, None)
        if my_bet is None:
            return  # didn't bet this round

        if result == "DRAW":
            pnl = 0.0
        elif my_bet == result:
            # Estimate payout (simplified)
            pnl = 0.65  # typical payout minus stake
        else:
            pnl = -1.0

        # UPDATE DOPAMINE ENGINE — this is the key line
        signal = self.engine.update(self.last_response_text, pnl)

        outcome = "WIN" if pnl > 0 else ("DRAW" if pnl == 0 else "LOSS")
        print(f"  Result: {result} | You bet: {my_bet} → {outcome} (${pnl:+.2f})")
        print(f"  RPE: {signal.phasic_response:+.3f} | Tonic: {signal.tonic_level:+.4f}")

    def on_payout(self, data: dict) -> None:
        amount = float(data.get("amount", 0))
        print(f"  Payout received: ${amount:.2f}")

    def _parse_decision(self, text: str) -> str:
        upper = text.upper()
        up_pos = upper.rfind("UP")
        down_pos = upper.rfind("DOWN")
        if up_pos > down_pos:
            return "UP"
        return "DOWN"

    def _send_usdc(self) -> str:
        """Send 1 USDC to platform wallet, wait for confirmation."""
        usdc = self.w3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS), abi=USDC_ABI
        )
        tx = usdc.functions.transfer(
            Web3.to_checksum_address(PLATFORM_WALLET),
            1_000_000,  # 1 USDC
        ).build_transaction({
            "from": self.account.address,
            "gas": 100_000,
            "maxFeePerGas": self.w3.eth.gas_price * 2,
            "maxPriorityFeePerGas": self.w3.to_wei(0.001, "gwei"),
            "nonce": self.w3.eth.get_transaction_count(self.account.address),
            "chainId": 8453,
        })
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        tx_hash_hex = receipt["transactionHash"].hex()

        # Wait for 1 block confirmation (required by server)
        tx_block = receipt["blockNumber"]
        while self.w3.eth.block_number - tx_block < 1:
            time.sleep(1)

        return tx_hash_hex

    async def _submit_bet(self, round_id: int, side: str, tx_hash: str) -> None:
        """POST bet to Moltrooms API."""
        async with self.session.post(
            f"{API_BASE}/bet",
            json={"round_id": round_id, "side": side, "tx_hash": tx_hash},
            headers={"Authorization": f"Bearer {self.jwt}"},
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Bet rejected ({resp.status}): {error}")


if __name__ == "__main__":
    bot = MoltroomsBot()
    asyncio.run(bot.run())
