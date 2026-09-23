#!/usr/bin/env python3
"""
FAST/RESUMABLE v2: recover a consistent per-edge token amount for an existing
Solana graph_edges.csv without re-running mint-address signature discovery.

Design goals
------------
1. Reuse the tx_signature values already present in graph_edges.csv.
2. Never call Birdeye. Amount recovery uses Solana RPC only.
3. Fetch each pending (tx_signature, token_mint) pair at most once per run.
4. Preserve exact integer raw token amounts with Python integers/Decimal; never use
   float arithmetic for the new normalized amount.
5. Save only a compact amount-resolution mapping, not transaction JSON.
6. Be resumable: a pair is written to the progress file only AFTER its CSV rows
   have been flushed to disk.
7. Resolve some legacy SPL-token rows locally ONLY when the raw amount is provably
   unique, avoiding getTransaction calls without sacrificing correctness.
8. Stream RPC work through independent workers: one slow/retrying transaction never
   blocks already-finished transactions. Heartbeats make network stalls visible.
9. Resume safely from the same CSV/progress files created by v1; old local shortcuts
   are audited once and any non-provable rows are removed and queued for RPC.

Output CSV
----------
One row per unique existing edge identity:
    tx_signature, token_mint, source_address, target_address, original_amount,
    amount_raw, decimals, amount_token, resolution_status, provenance

The mapping deliberately excludes experiment_id/q-window columns because the same
on-chain transfer is copied into multiple q windows. It can later be merged back to
all q-window rows using the first five columns as the key.

Progress file
-------------
One tab-separated line per completed (tx_signature, token_mint) pair:
    tx_signature<TAB>token_mint<TAB>pair_status

A completed pair is skipped on future runs. Temporary RPC failures are NOT marked
complete, so they are retried on the next run.

Recommended first commands
--------------------------
    python normalize_edge_amounts.py --self-test
    python normalize_edge_amounts.py --plan-only

Then run the full resolver. Public Solana RPC is rate-limited; defaults are kept
below the documented single-method public limit. With a dedicated RPC, pass that
URL and a rate/concurrency allowed by your provider.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import aiohttp
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: aiohttp. Install with: pip install aiohttp pandas tqdm"
    ) from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pandas. Install with: pip install aiohttp pandas tqdm"
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: tqdm. Install with: pip install aiohttp pandas tqdm"
    ) from exc


# Keep enough precision for very large SPL raw amounts and decimal conversion.
getcontext().prec = 80

DEFAULT_RPC_URL = os.environ.get(
    "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
)

# Legacy SPL Token program. Local no-RPC shortcuts are restricted to this program;
# Token-2022 is intentionally re-fetched because extensions can affect UI semantics.
LEGACY_TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

INPUT_COLUMNS = [
    "tx_signature",
    "token_mint",
    "source_address",
    "target_address",
    "amount",
]

OUTPUT_COLUMNS = [
    "tx_signature",
    "token_mint",
    "source_address",
    "target_address",
    "original_amount",
    "amount_raw",
    "decimals",
    "amount_token",
    "resolution_status",
    "provenance",
]

PUBLIC_RPC_HOST_MARKERS = (
    "api.mainnet-beta.solana.com",
    "api.mainnet.solana.com",
)


# ---------------------------------------------------------------------------
# Numeric/string helpers
# ---------------------------------------------------------------------------

def clean_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def safe_int(value: Any) -> Optional[int]:
    text = clean_string(value)
    if text == "":
        return None
    try:
        return int(text)
    except Exception:
        try:
            d = Decimal(text)
            if d == d.to_integral_value():
                return int(d)
        except Exception:
            pass
    return None


def to_decimal(value: Any) -> Optional[Decimal]:
    text = clean_string(value)
    if text == "":
        return None
    try:
        d = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    return d


def decimal_is_integral(value: Decimal) -> bool:
    return value == value.to_integral_value()


def decimal_to_plain(value: Optional[Decimal]) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def decimals_equal(a: Any, b: Any) -> bool:
    da = to_decimal(a)
    db = to_decimal(b)
    return da is not None and db is not None and da == db


def canonical_raw(raw: Any) -> Optional[int]:
    text = clean_string(raw)
    if text == "":
        return None
    try:
        # SPL raw amounts are integer strings. Avoid int(float(...)) because it can
        # silently lose precision for large token amounts.
        return int(text)
    except Exception:
        return None


def raw_to_token_amount(raw_amount: int, decimals: int) -> Decimal:
    return Decimal(raw_amount) / (Decimal(10) ** int(decimals))


# ---------------------------------------------------------------------------
# RPC client
# ---------------------------------------------------------------------------

class StartRateLimiter:
    """Global start-rate limiter with a shared cooldown after HTTP/RPC throttling."""

    def __init__(self, requests_per_second: float):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        self.interval = 1.0 / float(requests_per_second)
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            delay = self._next_time - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = asyncio.get_running_loop().time()
            self._next_time = max(now, self._next_time) + self.interval

    async def cooldown(self, seconds: float) -> None:
        """Pause ALL workers together after a rate-limit response."""
        seconds = max(0.0, float(seconds))
        if seconds <= 0:
            return
        async with self._lock:
            now = asyncio.get_running_loop().time()
            self._next_time = max(self._next_time, now + seconds)


@dataclass
class RpcResult:
    ok: bool
    result: Any = None
    error_kind: str = ""
    error_message: str = ""


class AsyncSolanaRPC:
    def __init__(
        self,
        rpc_url: str,
        rps: float,
        concurrency: int,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        self.rpc_url = rpc_url
        self.rps = rps
        self.concurrency = max(1, int(concurrency))
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, int(max_retries))
        self.rate_limiter = StartRateLimiter(rps)
        self.semaphore = asyncio.Semaphore(self.concurrency)
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.retry_count = 0
        self.rate_limit_count = 0

    async def __aenter__(self) -> "AsyncSolanaRPC":
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        connector = aiohttp.TCPConnector(
            limit=self.concurrency,
            ttl_dns_cache=300,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
        )
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"Content-Type": "application/json"},
            trust_env=True,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session is not None:
            await self.session.close()

    async def call(self, method: str, params: Sequence[Any]) -> RpcResult:
        if self.session is None:
            raise RuntimeError("RPC client session is not open")

        payload = {
            "jsonrpc": "2.0",
            "id": random.randint(1, 2_000_000_000),
            "method": method,
            "params": list(params),
        }

        last_error = ""

        for attempt in range(self.max_retries):
            await self.rate_limiter.wait()
            self.request_count += 1

            try:
                async with self.semaphore:
                    async with self.session.post(self.rpc_url, json=payload) as response:
                        if response.status == 429:
                            self.rate_limit_count += 1
                            self.retry_count += 1
                            retry_after = response.headers.get("Retry-After")
                            try:
                                wait_seconds = float(retry_after) if retry_after else min(8.0, 1.5 * (2 ** attempt))
                            except Exception:
                                wait_seconds = min(8.0, 1.5 * (2 ** attempt))
                            wait_seconds = max(0.75, wait_seconds) + random.uniform(0.0, 0.25)
                            # IMPORTANT: throttle every worker together, not only the worker
                            # that happened to receive this 429. This avoids a retry stampede.
                            await self.rate_limiter.cooldown(wait_seconds)
                            last_error = "HTTP 429"
                            continue

                        if response.status in {500, 502, 503, 504}:
                            self.retry_count += 1
                            body = await response.text()
                            last_error = f"HTTP {response.status}: {body[:200]}"
                            await asyncio.sleep(min(6.0, 0.75 * (2 ** attempt)) + random.uniform(0, 0.2))
                            continue

                        if response.status >= 400:
                            body = await response.text()
                            return RpcResult(
                                ok=False,
                                error_kind="permanent_http_error",
                                error_message=f"HTTP {response.status}: {body[:500]}",
                            )

                        try:
                            data = await response.json(content_type=None)
                        except Exception:
                            body = await response.text()
                            self.retry_count += 1
                            last_error = f"invalid_json: {body[:200]}"
                            await asyncio.sleep(min(6.0, 0.75 * (2 ** attempt)) + random.uniform(0, 0.2))
                            continue

                        if isinstance(data, dict) and "error" in data:
                            message = str(data.get("error"))
                            low = message.lower()
                            if any(term in low for term in ("rate", "too many", "limit", "temporar", "busy")):
                                self.rate_limit_count += 1
                                self.retry_count += 1
                                wait_seconds = min(8.0, 1.5 * (2 ** attempt)) + random.uniform(0.0, 0.25)
                                await self.rate_limiter.cooldown(wait_seconds)
                                last_error = message
                                continue
                            return RpcResult(
                                ok=False,
                                error_kind="rpc_error",
                                error_message=message,
                            )

                        if not isinstance(data, dict):
                            self.retry_count += 1
                            last_error = "unexpected_non_dict_rpc_response"
                            await asyncio.sleep(min(6.0, 0.75 * (2 ** attempt)) + random.uniform(0, 0.2))
                            continue

                        return RpcResult(ok=True, result=data.get("result"))

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                self.retry_count += 1
                last_error = f"{type(exc).__name__}: {exc}"
                # Keep retries short. With millions of historical transactions it is
                # faster and safer to defer a persistently bad request to the next run
                # than to let one worker wait for many minutes.
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(min(5.0, 0.5 * (2 ** attempt)) + random.uniform(0, 0.2))
            except Exception as exc:
                self.retry_count += 1
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(min(5.0, 0.5 * (2 ** attempt)) + random.uniform(0, 0.2))

        return RpcResult(
            ok=False,
            error_kind="temporary_rpc_failure",
            error_message=last_error or "RPC failed after retries",
        )

    async def get_transaction(self, signature: str) -> RpcResult:
        return await self.call(
            "getTransaction",
            [
                signature,
                {
                    "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": 0,
                    "commitment": "finalized",
                },
            ],
        )


# ---------------------------------------------------------------------------
# Mint metadata (cheap: getMultipleAccounts, max 100 mints/call)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MintMeta:
    decimals: Optional[int]
    owner_program: str


def parse_mint_account(account: Any) -> MintMeta:
    if not isinstance(account, dict):
        return MintMeta(decimals=None, owner_program="")

    owner_program = clean_string(account.get("owner"))
    decimals: Optional[int] = None
    data = account.get("data")

    if isinstance(data, dict):
        parsed = data.get("parsed")
        if isinstance(parsed, dict):
            info = parsed.get("info")
            if isinstance(info, dict):
                decimals = safe_int(info.get("decimals"))

    return MintMeta(decimals=decimals, owner_program=owner_program)


async def fetch_mint_metadata(
    rpc: AsyncSolanaRPC,
    mints: Sequence[str],
    batch_size: int = 100,
) -> Dict[str, MintMeta]:
    metadata: Dict[str, MintMeta] = {}
    mints = [m for m in mints if m]

    print(f"Fetching mint metadata for {len(mints):,} unique mints in batches of <= {batch_size} ...")

    for start in tqdm(range(0, len(mints), batch_size), desc="Mint metadata"):
        batch = list(mints[start : start + batch_size])
        rpc_result = await rpc.call(
            "getMultipleAccounts",
            [
                batch,
                {
                    "encoding": "jsonParsed",
                    "commitment": "finalized",
                },
            ],
        )

        if not rpc_result.ok:
            # This metadata is only an optimization. Missing metadata never causes
            # an amount guess; affected pairs fall back to getTransaction.
            print(
                f"WARNING: mint metadata batch failed; those mints will use transaction parsing. "
                f"error={rpc_result.error_kind}: {rpc_result.error_message}"
            )
            for mint in batch:
                metadata[mint] = MintMeta(decimals=None, owner_program="")
            continue

        result = rpc_result.result
        values = result.get("value") if isinstance(result, dict) else None
        if not isinstance(values, list) or len(values) != len(batch):
            for mint in batch:
                metadata[mint] = MintMeta(decimals=None, owner_program="")
            continue

        for mint, account in zip(batch, values):
            metadata[mint] = parse_mint_account(account)

    return metadata


# ---------------------------------------------------------------------------
# Transaction parsing, deliberately mirroring the original collector
# ---------------------------------------------------------------------------

def get_account_keys(tx: Dict[str, Any]) -> List[Optional[str]]:
    keys: List[Optional[str]] = []
    try:
        raw_keys = tx["transaction"]["message"]["accountKeys"]
    except Exception:
        return keys

    for item in raw_keys:
        if isinstance(item, str):
            keys.append(item)
        elif isinstance(item, dict):
            keys.append(item.get("pubkey"))
        else:
            keys.append(None)
    return keys


def build_token_account_maps(
    tx: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, str]]:
    account_keys = get_account_keys(tx)
    account_to_mint: Dict[str, str] = {}
    account_to_decimals: Dict[str, int] = {}
    account_to_owner: Dict[str, str] = {}
    meta = tx.get("meta") or {}

    balances = (meta.get("preTokenBalances") or []) + (meta.get("postTokenBalances") or [])

    for balance in balances:
        if not isinstance(balance, dict):
            continue
        account_index = safe_int(balance.get("accountIndex"))
        if account_index is None or account_index < 0 or account_index >= len(account_keys):
            continue
        token_account = account_keys[account_index]
        if not token_account:
            continue
        mint = clean_string(balance.get("mint"))
        owner = clean_string(balance.get("owner"))
        ui_token_amount = balance.get("uiTokenAmount") or {}
        decimals = safe_int(ui_token_amount.get("decimals")) if isinstance(ui_token_amount, dict) else None

        if mint:
            account_to_mint[token_account] = mint
        if owner:
            account_to_owner[token_account] = owner
        if decimals is not None:
            account_to_decimals[token_account] = decimals

    return account_to_mint, account_to_decimals, account_to_owner


def iter_instructions(tx: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (locator, instruction) without losing top/inner provenance."""
    try:
        top = tx["transaction"]["message"].get("instructions") or []
    except Exception:
        top = []

    for index, instruction in enumerate(top):
        if isinstance(instruction, dict):
            yield f"top:{index}", instruction

    for group in (tx.get("meta") or {}).get("innerInstructions") or []:
        if not isinstance(group, dict):
            continue
        parent_index = group.get("index")
        for inner_index, instruction in enumerate(group.get("instructions") or []):
            if isinstance(instruction, dict):
                yield f"inner:{parent_index}:{inner_index}", instruction


def legacy_amount_representation(
    info: Dict[str, Any],
    source: str,
    destination: str,
    account_to_decimals: Dict[str, int],
) -> Optional[str]:
    """Reproduce the amount string logic used by the original edge collector."""
    token_amount = info.get("tokenAmount")

    if isinstance(token_amount, dict):
        for key in ("uiAmountString", "uiAmount", "amount"):
            if token_amount.get(key) is not None:
                value = token_amount.get(key)
                if key == "amount" and token_amount.get("decimals") is not None:
                    try:
                        # Intentionally mirror the OLD collector's float division
                        # solely for matching the already-written CSV value.
                        return str(
                            int(value) / (10 ** int(token_amount["decimals"]))
                        )
                    except Exception:
                        return clean_string(value)
                return clean_string(value)

    raw_amount = info.get("amount")
    if raw_amount is None:
        return None

    decimals = account_to_decimals.get(source)
    if decimals is None:
        decimals = account_to_decimals.get(destination)

    if decimals is not None:
        try:
            # Again: this mirrors the old collector for matching only.
            return str(int(raw_amount) / (10 ** int(decimals)))
        except Exception:
            pass

    return clean_string(raw_amount)


@dataclass(frozen=True)
class TransferCandidate:
    source_address: str
    target_address: str
    raw_amount: Optional[int]
    decimals: Optional[int]
    amount_token: Optional[Decimal]
    legacy_amount: str
    provenance: str
    locator: str


def extract_transfer_candidates(
    tx: Dict[str, Any],
    requested_mint: str,
    mint_meta: Optional[MintMeta],
) -> List[TransferCandidate]:
    candidates: List[TransferCandidate] = []

    if not isinstance(tx, dict):
        return candidates
    meta = tx.get("meta") or {}
    if meta.get("err") is not None:
        return candidates

    account_to_mint, account_to_decimals, account_to_owner = build_token_account_maps(tx)

    for locator, instruction in iter_instructions(tx):
        parsed = instruction.get("parsed")
        if not isinstance(parsed, dict):
            continue
        if instruction.get("program") not in {"spl-token", "spl-token-2022"}:
            continue
        if parsed.get("type") not in {"transfer", "transferChecked"}:
            continue

        info = parsed.get("info") or {}
        if not isinstance(info, dict):
            continue

        source = clean_string(info.get("source"))
        destination = clean_string(info.get("destination"))
        if not source or not destination:
            continue

        mint = (
            clean_string(info.get("mint"))
            or account_to_mint.get(source, "")
            or account_to_mint.get(destination, "")
        )
        if mint != requested_mint:
            continue

        token_amount = info.get("tokenAmount")
        raw_amount: Optional[int] = None
        decimals: Optional[int] = None
        provenance_parts: List[str] = []

        if isinstance(token_amount, dict):
            raw_amount = canonical_raw(token_amount.get("amount"))
            decimals = safe_int(token_amount.get("decimals"))
            if raw_amount is not None:
                provenance_parts.append("tokenAmount.raw")
            if decimals is not None:
                provenance_parts.append("tokenAmount.decimals")

        if raw_amount is None:
            raw_amount = canonical_raw(info.get("amount"))
            if raw_amount is not None:
                provenance_parts.append("info.amount")

        if decimals is None:
            decimals = account_to_decimals.get(source)
            if decimals is None:
                decimals = account_to_decimals.get(destination)
            if decimals is not None:
                provenance_parts.append("tx_balance.decimals")

        if decimals is None and mint_meta is not None and mint_meta.decimals is not None:
            decimals = mint_meta.decimals
            provenance_parts.append("mint_account.decimals")

        amount_token: Optional[Decimal] = None
        if raw_amount is not None and decimals is not None and decimals >= 0:
            amount_token = raw_to_token_amount(raw_amount, decimals)

        old_amount = legacy_amount_representation(
            info=info,
            source=source,
            destination=destination,
            account_to_decimals=account_to_decimals,
        )

        candidates.append(
            TransferCandidate(
                source_address=account_to_owner.get(source) or source,
                target_address=account_to_owner.get(destination) or destination,
                raw_amount=raw_amount,
                decimals=decimals,
                amount_token=amount_token,
                legacy_amount=clean_string(old_amount),
                provenance="rpc:" + "+".join(provenance_parts or ["amount_unavailable"]),
                locator=locator,
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Edge resolution
# ---------------------------------------------------------------------------

def base_output_row(edge: Dict[str, str]) -> Dict[str, str]:
    return {
        "tx_signature": edge["tx_signature"],
        "token_mint": edge["token_mint"],
        "source_address": edge["source_address"],
        "target_address": edge["target_address"],
        "original_amount": edge["amount"],
        "amount_raw": "",
        "decimals": "",
        "amount_token": "",
        "resolution_status": "",
        "provenance": "",
    }


def _numeric_text_equal(a: str, b: str) -> bool:
    da = to_decimal(a)
    db = to_decimal(b)
    return da is not None and db is not None and da == db


def _old_float_scaled_repr(raw_amount: int, decimals: int) -> str:
    """Reproduce the old collector's int / 10**decimals -> str(float) path."""
    return str(int(raw_amount) / (10 ** int(decimals)))


def infer_provably_unique_raw_from_existing_amount(
    original_amount: str,
    decimals: int,
) -> Optional[int]:
    """
    Recover raw base units locally only when the answer is provably unique.

    Why this extra check exists:
    the old collector sometimes used Python float division before writing a scaled
    amount. For very large integers, two adjacent raw values can collapse to the
    same float/string. Merely multiplying the stored decimal by 10**decimals can
    therefore be wrong by one or more base units.

    For decimals>0 we only attempt this shortcut when the stored amount is
    non-integral, which proves it was NOT the raw-integer fallback. Then:
      * candidate_raw = stored_amount * 10**decimals must be an integer;
      * if the old float path could have produced the same stored numeric value,
        neither adjacent raw integer may produce that same value.
    This makes the local shortcut conservative; anything ambiguous goes to RPC.
    """
    amount = to_decimal(original_amount)
    if amount is None or amount < 0 or decimals < 0:
        return None

    scale = Decimal(10) ** int(decimals)

    if decimals == 0:
        if not decimal_is_integral(amount):
            return None
        return int(amount)

    # Raw SPL base units are integer strings. A non-integral stored value cannot
    # be the old raw fallback, so it must come from a scaled representation.
    if decimal_is_integral(amount):
        return None

    raw_decimal = amount * scale
    if not decimal_is_integral(raw_decimal):
        return None
    candidate = int(raw_decimal)
    if candidate < 0:
        return None

    # If candidate's float path does NOT equal the stored numeric value, the value
    # must have come from an exact scaled field such as uiAmountString; then the
    # exact decimal relation above uniquely determines the raw integer.
    candidate_float = _old_float_scaled_repr(candidate, decimals)
    if not _numeric_text_equal(candidate_float, original_amount):
        return candidate

    # Otherwise the historical float path is plausible. Verify that the float
    # representation is one-to-one at this integer. Because raw->float(raw/scale)
    # is monotone, checking immediate neighbors is sufficient to detect a local
    # collision plateau. If either neighbor maps to the same numeric text, RPC is
    # required; never guess which raw integer was original.
    for neighbor in (candidate - 1, candidate + 1):
        if neighbor < 0:
            continue
        if _numeric_text_equal(
            _old_float_scaled_repr(neighbor, decimals),
            original_amount,
        ):
            return None

    return candidate


def try_local_resolve_group(
    edges: List[Dict[str, str]],
    mint_meta: Optional[MintMeta],
) -> Optional[List[Dict[str, str]]]:
    """Resolve a legacy SPL-token group locally only when every raw value is provable."""
    if mint_meta is None:
        return None
    if mint_meta.owner_program != LEGACY_TOKEN_PROGRAM_ID:
        return None
    if mint_meta.decimals is None or mint_meta.decimals < 0:
        return None

    decimals = mint_meta.decimals
    inferred: List[Tuple[Dict[str, str], int, Decimal]] = []

    for edge in edges:
        original = clean_string(edge.get("amount"))
        raw = infer_provably_unique_raw_from_existing_amount(original, decimals)
        if raw is None:
            return None
        amount_token = raw_to_token_amount(raw, decimals)
        inferred.append((edge, raw, amount_token))

    rows: List[Dict[str, str]] = []
    for edge, raw, amount_token in inferred:
        row = base_output_row(edge)
        row.update(
            {
                "amount_raw": str(raw),
                "decimals": str(decimals),
                "amount_token": decimal_to_plain(amount_token),
                "resolution_status": (
                    "resolved_local_decimals0"
                    if decimals == 0
                    else "resolved_local_proven_unique"
                ),
                "provenance": "existing_amount+legacy_mint_account+unique_raw_proof",
            }
        )
        rows.append(row)

    return rows


def resolve_edge_against_candidates(
    edge: Dict[str, str],
    candidates: List[TransferCandidate],
) -> Dict[str, str]:
    row = base_output_row(edge)
    original_amount = clean_string(edge.get("amount"))

    endpoint_candidates = [
        c
        for c in candidates
        if c.source_address == edge["source_address"]
        and c.target_address == edge["target_address"]
    ]

    if not endpoint_candidates:
        row.update(
            {
                "resolution_status": "unresolved_no_endpoint_match",
                "provenance": "rpc_transaction_parsed",
            }
        )
        return row

    # First try the exact representation produced by the OLD collector.
    matches = [c for c in endpoint_candidates if clean_string(c.legacy_amount) == original_amount]

    # CSV/string formatting can occasionally differ while representing the exact
    # same decimal number. A numeric-equality fallback is still exact Decimal,
    # never an approximate float comparison.
    if not matches:
        matches = [c for c in endpoint_candidates if decimals_equal(c.legacy_amount, original_amount)]

    # Defensive fallback: compare against canonical raw/token forms. This can
    # recover rows if a historical CSV writer changed superficial formatting.
    if not matches:
        original_decimal = to_decimal(original_amount)
        if original_decimal is not None:
            candidate_matches: List[TransferCandidate] = []
            for c in endpoint_candidates:
                if c.amount_token is not None and c.amount_token == original_decimal:
                    candidate_matches.append(c)
                    continue
                if c.raw_amount is not None and Decimal(c.raw_amount) == original_decimal:
                    candidate_matches.append(c)
            matches = candidate_matches

    if not matches:
        row.update(
            {
                "resolution_status": "unresolved_no_amount_match",
                "provenance": "rpc_transaction_parsed",
            }
        )
        return row

    complete_matches = [
        c
        for c in matches
        if c.raw_amount is not None and c.decimals is not None and c.amount_token is not None
    ]

    if not complete_matches:
        row.update(
            {
                "resolution_status": "unresolved_missing_raw_or_decimals",
                "provenance": matches[0].provenance if matches else "rpc_transaction_parsed",
            }
        )
        return row

    semantic_values = {
        (c.raw_amount, c.decimals, decimal_to_plain(c.amount_token))
        for c in complete_matches
    }

    if len(semantic_values) != 1:
        row.update(
            {
                "resolution_status": "unresolved_ambiguous_multiple_amounts",
                "provenance": "rpc_multiple_non_equivalent_instructions",
            }
        )
        return row

    chosen = complete_matches[0]
    row.update(
        {
            "amount_raw": str(chosen.raw_amount),
            "decimals": str(chosen.decimals),
            "amount_token": decimal_to_plain(chosen.amount_token),
            "resolution_status": (
                "resolved_rpc_exact"
                if len(complete_matches) == 1
                else "resolved_rpc_multiple_equivalent"
            ),
            "provenance": chosen.provenance,
        }
    )
    return row


def resolve_group_from_transaction(
    edges: List[Dict[str, str]],
    tx: Dict[str, Any],
    requested_mint: str,
    mint_meta: Optional[MintMeta],
) -> List[Dict[str, str]]:
    candidates = extract_transfer_candidates(tx, requested_mint, mint_meta)
    return [resolve_edge_against_candidates(edge, candidates) for edge in edges]



# ---------------------------------------------------------------------------
# One-time v1 resume-file safety migration
# ---------------------------------------------------------------------------

OLD_LOCAL_STATUSES = {
    "resolved_local_legacy_noninteger",
    "resolved_local_legacy_decimals0",
}


def _old_local_row_is_safe(row: Dict[str, str]) -> bool:
    status = clean_string(row.get("resolution_status"))
    if status == "resolved_local_legacy_decimals0":
        original = to_decimal(row.get("original_amount"))
        raw = canonical_raw(row.get("amount_raw"))
        decimals = safe_int(row.get("decimals"))
        return (
            original is not None
            and decimals == 0
            and decimal_is_integral(original)
            and raw == int(original)
        )

    if status != "resolved_local_legacy_noninteger":
        return True

    decimals = safe_int(row.get("decimals"))
    raw = canonical_raw(row.get("amount_raw"))
    if decimals is None or raw is None:
        return False
    inferred = infer_provably_unique_raw_from_existing_amount(
        clean_string(row.get("original_amount")),
        decimals,
    )
    return inferred is not None and inferred == raw


def migrate_v1_resume_files_if_needed(
    output_path: Path,
    progress_path: Path,
) -> None:
    """
    v1 used a faster but not fully rigorous local inverse of old float-scaled amounts.
    Before resuming with v2, atomically rewrite only if old local statuses exist:
      * provably safe old rows are retained and relabeled with v2 statuses;
      * non-provable tx/mint pairs are removed from output AND progress, so v2 will
        refetch them through getTransaction.

    The rewrite uses temporary files and os.replace, so the original files survive
    intact unless the complete migration succeeds.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    tmp_output = output_path.with_suffix(output_path.suffix + ".v2tmp")
    old_status_rows = 0
    unsafe_pairs: set[str] = set()
    total_rows = 0

    # First pass: identify any unsafe old-local pair. Keep this pass streaming.
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != OUTPUT_COLUMNS:
            raise ValueError(
                f"Existing output schema does not match expected columns.\n"
                f"Existing: {reader.fieldnames}\nExpected: {OUTPUT_COLUMNS}"
            )
        for row in reader:
            total_rows += 1
            status = clean_string(row.get("resolution_status"))
            if status in OLD_LOCAL_STATUSES:
                old_status_rows += 1
                if not _old_local_row_is_safe(row):
                    key = f"{clean_string(row.get('tx_signature'))}|{clean_string(row.get('token_mint'))}"
                    unsafe_pairs.add(key)

    if old_status_rows == 0:
        return

    print(
        f"One-time v1 safety migration: found {old_status_rows:,} old local rows; "
        f"{len(unsafe_pairs):,} tx/mint pairs require RPC revalidation."
    )

    kept = 0
    removed = 0
    with output_path.open("r", encoding="utf-8", newline="") as src, \
         tmp_output.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in reader:
            key = f"{clean_string(row.get('tx_signature'))}|{clean_string(row.get('token_mint'))}"
            if key in unsafe_pairs:
                removed += 1
                continue

            status = clean_string(row.get("resolution_status"))
            if status == "resolved_local_legacy_noninteger":
                row["resolution_status"] = "resolved_local_proven_unique"
                row["provenance"] = "existing_amount+legacy_mint_account+unique_raw_proof"
            elif status == "resolved_local_legacy_decimals0":
                row["resolution_status"] = "resolved_local_decimals0"
                row["provenance"] = "existing_amount+legacy_mint_account+unique_raw_proof"
            writer.writerow(row)
            kept += 1
        dst.flush()
        os.fsync(dst.fileno())

    os.replace(tmp_output, output_path)

    if progress_path.exists():
        tmp_progress = progress_path.with_suffix(progress_path.suffix + ".v2tmp")
        kept_progress = 0
        removed_progress = 0
        with progress_path.open("r", encoding="utf-8", errors="replace") as src, \
             tmp_progress.open("w", encoding="utf-8", newline="") as dst:
            for line in src:
                stripped = line.rstrip("\r\n")
                if not stripped:
                    continue
                parts = stripped.split("\t")
                if len(parts) < 2:
                    continue
                key = f"{parts[0].strip()}|{parts[1].strip()}"
                if key in unsafe_pairs:
                    removed_progress += 1
                    continue
                dst.write(stripped + "\n")
                kept_progress += 1
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp_progress, progress_path)
        print(
            f"Migration complete: output kept={kept:,}, removed={removed:,}; "
            f"progress kept={kept_progress:,}, removed={removed_progress:,}."
        )
    else:
        print(f"Migration complete: output kept={kept:,}, removed={removed:,}.")

# ---------------------------------------------------------------------------
# Durable output/progress checkpointing
# ---------------------------------------------------------------------------

class CheckpointWriter:
    def __init__(
        self,
        output_path: Path,
        progress_path: Path,
        checkpoint_pairs: int,
        checkpoint_seconds: float,
    ) -> None:
        self.output_path = output_path
        self.progress_path = progress_path
        self.checkpoint_pairs = max(1, checkpoint_pairs)
        self.checkpoint_seconds = max(1.0, float(checkpoint_seconds))
        self.csv_buffer: List[Dict[str, str]] = []
        self.progress_buffer: List[Tuple[str, str, str]] = []
        self.pairs_since_flush = 0
        self.last_flush_time = time.monotonic()

        output_exists = output_path.exists() and output_path.stat().st_size > 0
        if output_exists:
            with output_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
            if header != OUTPUT_COLUMNS:
                raise ValueError(
                    f"Existing output schema does not match expected columns.\n"
                    f"Existing: {header}\nExpected: {OUTPUT_COLUMNS}"
                )

        self.output_file = output_path.open("a", encoding="utf-8", newline="")
        self.output_writer = csv.DictWriter(self.output_file, fieldnames=OUTPUT_COLUMNS)
        if not output_exists:
            self.output_writer.writeheader()
            self.output_file.flush()
            os.fsync(self.output_file.fileno())

        self.progress_file = progress_path.open("a", encoding="utf-8", newline="")

    def add_pair(
        self,
        signature: str,
        mint: str,
        pair_status: str,
        rows: List[Dict[str, str]],
    ) -> None:
        self.csv_buffer.extend(rows)
        self.progress_buffer.append((signature, mint, pair_status))
        self.pairs_since_flush += 1
        now = time.monotonic()
        if (
            self.pairs_since_flush >= self.checkpoint_pairs
            or now - self.last_flush_time >= self.checkpoint_seconds
        ):
            self.flush()

    def maybe_time_flush(self) -> None:
        if (
            self.progress_buffer
            and time.monotonic() - self.last_flush_time >= self.checkpoint_seconds
        ):
            self.flush()

    def flush(self) -> None:
        if not self.csv_buffer and not self.progress_buffer:
            self.last_flush_time = time.monotonic()
            return

        # Critical ordering: persist CSV rows first, then mark pairs complete.
        if self.csv_buffer:
            self.output_writer.writerows(self.csv_buffer)
            self.output_file.flush()
            os.fsync(self.output_file.fileno())

        if self.progress_buffer:
            for signature, mint, status in self.progress_buffer:
                self.progress_file.write(f"{signature}\t{mint}\t{status}\n")
            self.progress_file.flush()
            os.fsync(self.progress_file.fileno())

        self.csv_buffer.clear()
        self.progress_buffer.clear()
        self.pairs_since_flush = 0
        self.last_flush_time = time.monotonic()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.output_file.close()
            self.progress_file.close()


def load_processed_pairs(progress_path: Path) -> Dict[str, str]:
    processed: Dict[str, str] = {}
    if not progress_path.exists():
        return processed

    with progress_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                signature = parts[0].strip()
                mint = parts[1].strip()
                status = parts[2].strip() if len(parts) >= 3 else "done"
                if signature and mint:
                    processed[f"{signature}|{mint}"] = status
    return processed


# ---------------------------------------------------------------------------
# Input loading/planning
# ---------------------------------------------------------------------------

def load_compact_edge_mapping(input_path: Path) -> pd.DataFrame:
    print(f"Reading only required columns from: {input_path}")
    df = pd.read_csv(
        input_path,
        usecols=INPUT_COLUMNS,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    for col in INPUT_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    before = len(df)
    df = df[
        (df["tx_signature"] != "")
        & (df["token_mint"] != "")
        & (df["source_address"] != "")
        & (df["target_address"] != "")
        & (df["amount"] != "")
    ].copy()

    # q1/q2/q3 copies of the same on-chain transfer do not need separate amount
    # recovery. Preserve only the unique merge key used by the final mapping.
    df = df.drop_duplicates(subset=INPUT_COLUMNS, keep="first").reset_index(drop=True)

    print(f"Input edge rows:                 {before:,}")
    print(f"Unique amount-mapping edge keys: {len(df):,}")
    print(f"Unique tx signatures:            {df['tx_signature'].nunique():,}")
    print(f"Unique tx/mint pairs:             {df[['tx_signature','token_mint']].drop_duplicates().shape[0]:,}")
    print(f"Unique mints:                    {df['token_mint'].nunique():,}")
    return df


def dataframe_group_to_records(group: pd.DataFrame) -> List[Dict[str, str]]:
    return [
        {
            "tx_signature": clean_string(row.tx_signature),
            "token_mint": clean_string(row.token_mint),
            "source_address": clean_string(row.source_address),
            "target_address": clean_string(row.target_address),
            "amount": clean_string(row.amount),
        }
        for row in group.itertuples(index=False)
    ]


async def process_rpc_pair(
    item: Tuple[str, str, List[Dict[str, str]], Optional[MintMeta]],
    rpc: AsyncSolanaRPC,
) -> Tuple[str, str, str, Optional[List[Dict[str, str]]], bool]:
    """
    Process exactly one tx/mint pair.

    Return:
      signature, mint, pair_status, output_rows_or_None, should_mark_processed

    Crucially, each pair completes independently. A slow/retrying transaction no
    longer blocks results from other workers.
    """
    signature, mint, edges, mint_meta = item
    response = await rpc.get_transaction(signature)

    if not response.ok:
        return (
            signature,
            mint,
            f"deferred_{response.error_kind}",
            None,
            False,
        )

    if response.result is None:
        rows: List[Dict[str, str]] = []
        for edge in edges:
            row = base_output_row(edge)
            row.update(
                {
                    "resolution_status": "unresolved_transaction_unavailable",
                    "provenance": "rpc_getTransaction_null",
                }
            )
            rows.append(row)
        return signature, mint, "tx_unavailable", rows, True

    rows = resolve_group_from_transaction(
        edges=edges,
        tx=response.result,
        requested_mint=mint,
        mint_meta=mint_meta,
    )
    resolved_count = sum(r["resolution_status"].startswith("resolved_") for r in rows)
    if resolved_count == len(rows):
        status = "resolved_rpc"
    elif resolved_count > 0:
        status = "partial_rpc"
    else:
        status = "unresolved_rpc"
    return signature, mint, status, rows, True


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    mint = "Mint111111111111111111111111111111111111111"
    tx = {
        "meta": {
            "err": None,
            "preTokenBalances": [
                {
                    "accountIndex": 0,
                    "mint": mint,
                    "owner": "OwnerA",
                    "uiTokenAmount": {"decimals": 6},
                },
                {
                    "accountIndex": 1,
                    "mint": mint,
                    "owner": "OwnerB",
                    "uiTokenAmount": {"decimals": 6},
                },
            ],
            "postTokenBalances": [],
            "innerInstructions": [],
        },
        "transaction": {
            "message": {
                "accountKeys": [
                    {"pubkey": "TokenAccountA"},
                    {"pubkey": "TokenAccountB"},
                ],
                "instructions": [
                    {
                        "program": "spl-token",
                        "parsed": {
                            "type": "transfer",
                            "info": {
                                "source": "TokenAccountA",
                                "destination": "TokenAccountB",
                                "amount": "2500000",
                            },
                        },
                    },
                    {
                        "program": "spl-token",
                        "parsed": {
                            "type": "transferChecked",
                            "info": {
                                "source": "TokenAccountA",
                                "destination": "TokenAccountB",
                                "mint": mint,
                                "tokenAmount": {
                                    "amount": "5000000",
                                    "decimals": 6,
                                    "uiAmountString": "5",
                                    "uiAmount": 5.0,
                                },
                            },
                        },
                    },
                ],
            }
        },
    }

    meta = MintMeta(decimals=6, owner_program=LEGACY_TOKEN_PROGRAM_ID)
    candidates = extract_transfer_candidates(tx, mint, meta)
    assert len(candidates) == 2
    assert candidates[0].raw_amount == 2500000
    assert candidates[0].amount_token == Decimal("2.5")
    assert candidates[0].source_address == "OwnerA"
    assert candidates[0].target_address == "OwnerB"
    assert candidates[1].raw_amount == 5000000
    assert candidates[1].amount_token == Decimal("5")

    edge_1 = {
        "tx_signature": "sig1",
        "token_mint": mint,
        "source_address": "OwnerA",
        "target_address": "OwnerB",
        "amount": "2.5",
    }
    resolved_1 = resolve_edge_against_candidates(edge_1, candidates)
    assert resolved_1["amount_raw"] == "2500000"
    assert resolved_1["amount_token"] == "2.5"

    edge_2 = dict(edge_1)
    edge_2["amount"] = "5"
    resolved_2 = resolve_edge_against_candidates(edge_2, candidates)
    assert resolved_2["amount_raw"] == "5000000"
    assert resolved_2["amount_token"] == "5"

    local_noninteger = try_local_resolve_group([edge_1], meta)
    assert local_noninteger is not None
    assert local_noninteger[0]["amount_raw"] == "2500000"

    # Integer-looking amount with decimals>0 must NOT be guessed locally.
    assert try_local_resolve_group([edge_2], meta) is None

    # Regression test for the v1 float-rounding ambiguity. These adjacent raw
    # integers both stringify to the same old float-scaled value, so v2 MUST
    # refuse local inversion and send the pair to RPC.
    ambiguous_text = str(6036837554466187 / 10)
    assert ambiguous_text == str(6036837554466188 / 10)
    assert infer_provably_unique_raw_from_existing_amount(ambiguous_text, 1) is None

    # A normal small non-integer amount remains locally resolvable.
    assert infer_provably_unique_raw_from_existing_amount("6144.572708", 6) == 6144572708

    print("SELF-TEST PASSED")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    progress_path = Path(args.progress)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rpc_url = args.rpc_url
    rps = float(args.rps)

    is_public = any(marker in rpc_url for marker in PUBLIC_RPC_HOST_MARKERS)
    if is_public and rps > 3.8 and not args.unsafe_public_rate:
        print(
            f"WARNING: public Solana RPC detected. Capping --rps from {rps} to 3.8 "
            "to stay below the documented 40 requests/10s single-method limit."
        )
        rps = 3.8

    compact = load_compact_edge_mapping(input_path)

    # v2 correctness migration is intentionally skipped in --plan-only because
    # plan-only must not mutate any output/progress file.
    if not args.plan_only and not args.skip_v1_safety_migration:
        migrate_v1_resume_files_if_needed(output_path, progress_path)

    processed = load_processed_pairs(progress_path)
    print(f"Previously completed tx/mint pairs: {len(processed):,}")

    unique_mints = compact["token_mint"].drop_duplicates().tolist()

    async with AsyncSolanaRPC(
        rpc_url=rpc_url,
        rps=rps,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    ) as rpc:
        mint_metadata = await fetch_mint_metadata(rpc, unique_mints)

        # ---------------- PLAN ONLY ----------------
        if args.plan_only:
            local_pairs = 0
            rpc_pairs = 0
            skipped_pairs = 0
            local_edges = 0
            rpc_edges = 0

            grouped = compact.groupby(["tx_signature", "token_mint"], sort=False)
            for (signature, mint), group in tqdm(
                grouped,
                total=grouped.ngroups,
                desc="Planning",
            ):
                key = f"{signature}|{mint}"
                if key in processed:
                    skipped_pairs += 1
                    continue
                edges = dataframe_group_to_records(group)
                local_rows = try_local_resolve_group(edges, mint_metadata.get(mint))
                if local_rows is not None:
                    local_pairs += 1
                    local_edges += len(edges)
                else:
                    rpc_pairs += 1
                    rpc_edges += len(edges)

            print("\n" + "=" * 72)
            print("PLAN (v2 conservative local-proof rules)")
            print("=" * 72)
            print(f"Already completed pairs:                 {skipped_pairs:,}")
            print(f"Provably resolvable locally (no tx RPC): {local_pairs:,} pairs / {local_edges:,} edges")
            print(f"Need getTransaction:                     {rpc_pairs:,} pairs / {rpc_edges:,} edges")
            if rpc_pairs > 0:
                ideal_hours = rpc_pairs / rps / 3600.0
                print(f"Ideal request-start-time at {rps:.2f} req/s:       {ideal_hours:.2f} hours")
                if is_public:
                    print("Public RPC real time can be longer because of latency, retries, and rate limits.")
            print("No output/progress was changed in --plan-only mode.")
            return 0

        # ---------------- STREAMING FULL / BENCHMARK RUN ----------------
        writer = CheckpointWriter(
            output_path=output_path,
            progress_path=progress_path,
            checkpoint_pairs=args.checkpoint_pairs,
            checkpoint_seconds=args.checkpoint_seconds,
        )

        counters = {
            "skipped": 0,
            "local_pairs": 0,
            "rpc_pairs_queued": 0,
            "rpc_pairs_attempted": 0,
            "rpc_pairs_completed": 0,
            "rpc_pairs_deferred": 0,
            "resolved_edges": 0,
            "unresolved_edges": 0,
        }
        state = {
            "active_rpc": 0,
            "last_completion": time.monotonic(),
            "producer_done": False,
            "stop_requested": False,
        }
        start_time = time.time()

        queue_size = args.queue_size if args.queue_size > 0 else max(64, args.concurrency * 4)
        rpc_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        heartbeat_stop = asyncio.Event()

        grouped = compact.groupby(["tx_signature", "token_mint"], sort=False)
        progress_bar = tqdm(total=grouped.ngroups, desc="Amount recovery", dynamic_ncols=True)

        def update_postfix() -> None:
            progress_bar.set_postfix(
                local=counters["local_pairs"],
                rpc=counters["rpc_pairs_completed"],
                defer=counters["rpc_pairs_deferred"],
                q=rpc_queue.qsize(),
                active=state["active_rpc"],
                refresh=False,
            )

        async def heartbeat() -> None:
            while not heartbeat_stop.is_set():
                try:
                    await asyncio.wait_for(
                        heartbeat_stop.wait(),
                        timeout=max(2.0, float(args.heartbeat_seconds)),
                    )
                    break
                except asyncio.TimeoutError:
                    pass

                writer.maybe_time_flush()
                idle_for = time.monotonic() - state["last_completion"]
                elapsed = max(0.001, time.time() - start_time)
                attempted = counters["rpc_pairs_attempted"]
                rate = attempted / elapsed
                tqdm.write(
                    "[heartbeat] "
                    f"local={counters['local_pairs']:,} "
                    f"rpc_done={counters['rpc_pairs_completed']:,} "
                    f"deferred={counters['rpc_pairs_deferred']:,} "
                    f"queue={rpc_queue.qsize():,} "
                    f"active={state['active_rpc']:,} "
                    f"HTTP={rpc.request_count:,} retries={rpc.retry_count:,} "
                    f"429={rpc.rate_limit_count:,} "
                    f"attempt_rate={rate:.2f}/s "
                    f"last_completion={idle_for:.1f}s ago"
                )

        async def rpc_worker(worker_id: int) -> None:
            while True:
                item = await rpc_queue.get()
                if item is None:
                    rpc_queue.task_done()
                    return

                signature, mint, edges, mint_meta = item
                state["active_rpc"] += 1
                counters["rpc_pairs_attempted"] += 1
                try:
                    result = await process_rpc_pair(item, rpc)
                    (
                        result_signature,
                        result_mint,
                        pair_status,
                        rows,
                        should_mark_processed,
                    ) = result

                    if should_mark_processed and rows is not None:
                        writer.add_pair(
                            result_signature,
                            result_mint,
                            pair_status,
                            rows,
                        )
                        processed[f"{result_signature}|{result_mint}"] = pair_status
                        counters["rpc_pairs_completed"] += 1
                        for row in rows:
                            if row["resolution_status"].startswith("resolved_"):
                                counters["resolved_edges"] += 1
                            else:
                                counters["unresolved_edges"] += 1
                    else:
                        counters["rpc_pairs_deferred"] += 1
                        if args.verbose_deferred:
                            tqdm.write(
                                f"DEFERRED: signature={signature} mint={mint} status={pair_status}"
                            )

                except Exception as exc:
                    # Never let one malformed transaction kill a multi-million-pair run.
                    # It remains unprocessed and will be retried next run.
                    counters["rpc_pairs_deferred"] += 1
                    tqdm.write(
                        f"WORKER-{worker_id} deferred unexpected error: "
                        f"signature={signature} mint={mint} error={type(exc).__name__}: {exc}"
                    )
                finally:
                    state["active_rpc"] -= 1
                    state["last_completion"] = time.monotonic()
                    progress_bar.update(1)
                    update_postfix()
                    rpc_queue.task_done()

        workers = [
            asyncio.create_task(rpc_worker(i), name=f"rpc-worker-{i}")
            for i in range(args.concurrency)
        ]
        heartbeat_task = asyncio.create_task(heartbeat(), name="heartbeat")

        # Graceful Ctrl+C/SIGTERM on Unix: stop producing new work, finish queued
        # requests, checkpoint, and exit. On platforms without signal-handler support,
        # asyncio.run's normal cancellation still reaches the finally block below.
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()
        installed_signals = []
        try:
            import signal

            def request_stop() -> None:
                if not state["stop_requested"]:
                    state["stop_requested"] = True
                    tqdm.write(
                        "Stop requested. No new pairs will be queued; finishing the bounded queue and checkpointing..."
                    )
                elif main_task is not None and not main_task.cancelled():
                    tqdm.write("Second stop signal received: cancelling immediately after checkpoint cleanup...")
                    main_task.cancel()

            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, request_stop)
                    installed_signals.append(sig)
                except (NotImplementedError, RuntimeError):
                    pass
        except Exception:
            installed_signals = []

        try:
            yielded = 0
            for (signature, mint), group in grouped:
                if state["stop_requested"]:
                    break

                key = f"{signature}|{mint}"
                if key in processed:
                    counters["skipped"] += 1
                    progress_bar.update(1)
                    yielded += 1
                    if yielded % 2000 == 0:
                        await asyncio.sleep(0)
                    continue

                edges = dataframe_group_to_records(group)
                local_rows = try_local_resolve_group(edges, mint_metadata.get(mint))

                if local_rows is not None:
                    pair_status = "resolved_local_v2"
                    writer.add_pair(signature, mint, pair_status, local_rows)
                    processed[key] = pair_status
                    counters["local_pairs"] += 1
                    counters["resolved_edges"] += len(local_rows)
                    state["last_completion"] = time.monotonic()
                    progress_bar.update(1)
                    yielded += 1
                    if yielded % 2000 == 0:
                        update_postfix()
                        await asyncio.sleep(0)
                    continue

                if (
                    args.max_rpc_pairs > 0
                    and counters["rpc_pairs_queued"] >= args.max_rpc_pairs
                ):
                    state["stop_requested"] = True
                    break

                # This await is the bounded backpressure point. It prevents millions
                # of asyncio tasks from being created and keeps memory stable.
                await rpc_queue.put((signature, mint, edges, mint_metadata.get(mint)))
                counters["rpc_pairs_queued"] += 1
                yielded += 1
                if yielded % 1000 == 0:
                    update_postfix()
                    await asyncio.sleep(0)

            state["producer_done"] = True

            # Finish only the small bounded queue already accepted. One slow worker
            # cannot block other workers from persisting their completed results.
            await rpc_queue.join()

        except asyncio.CancelledError:
            state["stop_requested"] = True
            tqdm.write("Cancellation received. Checkpointing completed work...")
            raise
        finally:
            # Tell workers to exit after all already-dequeued work completes/cancels.
            for _ in workers:
                try:
                    rpc_queue.put_nowait(None)
                except asyncio.QueueFull:
                    # Queue should normally be empty after join; fallback to await.
                    await rpc_queue.put(None)

            await asyncio.gather(*workers, return_exceptions=True)
            heartbeat_stop.set()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

            writer.close()
            progress_bar.close()

            for sig in installed_signals:
                try:
                    loop.remove_signal_handler(sig)
                except Exception:
                    pass

        elapsed = max(0.001, time.time() - start_time)
        attempted = counters["rpc_pairs_attempted"]
        rpc_rate_observed = attempted / elapsed if attempted else 0.0

        print("\n" + "=" * 72)
        print("RUN SUMMARY — FAST STREAMING v2")
        print("=" * 72)
        for key, value in counters.items():
            print(f"{key:24s}: {value:,}")
        print(f"RPC HTTP requests total : {rpc.request_count:,}")
        print(f"RPC retries             : {rpc.retry_count:,}")
        print(f"RPC rate-limit events   : {rpc.rate_limit_count:,}")
        print(f"Elapsed                 : {elapsed/3600:.2f} hours")
        if attempted:
            print(f"Observed RPC pair rate  : {rpc_rate_observed:.3f} pairs/sec (whole-run denominator)")
        print(f"Output CSV              : {output_path}")
        print(f"Progress file           : {progress_path}")

        if state["stop_requested"] and args.max_rpc_pairs > 0:
            print(
                f"Stopped after --max-rpc-pairs={args.max_rpc_pairs:,}. "
                "Run again without the cap; completed pairs will be skipped."
            )
        elif state["stop_requested"]:
            print("Stopped gracefully. Run the same command again to continue from the checkpoint.")
        elif counters["rpc_pairs_deferred"] > 0:
            print(
                "Some RPC pairs were deferred and were NOT added to progress. "
                "Run the script again after this pass; only those pending pairs will be retried."
            )
        else:
            print("All encountered pending pairs were checkpointed.")

        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fast streaming recovery of consistent raw/decimal/token amounts for "
            "existing Solana graph edges. Reuses v1 output/progress files."
        )
    )
    parser.add_argument("--input", default="graph_edges.csv")
    parser.add_argument("--output", default="edge_amount_normalization.csv")
    parser.add_argument("--progress", default="processed_amount_pairs.txt")
    parser.add_argument("--rpc-url", default=DEFAULT_RPC_URL)
    parser.add_argument(
        "--rps",
        type=float,
        default=3.7,
        help="Global getTransaction request start rate. Public Solana is capped to <=3.8/s unless explicitly overridden.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=24,
        help="Independent streaming RPC workers. Rate is still controlled by --rps.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=0,
        help="Bounded pending RPC queue. 0 => max(64, concurrency*4).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Per HTTP attempt timeout in seconds. Short by design so one bad tx cannot look hung for minutes.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Per-pair attempt count for this run. Exhausted temporary failures are deferred to the next run.",
    )
    parser.add_argument(
        "--checkpoint-pairs",
        type=int,
        default=5000,
        help="Checkpoint after this many completed pairs, whichever comes first with --checkpoint-seconds.",
    )
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=30.0,
        help="Checkpoint completed CSV rows then progress at least this often while work is arriving.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=15.0,
        help="Print liveness/RPC counters this often even when some requests are slow.",
    )
    parser.add_argument(
        "--max-rpc-pairs",
        type=int,
        default=0,
        help="Benchmark cap. 0 means unlimited. Completed work remains resumable.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--verbose-deferred",
        action="store_true",
        help="Print every deferred RPC pair. Off by default to keep large runs fast and readable.",
    )
    parser.add_argument(
        "--unsafe-public-rate",
        action="store_true",
        help="Allow >3.8 rps on a detected public Solana endpoint. Not recommended.",
    )
    parser.add_argument(
        "--skip-v1-safety-migration",
        action="store_true",
        help="Skip the one-time audit of local rows created by v1. Not recommended when resuming a v1 run.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    if args.rps <= 0:
        parser.error("--rps must be > 0")
    if args.concurrency <= 0:
        parser.error("--concurrency must be > 0")
    if args.checkpoint_pairs <= 0:
        parser.error("--checkpoint-pairs must be > 0")
    if args.checkpoint_seconds <= 0:
        parser.error("--checkpoint-seconds must be > 0")
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be > 0")
    if args.queue_size < 0:
        parser.error("--queue-size must be >= 0")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.max_retries <= 0:
        parser.error("--max-retries must be > 0")
    if args.max_rpc_pairs < 0:
        parser.error("--max-rpc-pairs must be >= 0")

    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
