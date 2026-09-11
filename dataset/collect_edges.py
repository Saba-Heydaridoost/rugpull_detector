"""
This script reads valid token-experiment rows directly from the rule-based
label dataset and builds transfer edges inside the exact same
observation windows.

Each input row represents one independent edge-collection job identified by lp_address + token_mint + experiment_id

For every token, the collector:
    1. validates all remaining q1, q2, and q3 observation windows;
    2. retrieves successful Solana signatures once up to the largest cutoff;
    3. fetches and parses each historical transaction only once;
    4. extracts SPL Token and Token-2022 transfer instructions;
    5. distributes every transfer to the q1/q2/q3 windows containing it;
    6. writes independent edge rows for every token-experiment graph.

The edge interval is half-open: [observation_start, observation_end)
Therefore, no transaction at or after the experiment cutoff is included.

Important coverage limitation:
Signature discovery is based on getSignaturesForAddress(token_mint). A token
transfer may be missed when the mint address is not directly included in the
transaction account list. The output therefore represents transfers visible
through the mint-address signature history, not a guaranteed complete index
of every historical transfer of the token.
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGS
# ============================================================

# In test mode, process only a limited number of in-window signatures for each independent token-experiment pair.
TEST_MODE = 0
LIMIT_SIGNATURES_PER_TOKEN  = 10

# Experiment identifiers must match the label and node datasets.
VALID_EXPERIMENT_IDS = {
    "q1",
    "q2",
    "q3"
}

# RPC pacing and retry settings reduce pressure on the public Solana endpoint. Rate limits and temporary transport failures use exponential backoff before the current edge job is deferred for a later run.
SLEEP_BETWEEN_RPC_CALLS = 0.25
MAX_RETRIES = 5

INPUT_FILE = "rule_based_labels.csv"

if TEST_MODE == 1:
    OUTPUT_FILE = "graph_edges_test.csv"
    PROGRESS_FILE = "processed_edges_test.txt"
else:
    OUTPUT_FILE = "graph_edges.csv"
    PROGRESS_FILE = "processed_edges.txt"

RPC_URL = "https://api.mainnet-beta.solana.com"

# The first five columns identify the source token-experiment node and its temporal boundaries. The remaining columns describe one historical token transfer edge.
OUTPUT_COLUMNS = [
    "lp_address",
    "token_mint",
    "experiment_id",
    "observation_start",
    "observation_end",

    "tx_signature",
    "timestamp",
    "source_address",
    "target_address",
    "amount"
]

# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_string(value):
    """
    Convert nullable CSV or RPC values into consistently stripped strings.
    """
        
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
        
    except Exception:
        pass

    return str(value).strip()


def safe_int(value):
    """
    Safely normalize textual, floating-point, or nullable numeric values into integers without terminating the complete collection run.
    """
        
    try:
        if value is None or pd.isna(value):
            return None

        return int(float(value))
    
    except Exception:
        return None


# ============================================================
# FILE / PROGRESS HELPERS
# ============================================================

def make_edge_job_key(lp_address, token_mint, experiment_id):
    """
    Build the unique resume key for one edge-collection job.
    Progress cannot be tracked by mint alone because the same token may have independent q1, q2, and q3 observation windows. Including lp_address, token_mint, and experiment_id keeps the key aligned with the node dataset.
    """

    return f"{lp_address}|{token_mint}|{experiment_id}"


def load_observation_windows(input_file):
    """
    Load and validate one edge-collection job for every valid label row.
    Temporal boundaries are derived directly from rule_based_labels.csv so graph edges use exactly the same creation-to-cutoff interval.
    Only rows with:
        - a supported q1/q2/q3 experiment identifier,
        - label_status equal to VALID,
        - a binary rug_pull_label,
        - complete and internally consistent temporal metadata,
        - label availability strictly after the observation cutoff
    are accepted.
    Duplicate lp_address-token_mint-experiment combinations are rejected before any RPC request is made.
    """

    try:
        df = pd.read_csv(
            input_file,
            dtype=str,
            keep_default_na=False
        )

    except Exception as e:
        raise RuntimeError(f"Could not read rule-based label file {input_file}: {e}")

    # Fail immediately when the rule-based label file does not contain the complete schema required to reconstruct the exact edge observation windows.
    required_columns = [
        "lp_address",
        "token_mint",
        "experiment_id",
        "rug_pull_label",
        "label_status",
        "token_creation_time",
        "observation_window_seconds",
        "observation_end_time",
        "label_available_time"
    ]

    missing_columns = [ column for column in required_columns if column not in df.columns ]

    if missing_columns:
        raise ValueError(f"Rule-based label file {input_file} is missing columns: {missing_columns}")

    windows = []
    seen_job_keys = set()

    # Validate every token-experiment row independently because the same mint may be valid under several observation cutoffs.
    for row_number, row in df.iterrows():
        lp_address = clean_string(row.get("lp_address"))
        token_mint = clean_string(row.get("token_mint"))
        experiment_id = clean_string(row.get("experiment_id"))
        label_status = clean_string(row.get("label_status"))

        rug_pull_label = safe_int(row.get("rug_pull_label"))

        creation_time = safe_int(row.get("token_creation_time"))
        observation_window_seconds = safe_int(row.get("observation_window_seconds"))
        observation_start = creation_time
        observation_end = safe_int(row.get("observation_end_time"))
        label_available_time = safe_int(row.get("label_available_time"))

        if lp_address == "":
            raise ValueError(f"Rule-based label row {row_number + 2} has no lp_address.")

        if token_mint == "":
            raise ValueError(f"Rule-based label row {row_number + 2} has no token_mint.")

        if experiment_id not in VALID_EXPERIMENT_IDS:
            raise ValueError(f"Unexpected experiment_id={experiment_id} for token={token_mint}. Expected one of {sorted(VALID_EXPERIMENT_IDS)}.")

        # Only VALID binary labels are eligible for node and edge generation. Excluded and unknown label rows remain in rule_based_labels.csv for auditing but must not enter the graph dataset.
        if label_status != "VALID":
            continue

        if rug_pull_label not in [0, 1]:
            raise ValueError(f"Invalid binary label for token={token_mint}, experiment={experiment_id}.")

        if None in [creation_time, observation_window_seconds, observation_start, observation_end, label_available_time]:
            raise ValueError(f"Incomplete temporal metadata for token={token_mint}, experiment={experiment_id}.")

        if observation_window_seconds <= 0:
            raise ValueError(f"Invalid observation duration for token={token_mint}, experiment={experiment_id}.")

        # Every experiment is anchored at token creation. Using another start time would make edge extraction inconsistent with node-feature generation.
        if observation_start != creation_time:
            raise ValueError(f"Observation start does not match creation time for token={token_mint}, experiment={experiment_id}.")

        # Reconstruct the cutoff independently instead of trusting the stored observation_end value without validation.
        expected_observation_end = observation_start + observation_window_seconds

        if observation_end != expected_observation_end:
            raise ValueError(f"Observation-window mismatch for token={token_mint}, experiment={experiment_id}. Expected {expected_observation_end}, found {observation_end}.")

        # The target label must not be observable within the edge window; otherwise the graph structure could contain post-outcome information.
        if label_available_time <= observation_end:
            raise ValueError(f"Label is not available strictly after the observation window for token={token_mint}, experiment={experiment_id}.")

        # The pool-mint-experiment combination defines one unique graph sample.
        job_key = make_edge_job_key(
            lp_address=lp_address,
            token_mint=token_mint,
            experiment_id=experiment_id
        )

        if job_key in seen_job_keys:
            raise ValueError(f"Duplicate token-experiment label in {input_file}: {job_key}")

        seen_job_keys.add(job_key)

        # Store only the metadata required by downstream RPC collection. The binary label remains in rule_based_labels.csv and will later be associated with the final graph through the corresponding node row.
        windows.append(
            {
                "lp_address": lp_address,
                "token_mint": token_mint,
                "experiment_id": experiment_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
                "observation_window_seconds": (
                    observation_window_seconds
                )
            }
        )

    return windows


def group_windows_by_token(windows):
    """
    Group all remaining q1, q2, and q3 windows of the same token.
    Signature history and transaction bodies can then be retrieved only once up to the largest remaining observation cutoff of that token.
    """

    groups = {}

    for window in windows:
        token_mint = window["token_mint"]
        observation_start = window["observation_start"]

        if token_mint not in groups:
            groups[token_mint] = {
                "token_mint": token_mint,
                "observation_start": observation_start,
                "windows": []
            }

        group = groups[token_mint]

        # All experiments of one token must be anchored at the same creation time. Otherwise they cannot safely share one RPC history scan.
        if group["observation_start"] != observation_start:
            raise ValueError(f"Inconsistent observation start for token={token_mint}. Expected {group['observation_start']}, found {observation_start}.")

        group["windows"].append(window)

    grouped_tokens = []

    for group in groups.values():
        # Sorting is not required for correctness, but makes logging and distribution across q1/q2/q3 deterministic.
        group["windows"].sort(key=lambda item: item["observation_end"])

        group["max_observation_end"] = max(item["observation_end"] for item in group["windows"])

        grouped_tokens.append(group)

    return grouped_tokens


def load_processed_edge_jobs():
    """
    Load token-experiment jobs completed in previous executions.
    The progress file is required even for jobs that legitimately produce zero edge rows, because such jobs leave no row in graph_edges.csv.
    The current implementation treats PROGRESS_FILE as the resume source. Therefore, when restarting from scratch, both the edge CSV and its matching progress file must be removed together.
    """

    if not os.path.exists(PROGRESS_FILE):
        return set()

    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
            return { line.strip() for line in file if line.strip() }

    except Exception as e:
        print(f"Could not read progress file: {e}")
        return set()


def mark_edge_job_as_processed(lp_address, token_mint, experiment_id):
    """
    Persist completion of one token-experiment job.
    This function is called only after all extracted edge rows have been successfully appended. A zero-edge result is also marked complete when the RPC history was collected successfully and no matching transfer existed.
    """

    job_key = make_edge_job_key(
        lp_address=lp_address,
        token_mint=token_mint,
        experiment_id=experiment_id
    )

    try:
        with open(PROGRESS_FILE, "a", encoding="utf-8") as file:
            file.write(job_key + "\n")
            file.flush()
            os.fsync(file.fileno())

        return True

    except Exception as e:
        print(f" Could not update progress file for edge job {job_key}: {e}")
        return False


def ensure_output_file_exists():
    try:
        if not os.path.exists(OUTPUT_FILE):
            pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_FILE, index=False)
            return

        existing_columns = list(pd.read_csv(OUTPUT_FILE, nrows=0).columns)

        if existing_columns != OUTPUT_COLUMNS:
            raise RuntimeError(f"Output file {OUTPUT_FILE} has an incompatible header.\nExpected: {OUTPUT_COLUMNS}\nFound: {existing_columns}")
        
    except RuntimeError:
        raise

    except Exception as e:
        raise RuntimeError(f"Could not prepare output file: {e}")


def append_rows_to_output(rows):
    # A successfully scanned job with no matching transfers is not an error. Its completion is represented only in the progress file.
    if not rows:
        return True

    try:
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
            OUTPUT_FILE,
            mode="a",
            header=False,
            index=False
        )
        return True
    
    except Exception as e:
        print(f" Could not append rows to output file: {e}")
        return False

# ============================================================
# RPC HELPERS
# ============================================================

def check_rpc_health():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getHealth"
    }

    try:
        response = requests.post(
            RPC_URL, 
            json=payload, 
            timeout=15
        )

        response.raise_for_status()

        data = response.json()
        return data.get("result") == "ok"
    
    except Exception as e:
        print(f"RPC health check failed: {e}")
        return False


def rpc_call(method, params):
    """
    Execute one Solana JSON-RPC request with exponential retry handling.
    HTTP 429 responses, RPC rate-limit errors, and temporary transport failures are retried. The returned error flag distinguishes a failed request from a legitimate empty RPC result.
    """
        
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                RPC_URL, 
                json=payload, 
                timeout=20
            )

            # Exponential backoff avoids immediately repeating requests while the public endpoint is enforcing a rate limit.
            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()
            data = response.json()

            # RPC-level errors may be returned inside a successful HTTP response and therefore require separate inspection.
            if "error" in data:
                error_message = str(data["error"])

                if "rate" in error_message.lower():
                    time.sleep(2 ** attempt)
                    continue

                print(f" RPC error in {method}: {error_message}")
                return None, True

            return data.get("result"), False
        
        except Exception:
            time.sleep(2 ** attempt)

    print(f" RPC failed for method: {method}")
    return None, True


def get_signatures_for_window(address, observation_start, observation_end):
    """
    Retrieve successful signatures visible through the mint address inside the half-open interval [observation_start, observation_end).
    Solana returns signatures newest-first. Pagination continues backward until a signature older than observation_start is reached or the full available history is exhausted.
    Signatures at or after observation_end and failed transactions are excluded. In TEST_MODE, collection stops after a fixed number of selected in-window signatures.
    This method finds only transactions associated with the mint address. It may not include every SPL Token transfer of the mint.
    """
        
    selected = []
    before = None
    page_count = 0

    while True:
        # Request the largest supported RPC page to reduce the number of historical pagination calls.
        options = {
            "limit": 1000,
            "commitment": "finalized"
        }

        if before:
            options["before"] = before

        result, had_error = rpc_call(
            "getSignaturesForAddress",
            [address, options]
        )

        if had_error:
            return selected, True, "signature_history_error"

        if not result:
            return selected, False, "ok"

        reached_window_start = False

        # Each page is newest-first, so records are scanned backward toward the observation-window start.
        for item in result:
            block_time = safe_int(item.get("blockTime"))

            # A single signature without blockTime is skipped without discarding the remaining signature history.
            if block_time is None:
                signature = clean_string(item.get("signature"))

                print(f" Signature has no blockTime and was skipped: {signature or 'unknown'}")

                continue

            # Once a signature older than observation_start is reached, all subsequent signatures in this newest-first traversal are also outside the required interval.
            if block_time < observation_start:
                reached_window_start = True
                break

            # Enforce the half-open interval: [observation_start, observation_end)
            if block_time >= observation_end or item.get("err") is not None:
                continue

            signature = clean_string(item.get("signature"))

            if signature:
                selected.append(signature)

            # In test mode, limit signatures once for the complete token.
            if TEST_MODE == 1 and len(selected) >= LIMIT_SIGNATURES_PER_TOKEN:
                return selected, False, "ok"

        page_count += 1

        if page_count % 10 == 0:
            print(f" RPC signature pagination: mint={address}, pages={page_count}, in_window={len(selected)}")

        # Stop when the requested temporal region has been fully crossed or the RPC returns a short final page.
        if reached_window_start or len(result) < options["limit"]:
            return selected, False, "ok"

        # Continue backward using the oldest signature of the current page as the pagination cursor.
        before = clean_string(result[-1].get("signature"))

        if not before:
            return selected, False, "signature_pagination_missing_cursor"

        time.sleep(SLEEP_BETWEEN_RPC_CALLS)


def get_transaction(signature):
    """
    Retrieve a finalized transaction in jsonParsed format.
    Parsed encoding exposes SPL Token instructions, token-balance metadata, account owners, mint identifiers, and decimal precision required for edge construction.
    """
        
    return rpc_call(
        "getTransaction",
        [
            signature,
            {
                "encoding": "jsonParsed",
                "maxSupportedTransactionVersion": 0,
                "commitment": "finalized"
            }
        ]
    )

# ============================================================
# TRANSACTION PARSING HELPERS
# ============================================================

def get_account_keys(tx):
    """
    Normalize transaction account keys into an index-aligned public-key list.
    Preserving account-index order is necessary because preTokenBalances and postTokenBalances refer to accounts by numerical accountIndex.
    """
        
    keys = []

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


def build_token_account_maps(tx):
    """
    Reconstruct token-account metadata from pre- and post-token balances.
    The returned mappings associate each token account with its mint, token decimal precision, and its owner wallet when owner metadata is available.
    Both pre- and post-balances are inspected because newly created or closed token accounts may exist on only one side of the transaction.
    """
        
    account_keys = get_account_keys(tx)
    account_to_mint = {}
    account_to_decimals = {}
    account_to_owner = {}
    meta = tx.get("meta") or {}

    # Combining pre- and post-balances improves coverage for accounts created, initialized, emptied, or closed inside the transaction.
    balances = (meta.get("preTokenBalances") or []) + (meta.get("postTokenBalances") or [])

    for balance in balances:
        account_index = balance.get("accountIndex")

        if account_index is None or account_index >= len(account_keys):
            continue

        # accountIndex links token-balance metadata back to the transaction account key at the same position.
        token_account = account_keys[account_index]

        if not token_account:
            continue

        mint = balance.get("mint")
        owner = balance.get("owner")
        decimals = (balance.get("uiTokenAmount") or {}).get("decimals")

        if mint:
            account_to_mint[token_account] = mint

        if owner:
            account_to_owner[token_account] = owner

        if decimals is not None:
            account_to_decimals[token_account] = decimals

    return account_to_mint, account_to_decimals, account_to_owner


def flatten_instructions(tx):
    """
    Combine top-level and inner transaction instructions.
    Direct wallet transfers may appear in the main message, while swaps and program-mediated transfers commonly invoke SPL Token through inner instructions. Both levels must be inspected to recover visible edges.
    """
        
    instructions = []

    try:
        instructions.extend(tx["transaction"]["message"].get("instructions") or [])
        
    except Exception:
        pass

    for group in (tx.get("meta") or {}).get("innerInstructions") or []:
        instructions.extend(group.get("instructions") or [])

    return instructions


def parse_amount_from_transfer(info, source, destination, account_to_decimals):
    """
    Convert a parsed transfer amount into human-readable token units. transferChecked may provide a structured tokenAmount directly.
    Standard transfer instructions often provide only a raw integer amount, in which case decimal precision is recovered from source or destination token-balance metadata.
    When decimal metadata is unavailable, the raw on-chain integer amount is returned rather than discarding the transfer.
    """

    # Prefer the structured amount supplied by transferChecked because it may already contain decimal-adjusted values.
    token_amount = info.get("tokenAmount")

    if isinstance(token_amount, dict):
        for key in ["uiAmountString", "uiAmount", "amount"]:
            if token_amount.get(key) is not None:
                value = token_amount.get(key)

                if key == "amount" and token_amount.get("decimals") is not None:
                    return str(int(value) / (10 ** int(token_amount["decimals"])))

                return str(value)

    raw_amount = info.get("amount")

    if raw_amount is None:
        return None

    # Standard transfer instructions omit decimals, so infer them from either endpoint's token-balance metadata.
    decimals = account_to_decimals.get(source)

    if decimals is None:
        decimals = account_to_decimals.get(destination)

    if decimals is not None:
        try:
            return str(int(raw_amount) / (10 ** int(decimals)))
        except Exception:
            pass

    # Preserve the transfer even when human-readable scaling is impossible. In this fallback case, amount represents raw base units.
    return str(raw_amount)

# ============================================================
# EDGE EXTRACTION
# ============================================================

def parse_token_transfers_once(tx, signature, requested_mint):
    """
    Parse one transaction only once and extract transfer information that is independent of q1, q2, and q3.
    Experiment metadata is attached later when the transaction is distributed to the observation windows containing its timestamp.
    """

    rows = []

    if not tx:
        return rows

    meta = tx.get("meta") or {}

    # Failed transactions do not produce state-changing transfer edges.
    if meta.get("err") is not None:
        return rows

    timestamp = safe_int(tx.get("blockTime"))

    account_to_mint, account_to_decimals, account_to_owner = build_token_account_maps(tx)

    for instruction in flatten_instructions(tx):
        parsed = instruction.get("parsed")

        if not isinstance(parsed, dict):
            continue

        if instruction.get("program") not in ["spl-token", "spl-token-2022"]:
            continue

        if parsed.get("type") not in ["transfer", "transferChecked"]:
            continue

        info = parsed.get("info") or {}

        source = info.get("source")
        destination = info.get("destination")

        if not source or not destination:
            continue

        mint = info.get("mint") or account_to_mint.get(source) or account_to_mint.get(destination)

        # A transaction may contain transfers for several different tokens.
        if mint != requested_mint:
            continue

        amount = parse_amount_from_transfer(
            info=info,
            source=source,
            destination=destination,
            account_to_decimals=account_to_decimals
        )

        if amount is None:
            continue

        rows.append(
            {
                "token_mint": mint,
                "tx_signature": signature,
                "timestamp": timestamp,
                "source_address": account_to_owner.get(source) or source,
                "target_address": account_to_owner.get(destination) or destination,
                "amount": amount
            }
        )

    return rows


def build_edges_for_token(token_mint, experiment_windows):
    """
    Retrieve and parse transaction history only once for one token.
    The RPC scan extends from token creation to the largest remaining observation cutoff. Every parsed transfer is then distributed locally to each q1/q2/q3 graph whose half-open interval contains its timestamp.
    """

    if not experiment_windows:
        return {}, False, "ok"

    observation_start = experiment_windows[0]["observation_start"]

    # Validate that every q of this token starts at the same creation time.
    for window in experiment_windows:
        if window["observation_start"] != observation_start:
            return {}, False, "inconsistent_observation_start"

    max_observation_end = max(window["observation_end"] for window in experiment_windows)

    print(f" Token RPC window: [{observation_start}, {max_observation_end})")

    print(
        " Experiments collected together: "
        + ", ".join(
            window["experiment_id"]
            for window in experiment_windows
        )
    )

    # One output list is maintained for every independent graph sample.
    rows_by_job = {}

    for window in experiment_windows:
        job_key = make_edge_job_key(
            lp_address=window["lp_address"],
            token_mint=window["token_mint"],
            experiment_id=window["experiment_id"]
        )

        rows_by_job[job_key] = []

    # Signature history is retrieved only once, up to the largest cutoff.
    signatures, temporary_error, status = get_signatures_for_window(
        address=token_mint,
        observation_start=observation_start,
        observation_end=max_observation_end
    )

    if temporary_error or status != "ok":
        return {}, temporary_error, status

    print(f" In-window successful signatures found once for token: {len(signatures)}")

    available_transaction_count = 0
    unavailable_transaction_count = 0
    missing_block_time_count = 0
    outside_window_count = 0
    parsed_transfer_count = 0

    for signature in tqdm(signatures, desc="Processing token signatures", leave=False):
        # Each transaction body is retrieved only once.
        tx, tx_error = get_transaction(signature)

        # A broad RPC failure remains a token-level temporary error. The token will be retried later instead of being marked successfully complete during an RPC outage.
        if tx_error:
            return {}, True, "transaction_history_error"

        # A legitimate result=null affects only this transaction.
        if tx is None:
            unavailable_transaction_count += 1

            print(f" Transaction unavailable and skipped: {signature}")

            time.sleep(SLEEP_BETWEEN_RPC_CALLS)
            continue

        timestamp = safe_int(tx.get("blockTime"))

        if timestamp is None:
            missing_block_time_count += 1

            print(f" Transaction has no blockTime and was skipped: {signature}")

            time.sleep(SLEEP_BETWEEN_RPC_CALLS)
            continue

        # Defensive validation against inconsistent RPC responses.
        if timestamp < observation_start or timestamp >= max_observation_end:
            outside_window_count += 1

            print(f" Transaction outside token RPC window and skipped: {signature}; timestamp={timestamp}")

            time.sleep(SLEEP_BETWEEN_RPC_CALLS)
            continue

        available_transaction_count += 1

        # Parse the transaction only once, independent of q1/q2/q3.
        base_transfer_rows = parse_token_transfers_once(
            tx=tx,
            signature=signature,
            requested_mint=token_mint
        )

        parsed_transfer_count += len(base_transfer_rows)

        # A transfer is copied to every graph sample whose temporal interval contains its transaction timestamp.
        for window in experiment_windows:
            if not window["observation_start"] <= timestamp < window["observation_end"]:
                continue

            job_key = make_edge_job_key(
                lp_address=window["lp_address"],
                token_mint=window["token_mint"],
                experiment_id=window["experiment_id"]
            )

            for base_row in base_transfer_rows:
                rows_by_job[job_key].append(
                    {
                        "lp_address": window["lp_address"],
                        "token_mint": base_row["token_mint"],
                        "experiment_id": window["experiment_id"],
                        "observation_start": window["observation_start"],
                        "observation_end": window["observation_end"],

                        "tx_signature": base_row["tx_signature"],
                        "timestamp": base_row["timestamp"],
                        "source_address": base_row["source_address"],
                        "target_address": base_row["target_address"],
                        "amount": base_row["amount"]
                    }
                )

        time.sleep(SLEEP_BETWEEN_RPC_CALLS)

    skipped_transaction_count = unavailable_transaction_count + missing_block_time_count + outside_window_count

    print(
        " Token transaction collection summary:"
        f" signatures={len(signatures)},"
        f" available={available_transaction_count},"
        f" unavailable={unavailable_transaction_count},"
        f" missing_block_time={missing_block_time_count},"
        f" outside_window={outside_window_count},"
        f" skipped={skipped_transaction_count},"
        f" parsed_transfers_once={parsed_transfer_count}"
    )

    for window in experiment_windows:
        job_key = make_edge_job_key(
            lp_address=window["lp_address"],
            token_mint=window["token_mint"],
            experiment_id=window["experiment_id"]
        )

        print(f" Distributed edges for {window['experiment_id']}: {len(rows_by_job[job_key])}")

    return rows_by_job, False, "ok"

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Coordinate the complete resumable graph-edge collection workflow.
    The procedure:
        1. loads and validates token-experiment windows from rule-based labels;
        2. verifies Solana RPC availability;
        3. validates the edge-output schema;
        4. removes previously completed jobs from the queue;
        5. collects and saves transfer edges job by job;
        6. leaves temporary or incomplete jobs pending for later inspection;
        7. reports execution totals.
    """
        
    print("=" * 70)
    print("STARTED EDGES COLLECTION")
    print("=" * 70)

    # The rule-based label file is the authoritative input manifest for edge collection. This allows node and edge collectors to run independently after label generation has completed.
    windows = load_observation_windows(INPUT_FILE)

    print(f"Temporally validated token-experiment windows: {len(windows)}")

    if not check_rpc_health():
        raise RuntimeError("RPC is not healthy or not reachable.")
    print("RPC is healthy.")

    ensure_output_file_exists()

    # Progress entries represent completed jobs, including successful jobs with zero edge rows that cannot be inferred from the edge CSV alone.
    processed_edge_jobs = load_processed_edge_jobs()

    # Keep only q1/q2/q3 jobs whose pool-mint-experiment key is not already recorded as complete.
    remaining_windows = [
        window
        for window in windows
        if make_edge_job_key(
            lp_address=window["lp_address"],
            token_mint=window["token_mint"],
            experiment_id=window["experiment_id"]
        ) not in processed_edge_jobs
    ]

    # Group only the remaining jobs. Already processed q experiments are not requested or written again.
    remaining_token_groups = group_windows_by_token(remaining_windows)

    print(f"Already processed token-experiment jobs: {len(processed_edge_jobs)}")
    print(f"Remaining token-experiment jobs: {len(remaining_windows)}")
    print(f"Remaining unique tokens: {len(remaining_token_groups)}")
    print("-" * 70)

    saved_jobs = 0
    skipped_temporary_errors = 0
    skipped_incomplete_data = 0

    # Process one token at a time. All remaining q experiments of the token share one signature scan and one getTransaction pass.
    for token_group in tqdm(remaining_token_groups, desc="Processing edge tokens"):
        token_mint = token_group["token_mint"]
        experiment_windows = token_group["windows"]

        try:
            print("")
            print("-" * 70)

            print(f" Processing token once: {token_mint}; experiments={[window['experiment_id'] for window in experiment_windows]}")

            rows_by_job, temporary_error, status = (
                build_edges_for_token(
                    token_mint=token_mint,
                    experiment_windows=experiment_windows
                )
            )

            # A token-level RPC failure affects all q jobs being collected in this pass. None of them is marked complete.
            if temporary_error:
                skipped_temporary_errors += len(experiment_windows)

                print(f" Token skipped because of a temporary RPC error: {token_mint}")

                continue

            if status != "ok":
                skipped_incomplete_data += len(experiment_windows)

                print(f" Token skipped because RPC edge data is incomplete: {token_mint}; status={status}")

                continue

            # Each q remains an independent output and progress job even though its RPC data was collected jointly with the other q jobs.
            for window in experiment_windows:
                lp_address = window["lp_address"]
                experiment_id = window["experiment_id"]

                job_key = make_edge_job_key(
                    lp_address=lp_address,
                    token_mint=token_mint,
                    experiment_id=experiment_id
                )

                rows = rows_by_job.get(job_key, [])

                # Persist this experiment's edges independently.
                if not append_rows_to_output(rows):
                    print(f" Could not save edge rows for job: {job_key}")
                    continue

                # Mark the q job complete only after its rows are saved.
                if not mark_edge_job_as_processed(
                    lp_address=lp_address,
                    token_mint=token_mint,
                    experiment_id=experiment_id
                ):
                    print(f" Edge rows were saved, but progress was not updated: {job_key}")
                    continue

                saved_jobs += 1

                print(f" Edge job done: {job_key}. Rows added: {len(rows)}")

        except Exception as e:
            print(f" Error while processing token {token_mint}: {e}")
            
    print("-" * 70)
    print(f"Saved token-experiment jobs: {saved_jobs}")
    print(f"Skipped temporary RPC errors: {skipped_temporary_errors}")
    print(f"Skipped incomplete RPC edge data: {skipped_incomplete_data}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 70)
    print("FINISHED EDGES COLLECTION")
    print("=" * 70)


if __name__ == "__main__":
    main()
