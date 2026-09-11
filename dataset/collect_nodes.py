"""
This script reads the labels produced by the rule-based labeling stage and
creates one feature row for every valid (token_mint, experiment_id) pair.

Each experiment uses its own observation window starting at token creation.
Only data available inside that window is used for feature extraction. Any
qualifying liquidity event used to determine the final label must occur
strictly after the observation window, preventing temporal leakage.

The collector retrieves:
1. Token-creation metadata and signer accounts from Birdeye and Solana RPC.
2. Historical token-level liquidity near token creation and observation end.
3. Buy/sell volume and unique-wallet activity inside the observation window.

The output CSV contains one row per valid token-experiment pair. The output
CSV is also used as the source of truth for resume support so completed nodes
are not requested again after an interrupted execution.
"""

import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGS
# ============================================================

# TEST_MODE only limits the RPC fallback used to search for the oldest mint-related signature
TEST_MODE = 0
LIMIT_SIGNATURE_PAGES_FOR_CREATION_TIME = 10

# Birdeye trade-history pagination settings.
TRADE_PAGE_LIMIT = 100
TRADE_MAX_PAGES = 100

# Delay and retry settings reduce rate-limit pressure on the APIs.
SLEEP_BETWEEN_RPC_CALLS = 0.5
SLEEP_BETWEEN_API_CALLS = 0.8
MAX_RETRIES = 5

# Because liquidity history has one-minute resolution, a snapshot must be no more than one minute away from its target timestamp.
LIQUIDITY_SNAPSHOT_TOLERANCE_SECONDS = 60

# The labeling stage defines three observation-window experiments, q1 = short window, q2 = medium window, and q3 = long window.
VALID_EXPERIMENT_IDS = {
    "q1",
    "q2",
    "q3"
}

LABEL_FILE = "rule_based_labels.csv"

if TEST_MODE == 1:
    OUTPUT_FILE = "graph_nodes_test.csv"
    PROGRESS_FILE = "processed_nodes_test.txt"
else:
    OUTPUT_FILE = "graph_nodes.csv"
    PROGRESS_FILE = "processed_nodes.txt"

BIRDEYE_API_KEY = os.environ.get("BIRDEYE_API_KEY", "").strip()  # To set: $env:BIRDEYE_API_KEY="***"
BIRDEYE_BASE_URL = "https://public-api.birdeye.so"

RPC_URL = "https://api.mainnet-beta.solana.com"

# Identification and label metadata are stored first, followed by temporal boundaries, trade/activity features, creation metadata, and historical liquidity features.
OUTPUT_COLUMNS = [
    "lp_address",
    "token_mint",
    "experiment_id",
    "rug_pull_label",
    "label_status",

    "creation_time",
    "observation_window_seconds",
    "observation_start",
    "observation_end",
    "label_available_time",

    "volume_sell_change_percent",
    "volume_buy_change_percent",
    "unique_wallet_count",
    "unique_wallet_change_percent",

    "creators",
    "mint_authority",
    "freeze_authority",

    "liquidity_at_creation",
    "liquidity_at_observation_end"
]

# Creation metadata is identical across the q1, q2, and q3 experiments of the same token. Cache it once per mint to avoid repeating expensive Birdeye and RPC requests for every experiment.
MINT_CREATION_INFO_CACHE = {}

# ============================================================
# FILE / PROGRESS HELPERS
# ============================================================

def clean_string(value):
    """
    Convert nullable CSV/API values into a consistently stripped string.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def parse_timestamp(value):
    """
    Parse either textual timestamps or numeric Unix timestamps.
    Numeric magnitude is used to distinguish seconds, milliseconds, microseconds, and nanoseconds before converting everything to UTC.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()

    if text == "":
        return None

    try:
        numeric_value = float(text)
        if numeric_value > 1e17:
            ts = pd.to_datetime(numeric_value, unit="ns", utc=True, errors="coerce")
        elif numeric_value > 1e14:
            ts = pd.to_datetime(numeric_value, unit="us", utc=True, errors="coerce")
        elif numeric_value > 1e11:
            ts = pd.to_datetime(numeric_value, unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(numeric_value, unit="s", utc=True, errors="coerce")

    except Exception:
        ts = pd.to_datetime(text, utc=True, errors="coerce")

    if pd.isna(ts):
        return None

    return ts


def unix_timestamp_from_datetime(dt):
    if dt is None:
        return None

    try:
        return int(dt.timestamp())
    except Exception:
        return None


def safe_float(value):
    """
    Safely convert a nullable or textual value into a float without allowing malformed API fields to terminate the entire collection run.
    """

    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        text = str(value).strip()
        if text == "":
            return None

        return float(text)
    except Exception:
        return None


def safe_int(value):
    """
    Integer conversion is performed through safe_float because CSV values may be represented as strings.
    """

    number = safe_float(value)

    if number is None:
        return None

    return int(number)


def first_non_empty(mapping, keys):
    """
    Birdeye response field names may differ between endpoint versions. Return the first available alias from a prioritized list of keys.
    """

    if not isinstance(mapping, dict):
        return None

    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value

    return None


def normalize_binary_label(value):
    """
    Normalize common CSV representations of binary labels.
    """

    text = clean_string(value).lower()

    if text in ["1", "1.0", "true"]:
        return 1

    if text in ["0", "0.0", "false"]:
        return 0

    return None


def load_label_map(label_file):
    """
    Load one independent label entry for every (token_mint, experiment_id) pair.
    Only VALID binary labels are eligible for node-feature generation. IN_WINDOW_EVENT and UNKNOWN rows are excluded from the training-node dataset.
    """

    if not os.path.exists(label_file):
        raise RuntimeError(f"Label file not found: {label_file}. Run the label-building stage first.")

    try:
        df_labels = pd.read_csv(
            label_file,
            dtype=str,
            keep_default_na=False
        )

    except Exception as e:
        raise RuntimeError(f"Could not read label file {label_file}: {e}")

    required_columns = [
        "lp_address",
        "token_mint",
        "experiment_id",
        "observation_window_seconds",
        "observation_end_time",
        "rug_pull_label",
        "token_creation_time",
        "liquidity_drop_factor_time",
        "label_available_time",
        "label_status"
    ]

    missing_columns = [ column for column in required_columns if column not in df_labels.columns ]

    if missing_columns:
        raise ValueError(f"Label file {label_file} is missing columns: {missing_columns}")

    label_map = {}

    # Validate each label row independently because the same token may have different validity and labels under the m, n, and d observation windows.
    for _, row in df_labels.iterrows():
        token_mint = clean_string(row.get("token_mint"))

        lp_address = clean_string(row.get("lp_address"))

        experiment_id = clean_string(row.get("experiment_id"))

        label_status = clean_string(row.get("label_status"))

        rug_pull_label = normalize_binary_label(row.get("rug_pull_label"))

        # Only completed binary labels are allowed in the node-feature dataset.
        if token_mint == "" or lp_address == "" or experiment_id == "" or label_status != "VALID" or rug_pull_label is None:
            continue

        if experiment_id not in VALID_EXPERIMENT_IDS:
            raise ValueError(f"Unexpected experiment_id={experiment_id} for token={token_mint}. Expected one of {sorted(VALID_EXPERIMENT_IDS)}.")

        # Normalize all temporal fields to Unix seconds so label validation and feature extraction use one consistent time representation.
        token_creation_time = (unix_timestamp_from_datetime(parse_timestamp(row.get("token_creation_time"))))
        observation_end_time = (unix_timestamp_from_datetime(parse_timestamp(row.get("observation_end_time"))))
        label_available_time = (unix_timestamp_from_datetime(parse_timestamp(row.get("label_available_time"))))
        liquidity_event_time = (unix_timestamp_from_datetime(parse_timestamp(row.get("liquidity_drop_factor_time"))))

        observation_window_seconds = safe_int( row.get("observation_window_seconds"))

        if token_creation_time is None or observation_end_time is None or label_available_time is None or observation_window_seconds is None or observation_window_seconds <= 0:
            raise ValueError(f"Invalid label timestamps for token={token_mint}, experiment={experiment_id}")

        # The label and feature pipelines must use exactly the same observation boundary. Any mismatch would make the label and its node features refer to different temporal windows.
        expected_observation_end = token_creation_time + observation_window_seconds

        if observation_end_time != expected_observation_end:
            raise ValueError(f"Observation-window mismatch for token={token_mint}, experiment={experiment_id}. Expected {expected_observation_end}, found {observation_end_time}.")

        # Every positive label must have a qualifying liquidity event. A negative label may legitimately have no event.
        if rug_pull_label == 1 and liquidity_event_time is None:
            raise ValueError(f"Positive label has no liquidity event: token={token_mint}, experiment={experiment_id}")

        # The token mint and experiment identifier jointly define one unique node.
        label_key = (
            token_mint,
            experiment_id
        )

        if label_key in label_map:
            raise ValueError(f"Duplicate label row for token={token_mint}, experiment={experiment_id}")

        label_map[label_key] = {
            "lp_address": lp_address,
            "token_mint": token_mint,
            "experiment_id": experiment_id,
            "rug_pull_label": rug_pull_label,
            "label_status": label_status,
            "token_creation_time": token_creation_time,
            "observation_window_seconds": observation_window_seconds,
            "observation_end_time": observation_end_time,
            "liquidity_event_time": liquidity_event_time,
            "label_available_time": label_available_time
        }

    return label_map


def load_label_nodes(label_map):
    """
    Convert the label map into one node-input row for every valid token-experiment pair.
    """

    rows = []

    for (token_mint, experiment_id) in sorted(label_map):
        label = label_map[(token_mint, experiment_id)]

        rows.append(
            {
                "lp_address": label["lp_address"],
                "token_mint": token_mint,
                "experiment_id": experiment_id,
                "rug_pull_label": label["rug_pull_label"],
                "label_status": label["label_status"],
                "token_creation_time": label["token_creation_time"],
                "observation_window_seconds": label["observation_window_seconds"],
                "observation_end_time": label["observation_end_time"],
                "liquidity_event_time": label["liquidity_event_time"],
                "label_available_time": label["label_available_time"]
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "lp_address",
            "token_mint",
            "experiment_id",
            "rug_pull_label",
            "label_status",
            "token_creation_time",
            "observation_window_seconds",
            "observation_end_time",
            "liquidity_event_time",
            "label_available_time"
        ]
    )


def make_node_key(lp_address, token_mint, experiment_id):
    """
    Include the pool, mint, and experiment in the resume key so different experiment windows for the same token are never treated as one node.
    """

    return (f"{lp_address}|{token_mint}|{experiment_id}")


def load_processed_nodes():
    """
    Use the output CSV as the source of truth for completed token-experiment nodes.
    """

    output_keys = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            output_rows = pd.read_csv(
                OUTPUT_FILE,
                dtype=str,
                usecols=[
                    "lp_address",
                    "token_mint",
                    "experiment_id"
                ],
                keep_default_na=False
            )

            for _, row in output_rows.iterrows():
                lp_address = clean_string(row.get("lp_address"))
                token_mint = clean_string(row.get("token_mint"))
                experiment_id = clean_string(row.get("experiment_id"))

                if lp_address and token_mint and experiment_id:
                    output_keys.add(
                        make_node_key(
                            lp_address=lp_address,
                            token_mint=token_mint,
                            experiment_id=experiment_id
                        )
                    )

        except pd.errors.EmptyDataError:
            pass

        except Exception as e:
            raise RuntimeError(f"Could not inspect existing output rows: {e}")

    progress_keys = set()

    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
                progress_keys = { line.strip() for line in file if line.strip() }

        except Exception as e:
            print(f"Could not read progress file: {e}")

    # A crash may occur after writing the progress file or after manually deleting the CSV. Such stale progress entries must not suppress a node that is absent from the actual output dataset.
    stale_progress_keys = progress_keys - output_keys
    
    if stale_progress_keys:
        print(f"Ignoring {len(stale_progress_keys)} progress entries that have no matching row in the output CSV.")

    return output_keys


def mark_node_as_processed(lp_address, token_mint, experiment_id):
    node_key = make_node_key(
        lp_address=lp_address,
        token_mint=token_mint,
        experiment_id=experiment_id
    )

    try:
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(node_key + "\n")
            f.flush()
            os.fsync(f.fileno())
        return True

    except Exception as e:
        print(f"Could not update progress file for node {node_key}: {e}")
        return False


def ensure_output_file_exists():
    try:
        if not os.path.exists(OUTPUT_FILE):
            pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(OUTPUT_FILE, index=False)
            return

        existing_columns = list(pd.read_csv(OUTPUT_FILE, nrows=0).columns)

        if existing_columns != OUTPUT_COLUMNS:
            raise RuntimeError(f"Output file {OUTPUT_FILE} has an incompatible header. Delete it before running this collector.\nExpected: {OUTPUT_COLUMNS}\nFound: {existing_columns}")

    except RuntimeError:
        raise

    except Exception as e:
        raise RuntimeError(f"Could not validate output file: {e}")


def append_row_to_output(row):
    try:
        df_row = pd.DataFrame([row], columns=OUTPUT_COLUMNS)

        df_row.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=False,
            index=False
        )

        return True

    except Exception as e:
        print(f"Could not append row to output file: {e}")
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
        response = requests.post(RPC_URL, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("result") == "ok"

    except Exception as e:
        print(f"RPC health check failed: {e}")
        return False


def rpc_call(method, params):
    """
    Shared JSON-RPC wrapper with exponential backoff for transport failures and rate limits. The returned error flag distinguishes retryable data access failures from valid empty RPC results.
    """

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(RPC_URL, json=payload, timeout=20)

            # Exponential backoff prevents immediate repeated requests while the public RPC endpoint is rate limiting the client.
            if response.status_code == 429:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            if "error" in data:
                error_message = str(data["error"])

                if "Too many requests" in error_message or "rate" in error_message.lower():
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue

                print(f" RPC error in {method}: {error_message}")
                return None, True

            return data.get("result"), False

        except Exception:
            wait_time = 2 ** attempt
            time.sleep(wait_time)
            continue

    print(f" RPC failed for method: {method}")
    return None, True


def get_transaction_rpc(signature):
    """
    Retrieve a historical transaction in jsonParsed format so signer flags and parsed SPL Token initialization instructions can be inspected.
    """

    params = [
        signature,
        {
            "encoding": "jsonParsed",
            "maxSupportedTransactionVersion": 0,
            "commitment": "finalized"
        }
    ]

    return rpc_call("getTransaction", params)


def get_oldest_signature_for_address(address):
    """
    Go backward through all signatures associated with an address.
    Solana returns signatures from newest to oldest. Pagination therefore continues using the final signature of each page as the 'before' cursor until the earliest available record is reached.
    In TEST_MODE the scan stops after a fixed number of pages. In the final run it continues through the complete available history.
    """

    oldest_item = None
    before = None
    page_count = 0

    while True:
        options = {
            "limit": 1000
        }

        if before:
            options["before"] = before

        result, had_error = rpc_call("getSignaturesForAddress", [address, options])

        if had_error:
            print(f" Temporary RPC error while getting signatures for {address}.")
            return None, True

        if result is None:
            return oldest_item, False

        if len(result) == 0:
            return oldest_item, False

        # RPC pages are newest-first, so the final entry is the oldest signature observed on the current page.
        oldest_item = result[-1]
        before = oldest_item.get("signature")

        page_count += 1

        # The page limit is a speed safeguard for local testing only and is intentionally disabled for final feature generation.
        if TEST_MODE == 1 and page_count >= LIMIT_SIGNATURE_PAGES_FOR_CREATION_TIME:
            return oldest_item, False

        if page_count % 10 == 0:
            print(f" Creation-history pagination: address={address}, pages={page_count}, signatures_scanned={page_count * 1000}")

        if len(result) < 1000:
            break

        if not before:
            break

        time.sleep(SLEEP_BETWEEN_RPC_CALLS)

    return oldest_item, False


def get_account_keys(tx):
    """
    Normalize transaction account keys into a common structure containing each public key and its signer flag.
    """
    
    keys = []

    try:
        raw_keys = tx["transaction"]["message"]["accountKeys"]
    except Exception:
        return []

    for item in raw_keys:
        if isinstance(item, str):
            keys.append({
                "pubkey": item,
                "signer": False
            })
        elif isinstance(item, dict):
            keys.append({
                "pubkey": item.get("pubkey"),
                "signer": bool(item.get("signer"))
            })

    return keys


def flatten_instructions(tx):
    """
    SPL Token initialization may appear either in top-level instructions or in inner instructions created by another program. Flatten both sources so the mint initialization is not missed.
    """
    
    all_instructions = []

    try:
        main_instructions = tx["transaction"]["message"].get("instructions") or []
        for ix in main_instructions:
            all_instructions.append(ix)
    except Exception:
        pass

    meta = tx.get("meta") or {}
    inner_groups = meta.get("innerInstructions") or []

    for group in inner_groups:
        for ix in group.get("instructions") or []:
            all_instructions.append(ix)

    return all_instructions


def get_mint_creation_info(token_mint):
    """
    Recover creation-time metadata for one token mint.
    Birdeye is queried first because it may directly provide the mint creation timestamp and transaction signature.
    If either value is unavailable, Solana RPC scans the mint's signature history and uses the oldest available transaction as the fallback creation transaction.
    The selected transaction is then parsed to obtain non-mint signer accounts used as creator-side accounts, the initial mint authority, and the initial freeze authority.
    Results are cached because this metadata is shared by all experiments of the same token.
    """

    # Reuse metadata collected for a previous experiment of the same mint.
    if token_mint in MINT_CREATION_INFO_CACHE:
        return MINT_CREATION_INFO_CACHE[token_mint], False

    info = {
        "token_creation_time": None,
        "creation_signature": None,
        "creators": [],
        "mint_authority": None,
        "freeze_authority": None
    }

    # Prefer the dedicated creation-info endpoint because it avoids scanning a potentially large RPC signature history.
    creation_payload, creation_endpoint_error = birdeye_get(
        "/defi/token_creation_info",
        params={
            "address": token_mint
        },
        allow_forbidden=True
    )

    if not creation_endpoint_error and isinstance(creation_payload, dict):
        creation_data = creation_payload.get("data")
        if isinstance(creation_data, dict):
            info["token_creation_time"] = safe_int(
                first_non_empty(
                    creation_data,
                    [
                        "block_unix_time",
                        "blockUnixTime",
                        "block_time",
                        "blockTime",
                        "timestamp",
                        "time"
                    ]
                )
            )

            info["creation_signature"] = clean_string(
                first_non_empty(
                    creation_data,
                    [
                        "tx_hash",
                        "txHash",
                        "signature",
                        "transaction_hash",
                        "transactionHash"
                    ]
                )
            ) or None

    # RPC fallback is required when either the creation time or the creation transaction signature is unavailable.
    if info["token_creation_time"] is None or not info["creation_signature"]:
        oldest_item, had_error = get_oldest_signature_for_address(token_mint)

        if had_error:
            return info, True

        if not oldest_item:
            MINT_CREATION_INFO_CACHE[token_mint] = info
            return info, False

        if info["token_creation_time"] is None:
            info["token_creation_time"] = oldest_item.get("blockTime")

        if not info["creation_signature"]:
            info["creation_signature"] = oldest_item.get("signature")

    if info["creation_signature"]:
        tx, tx_error = get_transaction_rpc(info["creation_signature"])

        if tx_error:
            return info, True

        if tx:
            # In this dataset, "creators" means non-mint signer accounts from the mint creation transaction. It does not refer to later Metaplex creator metadata.
            account_keys = get_account_keys(tx)

            # Exclude the mint account itself because a newly created mint may also be a signer, but it is not a creator wallet. dict.fromkeys removes duplicate signer addresses while preserving transaction order.
            info["creators"] = list(dict.fromkeys(key.get("pubkey") for key in account_keys if key.get("pubkey") and key.get("signer") and key.get("pubkey") != token_mint))

            # Search every parsed instruction for the SPL Token initializeMint or initializeMint2 operation belonging specifically to this mint.
            for ix in flatten_instructions(tx):
                parsed = ix.get("parsed")
                if not isinstance(parsed, dict):
                    continue

                ix_type = parsed.get("type")
                parsed_info = parsed.get("info") or {}

                if ix_type not in ["initializeMint", "initializeMint2"]:
                    continue

                mint = clean_string(parsed_info.get("mint"))
                # A transaction may initialize more than one account so ignore instructions that explicitly refer to another mint.
                if mint and mint != token_mint:
                    continue

                mint_authority = clean_string(parsed_info.get("mintAuthority"))
                freeze_authority = clean_string(parsed_info.get("freezeAuthority"))

                # Empty authority fields remain None in the output rather than being replaced with an artificial value.
                if mint_authority:
                    info["mint_authority"] = mint_authority

                if freeze_authority:
                    info["freeze_authority"] = freeze_authority

    MINT_CREATION_INFO_CACHE[token_mint] = info
    return info, False

# ============================================================
# BIRDEYE HELPERS
# ============================================================

def get_birdeye_headers(content_type=None):
    """
    Build the common Solana-specific Birdeye headers and attach the API key only when it has been configured.
    """
    
    headers = {
        "accept": "application/json",
        "x-chain": "solana"
    }

    if content_type:
        headers["content-type"] = content_type

    if BIRDEYE_API_KEY:
        headers["X-API-KEY"] = BIRDEYE_API_KEY

    return headers


def http_get_json(url, headers=None, params=None, service_name="HTTP", allow_forbidden=False):
    """
    Execute a GET request with retry and explicit HTTP-status handling.
    Rate limits and transient transport failures are retried with exponential backoff. Authentication, permission, and malformed-request errors are treated as fatal because continuing would create a large dataset with silently missing API-derived features.
    allow_forbidden is used only where an RPC fallback is available.
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=30
            )

            if response.status_code == 429:
                wait_time = 2 ** attempt
                print(f" {service_name} rate limited. Sleeping {wait_time}s.")
                time.sleep(wait_time)
                continue

            if response.status_code == 401:
                raise SystemExit(f"\nFATAL: {service_name} returned 401. The API key is missing or invalid.")

            # A forbidden response may be tolerated only for endpoints that have a complete alternative data source, such as creation-info via Solana RPC.
            if response.status_code == 403:
                if allow_forbidden:
                    return None, False

                raise SystemExit(f"\nFATAL: {service_name} returned 403. The required endpoint is not available under the current API plan.")

            if response.status_code in [400, 422]:
                raise SystemExit(f"\nFATAL: {service_name} returned {response.status_code}. URL: {url}\nResponse: {response.text[:300]}")

            if response.status_code == 404:
                return {
                    "success": False,
                    "_status_code": 404,
                    "message": "not_found"
                }, False

            response.raise_for_status()
            return response.json(), False

        except Exception as e:
            wait_time = 2 ** attempt
            print(f" {service_name} failed attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            time.sleep(wait_time)

    print(f" {service_name} failed, url={url}")
    return None, True


def http_post_json(url, headers=None, payload=None, service_name="HTTP"):
    """
    POST equivalent of the shared HTTP helper. It is retained for Birdeye endpoints that require JSON request bodies.
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 429:
                wait_time = 2 ** attempt
                print(f" {service_name} rate limited. Sleeping {wait_time}s.")
                time.sleep(wait_time)
                continue

            if response.status_code in [401, 403]:
                return {
                    "success": False,
                    "message": response.text[:300],
                    "_status_code": response.status_code
                }, False

            if response.status_code == 404:
                return None, False

            if response.status_code in [400, 422]:
                return {
                    "success": False,
                    "message": response.text[:300],
                    "_status_code": response.status_code
                }, False

            response.raise_for_status()
            return response.json(), False

        except Exception as e:
            wait_time = 2 ** attempt
            print(f" {service_name} failed attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            time.sleep(wait_time)

    print(f" {service_name} failed, url={url}")
    return None, True


def birdeye_get(path, params=None, allow_forbidden=False):
    """
    Combine the Birdeye base URL, shared headers, request execution, and pacing delay into one endpoint-level helper.
    """

    url = BIRDEYE_BASE_URL + path

    data, had_error = http_get_json(
        url=url,
        headers=get_birdeye_headers(),
        params=params,
        service_name="Birdeye",
        allow_forbidden=allow_forbidden
    )

    time.sleep(SLEEP_BETWEEN_API_CALLS)

    return data, had_error


def birdeye_post(path, payload=None):
    """
    Birdeye POST helper using the same authentication and pacing policy.
    """

    url = BIRDEYE_BASE_URL + path

    data, had_error = http_post_json(
        url=url,
        headers=get_birdeye_headers(content_type="application/json"),
        payload=payload,
        service_name="Birdeye"
    )

    time.sleep(SLEEP_BETWEEN_API_CALLS)

    return data, had_error


def extract_items(data):
    """
    Normalize common Birdeye response layouts into one list of records.
    Different endpoints and API versions may place result arrays directly under data or under nested keys such as items, list, or records.
    Returning a single normalized list keeps later parsers independent of those response-layout differences.
    """

    if not isinstance(data, dict):
        return []

    payload = data.get("data")

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["items", "list", "records", "data"]:
            value = payload.get(key)

            if isinstance(value, list):
                return value

    for key in ["items", "list", "records"]:
        value = data.get(key)

        if isinstance(value, list):
            return value

    return []


def extract_history_timestamp(item):
    """
    Extract a historical record timestamp using the known field aliases returned by different Birdeye history endpoints.
    """

    if not isinstance(item, dict):
        return None

    return safe_int(
        first_non_empty(
            item,
            [
                "unix_time",
                "unixTime",
                "block_unix_time",
                "blockUnixTime",
                "timestamp",
                "time"
            ]
        )
    )


def extract_total_liquidity(item):
    """
    Extract total token-level USD liquidity from either a direct scalar field or a nested liquidity object.
    """

    if not isinstance(item, dict):
        return None

    direct_value = first_non_empty(
        item,
        [
            "total_liquidity",
            "totalLiquidity",
            "total_liquidity_usd",
            "totalLiquidityUsd",
            "liquidity_usd",
            "liquidityUsd",
            "liquidity",
            "value",
            "o",
            "open"
        ]
    )

    direct_value = safe_float(direct_value)

    if direct_value is not None:
        return direct_value

    liquidity_object = item.get("liquidity")

    if isinstance(liquidity_object, dict):
        return safe_float(
            first_non_empty(
                liquidity_object,
                [
                    "total",
                    "usd",
                    "value",
                    "totalLiquidity",
                    "total_liquidity"
                ]
            )
        )

    return None


def get_birdeye_liquidity_snapshot(token_mint, target_time, direction):
    """
    Fetch the nearest token-level one-minute liquidity snapshot.
    A forward snapshot is requested at token creation because no observation before creation is meaningful.
    A backward snapshot is requested at the observation cutoff so no post-window liquidity value can leak into the node features.
    The caller later verifies that the selected snapshot is no more than LIQUIDITY_SNAPSHOT_TOLERANCE_SECONDS away from the requested timestamp.
    """

    # Request exactly one nearest one-minute token-level liquidity observation.
    params = {
        "address": token_mint,
        "resolution": "1m",
        "time": target_time,
        "direction": direction,
        "count": 1
    }

    data, had_error = birdeye_get(
        "/defi/v3/liquidity/history/token",
        params=params
    )

    if had_error:
        return None, None, "temporary_error"

    if isinstance(data, dict) and data.get("success") is False:
        status_code = data.get("_status_code")

        if status_code in [401, 403]:
            return None, None, "permission_denied"

        return None, None, "unavailable"

    items = extract_items(data)

    if not items:
        return None, None, "empty"

    item = items[0]

    snapshot_time = extract_history_timestamp(item)
    liquidity = extract_total_liquidity(item)

    if snapshot_time is None:
        return None, None, "missing_timestamp"

    if liquidity is None or liquidity < 0:
        return None, snapshot_time, "unparsed_liquidity"

    # Independently enforce the requested temporal direction even if the API returns an unexpected record.
    if direction == "forward" and snapshot_time < target_time:
        return (
            None,
            snapshot_time,
            "invalid_forward_snapshot"
        )

    if direction == "back" and snapshot_time > target_time:
        return (
            None,
            snapshot_time,
            "invalid_backward_snapshot"
        )

    return liquidity, snapshot_time, "ok"


def extract_trade_timestamp(item):
    if not isinstance(item, dict):
        return None

    return safe_int(
        first_non_empty(
            item,
            [
                "block_unix_time",
                "blockUnixTime",
                "unix_time",
                "unixTime",
                "timestamp",
                "time"
            ]
        )
    )


def infer_trade_side(item, token_mint):
    """
    Classify a swap as a buy or sell of the requested token.
    The explicit Birdeye side field is preferred. When it is absent, the direction is inferred from the nested from/to token addresses, receiving the requested token is a buy, while sending it is a sell.
    """

    if not isinstance(item, dict):
        return None

    explicit_side = clean_string(item.get("side")).lower()

    if explicit_side in {"buy", "sell"}:
        return explicit_side

    from_data = item.get("from")
    to_data = item.get("to")

    from_address = ""

    if isinstance(from_data, dict):
        from_address = clean_string(from_data.get("address"))

    to_address = ""

    if isinstance(to_data, dict):
        to_address = clean_string(to_data.get("address"))

    # Receiving the requested token means the wallet bought it.
    if to_address == token_mint and from_address != token_mint:
        return "buy"

    # Sending the requested token means the wallet sold it.
    if from_address == token_mint and to_address != token_mint:
        return "sell"

    return None


def get_birdeye_trades(token_mint, time_from, time_to):
    """
    Retrieve all deduplicated token swaps in the half-open interval [time_from, time_to).
    Results are paginated, filtered again locally by timestamp, and deduplicated by transaction hash and instruction indexes.
    A history is considered complete only when pagination ends naturally and every returned record contains a parseable timestamp.
    """
        
    all_items = []
    seen_trade_ids = set()

    unparsed_timestamp_count = 0

    offset = 0
    page_count = 0

    while page_count < TRADE_MAX_PAGES:
        if page_count > 0 and page_count % 5 == 0:
            print(f" Birdeye trades pagination: token={token_mint}, pages={page_count}, rows={len(all_items)}")

        params = {
            "address": token_mint,
            "sort_by": "block_unix_time",
            "sort_type": "desc",
            "tx_type": "swap",
            "ui_amount_mode": "scaled",
            "after_time": max(1, time_from - 1),  # Birdeye treats after_time as an exclusive lower bound. Query one second earlier, then enforce the exact [time_from, time_to) interval locally.
            "before_time": time_to,
            "offset": offset,
            "limit": TRADE_PAGE_LIMIT
        }

        data, had_error = birdeye_get(
            "/defi/v3/token/txs",
            params=params
        )

        if had_error:
            return [], True, False

        if not data:
            return (
                all_items,
                False,
                unparsed_timestamp_count == 0
            )

        if isinstance(data, dict) and data.get("success") is False:
            return all_items, False, False

        items = extract_items(data)

        # Validate and deduplicate every record before it is allowed to contribute to volume or wallet features.
        for item in items:
            trade_time = extract_trade_timestamp(item)

            # A trade without a timestamp cannot be safely assigned to the experiment observation interval.
            if trade_time is None:
                unparsed_timestamp_count += 1
                continue

            # Enforce the half-open feature interval [start, end).
            if not time_from <= trade_time < time_to:
                continue

            tx_hash = clean_string(item.get("tx_hash") or item.get("txHash"))
            ins_index = clean_string(item.get("ins_index"))
            inner_ins_index = clean_string(item.get("inner_ins_index"))

            # Instruction indexes are unique only inside one transaction. Therefore, the compact identifier is safe only when the transaction hash is available.
            if tx_hash:
                trade_id = f"{tx_hash}|{ins_index}|{inner_ins_index}"
            else:
                trade_id = json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":")
                )

            if trade_id in seen_trade_ids:
                continue

            seen_trade_ids.add(trade_id)
            all_items.append(item)

        # A short page indicates natural pagination completion. Timestamp parsing failures still make the resulting history incomplete.
        if len(items) < TRADE_PAGE_LIMIT:
            if unparsed_timestamp_count > 0:
                print(f" Unparseable trade timestamps: token={token_mint}, count={unparsed_timestamp_count}")

            return (
                all_items,
                False,
                unparsed_timestamp_count == 0
            )

        offset += TRADE_PAGE_LIMIT
        page_count += 1

    # Reaching TRADE_MAX_PAGES means the complete history is unknown, so the caller must not compute potentially underestimated numerical features.
    return all_items, False, False


# ============================================================
# FEATURE HELPERS
# ============================================================

def compute_percent_change(current_value, previous_value):
    """
    Compute percentage change from a baseline value.
    A zero baseline returns None because percentage change is mathematically undefined in that case. Returning zero would incorrectly imply that no change occurred.
    """

    current_value = safe_float(current_value)
    previous_value = safe_float(previous_value)

    if current_value is None or previous_value is None:
        return None

    if previous_value == 0:
        return None

    return ((current_value - previous_value) / previous_value) * 100.0


def compute_trade_features(token_mint, observation_start, observation_end):
    """
    Compute historical trade and wallet features for one experiment.
    The observation interval is divided into two consecutive halves:
        first half  = [observation_start, midpoint)
        second half = [midpoint, observation_end)
    Buy volume, sell volume, and unique-wallet activity in the second half are compared with their corresponding first-half baselines. This keeps every input inside the experiment window and avoids using a nonexistent pre-creation baseline.
    Also, unique_wallet_count is the number of distinct wallets observed anywhere in the complete experiment window.
    Numerical features are returned only when both halves have complete and fully parseable trade histories.
    """

    window_seconds = observation_end - observation_start

    if window_seconds < 2:
        return (
            {},
            False,
            "invalid_trade_window"
        )

    # Integer division assigns an odd extra second to the second half while preserving two non-overlapping half-open intervals.
    midpoint = observation_start + window_seconds // 2

    # Fetch each half independently so their totals can be compared directly.
    first_half_trades, first_error, first_complete = (
        get_birdeye_trades(
            token_mint=token_mint,
            time_from=observation_start,
            time_to=midpoint
        )
    )

    if first_error:
        return (
            {},
            True,
            "trade_temporary_error_first_half"
        )

    second_half_trades, second_error, second_complete = (
        get_birdeye_trades(
            token_mint=token_mint,
            time_from=midpoint,
            time_to=observation_end
        )
    )

    if second_error:
        return (
            {},
            True,
            "trade_temporary_error_second_half"
        )

    # Never calculate volume or wallet features from capped, rejected, or partially timestamped trade histories.
    incomplete_statuses = []

    if not first_complete:
        incomplete_statuses.append("first_half_trade_window_incomplete")

    if not second_complete:
        incomplete_statuses.append("second_half_trade_window_incomplete")

    # Partial trade histories must not produce numerical features, because their volumes and wallet counts would be underestimated.
    if incomplete_statuses:
        return (
            {},
            False,
            ",".join(incomplete_statuses)
        )

    def summarize(trades):
        """
        Aggregate one half-window of fully parseable swaps.
        Each accepted trade must provide a buy/sell classification for the requested token, non-negative USD volume, a wallet owner, and a timestamp already validated by get_birdeye_trades.
        The function returns buy/sell totals, the set of unique wallets, and the number of rejected records.
        """

        buy_volume = 0.0
        sell_volume = 0.0
        unique_wallets = set()
        unparsed_trade_count = 0

        for item in trades:
            trade_side = infer_trade_side(
                item=item,
                token_mint=token_mint
            )

            volume_usd = safe_float(
                first_non_empty(
                    item,
                    [
                        "volume_usd",
                        "volumeUsd",
                        "volumeUSD"
                    ]
                )
            )

            owner = clean_string(item.get("owner"))

            if trade_side not in {"buy", "sell"}:
                unparsed_trade_count += 1
                continue

            if volume_usd is None or volume_usd < 0:
                unparsed_trade_count += 1
                continue

            # Without an owner, the trade cannot safely contribute to the unique-wallet features.
            if owner == "":
                unparsed_trade_count += 1
                continue

            unique_wallets.add(owner)

            if trade_side == "buy":
                buy_volume += volume_usd

            else:
                sell_volume += volume_usd

        return {
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "unique_wallets": unique_wallets,
            "unparsed_trade_count": unparsed_trade_count
        }

    first_summary = summarize(first_half_trades)
    second_summary = summarize(second_half_trades)

    first_unparsed_count = first_summary["unparsed_trade_count"]
    second_unparsed_count = second_summary["unparsed_trade_count"]

    # Do not generate trade features from partially parseable histories because volumes and wallet counts would be underestimated.
    if first_unparsed_count > 0 or second_unparsed_count > 0:
        return (
            {},
            False,
            f"unparsed_trade_records:first={first_unparsed_count},second={second_unparsed_count}"
        )

    # Set union counts each wallet once even when it appears in both halves.
    all_unique_wallets = first_summary["unique_wallets"] | second_summary["unique_wallets"]

    # Percentage-change features compare the second half with the first half. The absolute unique-wallet feature covers the entire observation window.
    features = {
        "volume_sell_change_percent": (
            compute_percent_change(
                second_summary["sell_volume"],
                first_summary["sell_volume"]
            )
        ),
        "volume_buy_change_percent": (
            compute_percent_change(
                second_summary["buy_volume"],
                first_summary["buy_volume"]
            )
        ),
        "unique_wallet_count": len(all_unique_wallets),
        "unique_wallet_change_percent": (
            compute_percent_change(
                len(second_summary["unique_wallets"]),
                len(first_summary["unique_wallets"])
            )
        )
    }

    return features, False, "ok"


def validate_temporal_window(rug_pull_label, liquidity_event_time, label_available_time, observation_end):
    """
    Validate that a node can be built without temporal leakage.
    A valid node must satisfy all of the following:
        1. The final label is binary.
        2. The label becomes available strictly after the observation window.
        3. Any qualifying liquidity event occurs strictly after the cutoff.
        4. Every positive label has a qualifying liquidity-event timestamp.
    """

    normalized_label = normalize_binary_label(rug_pull_label)

    if normalized_label is None:
        return (
            False, 
            "invalid_binary_label"
        )

    if label_available_time is None:
        return (
            False,
            "missing_label_available_time_strict_skip"
        )

    # A label that is already available by the feature cutoff could expose post-outcome information to the model.
    if label_available_time <= observation_end:
        return (
            False,
            "label_available_not_after_observation_window"
        )

    # Any detected qualifying liquidity event must occur strictly after the feature observation window, regardless of whether the final label is 0 or 1.
    if liquidity_event_time is not None and liquidity_event_time <= observation_end:
        return (
            False,
            "liquidity_event_not_after_observation_window"
        )

    # Positive labels must always have a qualifying liquidity event.
    if normalized_label == 1 and liquidity_event_time is None:
        return (
            False,
            "missing_liquidity_event_time_strict_skip"
        )

    return True, "temporally_valid"


def evaluate_required_features(row):
    """
    Audit whether the core numerical feature set is complete.
    This status is used internally for diagnostics. Missing values remain empty in the output so downstream preprocessing can apply an explicit missing-data policy.
    """
    
    required_columns = [
        "creation_time",
        "observation_start",
        "observation_end",
        "volume_sell_change_percent",
        "volume_buy_change_percent",
        "unique_wallet_count",
        "unique_wallet_change_percent",
        "liquidity_at_creation",
        "liquidity_at_observation_end"
    ]

    missing_columns = []

    for column in required_columns:
        value = row.get(column)

        if value is None or clean_string(value) == "":
            missing_columns.append(column)

    if not missing_columns:
        return True, "complete"

    return (
        False,
        "missing:" + ",".join(missing_columns)
    )


def collect_node_features(lp_address, token_mint, experiment_id, rug_pull_label, label_status, label_creation_time, label_observation_window_seconds, label_observation_end_time, liquidity_event_time, label_available_time):
    """
    Build one historical feature row for one token-experiment pair.
    Processing order:
        1. Initialize the output and internal diagnostic fields.
        2. Validate the label-provided experiment boundaries.
        3. Reject temporally invalid or leakage-prone nodes.
        4. Retrieve creation signers and initial token authorities.
        5. Retrieve liquidity near creation and observation end.
        6. Retrieve and summarize trades inside the experiment window.
        7. Audit feature completeness and return the row.
    Temporary API/RPC errors cause the node to remain unprocessed so a later execution can retry it. Structurally invalid temporal nodes are skipped.
    """

    # Initialize every potential output field explicitly. Internal diagnostic fields may be added to this dictionary but are excluded from OUTPUT_COLUMNS and therefore do not appear in the final CSV.
    row = {
        "lp_address": lp_address,
        "token_mint": token_mint,
        "experiment_id": experiment_id,
        "rug_pull_label": rug_pull_label,
        "label_status": label_status,

        "creation_time": label_creation_time,
        "observation_window_seconds": label_observation_window_seconds,
        "observation_start": label_creation_time,
        "observation_end": label_observation_end_time,
        "label_available_time": label_available_time,
        "temporal_validation_status": None,

        "volume_sell_change_percent": None,
        "volume_buy_change_percent": None,
        "unique_wallet_count": None,
        "unique_wallet_change_percent": None,

        "creators": None,
        "mint_authority": None,
        "freeze_authority": None,

        "liquidity_at_creation": None,
        "liquidity_at_observation_end": None,

        "required_feature_status": None,
        "feature_status": None
    }

    # Internal control flags communicate retry/skip decisions to main().
    row["_temporary_error"] = False
    row["_temporal_skip"] = False

    status_parts = [
        f"experiment_{experiment_id}",
        "creation_anchor_label_file"
    ]

    # The label file is the temporal anchor for feature extraction. API-derived creation timestamps are not allowed to redefine the experiment window.
    creation_time = safe_int(label_creation_time)
    observation_window_seconds = safe_int(label_observation_window_seconds)
    observation_end = safe_int(label_observation_end_time)

    if creation_time is None or observation_window_seconds is None or observation_window_seconds <= 0 or observation_end is None:
        row["_temporal_skip"] = True

        status_parts.append("invalid_experiment_time_metadata")

        row["feature_status"] = ",".join(status_parts)
        return row

    # Recheck label-window consistency immediately before collecting any external feature data.
    expected_observation_end = creation_time + observation_window_seconds

    if observation_end != expected_observation_end:
        row["_temporal_skip"] = True

        status_parts.append("observation_end_mismatch")

        row["feature_status"] = ",".join(status_parts)
        return row

    # Reject the node before making feature requests when the label or event timing would create temporal leakage.
    is_temporally_valid, temporal_status = (
        validate_temporal_window(
            rug_pull_label=rug_pull_label,
            liquidity_event_time=liquidity_event_time,
            label_available_time=label_available_time,
            observation_end=observation_end
        )
    )

    row["temporal_validation_status"] = temporal_status

    if not is_temporally_valid:
        row["_temporal_skip"] = True


        status_parts.append(temporal_status)

        row["feature_status"] = ",".join(status_parts)
        return row

    # Creation metadata is independent of experiment length and is reused from the per-mint cache whenever another experiment of this token is processed.
    mint_info, mint_info_error = get_mint_creation_info(token_mint)

    if mint_info_error:
        row["_temporary_error"] = True

        status_parts.append("mint_creation_info_temporary_error")

        row["feature_status"] = ",".join(status_parts)
        return row

    row["creators"] = json.dumps(
        mint_info.get("creators") or [],
        separators=(",", ":")
    )

    row["mint_authority"] = mint_info.get("mint_authority")

    row["freeze_authority"] = mint_info.get("freeze_authority")

    print(f" Collecting historical liquidity for {token_mint}, experiment={experiment_id}.")

    # Use a forward snapshot at creation and a backward snapshot at the cutoff. This brackets the observation window without consuming post-cutoff data.
    (liquidity_at_creation, creation_snapshot_time, creation_liquidity_status) = get_birdeye_liquidity_snapshot(
                                                                                    token_mint=token_mint,
                                                                                    target_time=creation_time,
                                                                                    direction="forward"
                                                                                )

    (liquidity_at_end, end_snapshot_time, end_liquidity_status) = get_birdeye_liquidity_snapshot(
                                                                    token_mint=token_mint,
                                                                    target_time=observation_end,
                                                                    direction="back"
                                                                )

    if creation_liquidity_status == "temporary_error" or end_liquidity_status == "temporary_error":
        row["_temporary_error"] = True

        status_parts.append("liquidity_temporary_error")

        row["feature_status"] = ",".join(status_parts)
        return row

    # The first liquidity snapshot must be at or shortly after token creation and must remain inside the experiment window.
    if creation_snapshot_time is not None:
        creation_snapshot_delay = creation_snapshot_time - creation_time

        if creation_snapshot_delay < 0 or creation_snapshot_delay > LIQUIDITY_SNAPSHOT_TOLERANCE_SECONDS or creation_snapshot_time > observation_end:
            liquidity_at_creation = None
            creation_liquidity_status = "creation_snapshot_too_far"

    # The endpoint snapshot must be at or shortly before the observation boundary and must not predate token creation.
    if end_snapshot_time is not None:
        end_snapshot_delay =  observation_end - end_snapshot_time

        if end_snapshot_delay < 0 or end_snapshot_delay > LIQUIDITY_SNAPSHOT_TOLERANCE_SECONDS or end_snapshot_time < creation_time:
            liquidity_at_end = None
            end_liquidity_status = "observation_end_snapshot_too_far"

    # Unavailable or temporally distant snapshots remain None rather than being replaced with current-state liquidity or another leakage-prone estimate.
    row["liquidity_at_creation"] = liquidity_at_creation

    row["liquidity_at_observation_end"] = liquidity_at_end

    status_parts.append(f"liquidity_creation_{creation_liquidity_status}")

    status_parts.append(f"liquidity_observation_end_{end_liquidity_status}")

    print(f" Collecting trades for {token_mint}, experiment={experiment_id}.")

    # Trade features use only swaps whose timestamps fall inside this experiment's creation-to-cutoff interval.
    (trade_features, trade_error, trade_status) = compute_trade_features(
                                                    token_mint=token_mint,
                                                    observation_start=creation_time,
                                                    observation_end=observation_end
                                                )

    if trade_error:
        row["_temporary_error"] = True

        status_parts.append(f"trade_{trade_status}")

        row["feature_status"] = ",".join(status_parts)
        return row

    # Copy only recognized feature names into the row scaffold.
    for key, value in trade_features.items():
        if key in row:
            row[key] = value

    status_parts.append(trade_status)

    # Preserve an internal completeness audit even though its status columns are not included in the exported CSV schema.
    (required_features_complete, required_feature_status) = evaluate_required_features(row)

    required_status_parts = []

    if not required_features_complete:
        required_status_parts.append(required_feature_status)

    if trade_status != "ok":
        required_status_parts.append(trade_status)

    row["required_feature_status"] = "complete" if not required_status_parts else "|".join(required_status_parts)

    row["feature_status"] = ",".join(part for part in status_parts if clean_string(part) != "")

    return row

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Coordinate the complete resumable node-feature collection workflow.
    """

    print("=" * 70)
    print("STARTED NODES COLLECTION")
    print("=" * 70)

    # Validate mandatory external dependencies before loading or modifying any dataset files.
    if not BIRDEYE_API_KEY:
        raise RuntimeError("BIRDEYE_API_KEY is empty.")

    rpc_health = check_rpc_health()
    if rpc_health:
        print("RPC is healthy.")
    else:
        raise RuntimeError("RPC is not healthy or not reachable.")

    # Load only valid supervised-label rows and preserve one entry for each independent token-experiment pair.
    label_map = load_label_map(LABEL_FILE)

    if len(label_map) == 0:
        raise RuntimeError(f"Label file {LABEL_FILE} contains no usable completed labels. Run build_rule_based_labels.py to completion before starting the historical node collector.")

    df = load_label_nodes(label_map)

    # Convert the validated DataFrame into an immutable processing queue with a fixed field order for the main loop.
    nodes = list(
        df[
            [
                "lp_address",
                "token_mint",
                "experiment_id",
                "rug_pull_label",
                "label_status",
                "token_creation_time",
                "observation_window_seconds",
                "observation_end_time",
                "liquidity_event_time",
                "label_available_time"
            ]
        ].itertuples(
            index=False,
            name=None
        )
    )

    print(f"Labels loaded: {len(label_map)}")
    print(f"Token-experiment nodes to consider: {len(nodes)}")

    # Validate the current output schema before resume information is loaded.
    ensure_output_file_exists()

    # Completed CSV rows are excluded from the new queue so interrupted runs continue without repeating successful API requests.
    processed_nodes = load_processed_nodes()

    # Keep only nodes whose unique pool-mint-experiment key does not already exist in the output dataset.
    remaining_nodes = [ 
        (
            lp_address,
            token_mint,
            experiment_id,
            rug_pull_label,
            label_status,
            label_creation_time,
            observation_window_seconds,
            observation_end_time,
            liquidity_event_time,
            label_available_time
        )
        for (
            lp_address,
            token_mint,
            experiment_id,
            rug_pull_label,
            label_status,
            label_creation_time,
            observation_window_seconds,
            observation_end_time,
            liquidity_event_time,
            label_available_time
        ) in nodes
        if make_node_key(
            lp_address=lp_address,
            token_mint=token_mint,
            experiment_id=experiment_id
        ) not in processed_nodes
    ]

    print(f"Already processed nodes: {len(processed_nodes)}")
    print(f"Remaining nodes: {len(remaining_nodes)}")
    print("-" * 70)

    saved_nodes = 0
    skipped_temporal = 0
    skipped_temporary = 0

    # Process each experiment independently because the same token has different cutoffs and therefore different historical trade and liquidity features.
    for (
        lp_address,
        token_mint,
        experiment_id,
        rug_pull_label,
        label_status,
        label_creation_time,
        observation_window_seconds,
        observation_end_time,
        liquidity_event_time,
        label_available_time
    ) in tqdm(
        remaining_nodes,
        desc="Processing token-experiment nodes"
    ):
        node_key = make_node_key(
            lp_address=lp_address,
            token_mint=token_mint,
            experiment_id=experiment_id
        )

        try:
            print("")
            print("-" * 70)
            print(f" Processing node: {node_key}")

            row = collect_node_features(
                lp_address=lp_address,
                token_mint=token_mint,
                experiment_id=experiment_id,
                rug_pull_label=rug_pull_label,
                label_status=label_status,
                label_creation_time=label_creation_time,
                label_observation_window_seconds=observation_window_seconds,
                label_observation_end_time=observation_end_time,
                liquidity_event_time=liquidity_event_time,
                label_available_time=label_available_time
            )

            # Temporary failures are deliberately not saved or marked as processed, allowing a future run to retry the same node.
            if row.get("_temporary_error"):
                skipped_temporary += 1
                print(f" Node skipped because a temporary RPC/API error occurred: {node_key}")
                continue

            # Temporal violations are excluded permanently because they would expose label-defining information inside the feature observation window.
            if row.get("_temporal_skip"):
                skipped_temporal += 1
                print(f" Node skipped by temporal filter: {node_key}; status={row.get('temporal_validation_status') or row.get('feature_status')}")
                continue

            # Project the internal row onto the public output schema. Diagnostic fields and execution flags are removed at this boundary.
            clean_row = { key: row.get(key) for key in OUTPUT_COLUMNS }

            # Write the feature row before updating progress so a node is never treated as complete without a corresponding CSV record.
            saved = append_row_to_output(clean_row)

            if not saved:
                print(f" Node not marked as processed because saving failed: {node_key}")
                continue

            # Record successful completion only after the row has been persisted.
            marked = mark_node_as_processed(
                lp_address=lp_address,
                token_mint=token_mint,
                experiment_id=experiment_id
            )

            if not marked:
                print(f" Node data was saved, but progress was not updated: {node_key}")
                continue

            saved_nodes += 1
            print(" Node done.")

        except Exception as e:
            print(f" Error while processing node {node_key}: {e}")
            continue

    print("-" * 70)
    print(f"Saved nodes: {saved_nodes}")
    print(f"Skipped temporal filter: {skipped_temporal}")
    print(f"Skipped temporary RPC/API errors: {skipped_temporary}")

    print("=" * 70)
    print("FINISHED NODES COLLECTION")
    print("=" * 70)


if __name__ == "__main__":
    main()
