""" 
This script builds rule-based rug-pull labels for tokens in
the 2024 dataset. Each token is evaluated under three different
observation windows, while the outcome is checked during the
first 24 hours after token creation.

The final binary label uses three required factors:
1. a liquidity drop greater than 80% from the running token-level liquidity peak;
2. a price drop greater than 90% after the liquidity-drop event;
3. confirmation that the liquidity loss remains persistent during a fixed post-event period.

Final rule:
LiquidityDrop AND PriceDrop AND PersistentLiquidityDepletion

Tokens whose liquidity event occurs inside the observation
window, or whose required historical coverage is unavailable,
are not assigned a normal binary label.

The post-event confirmation interval is used only for labeling
and must never be included in model features or graph edges.

Label-status policy:
    - VALID rows receive a binary label and may be used for supervised learning.
    - IN_WINDOW_EVENT rows are excluded because the target event is already visible inside the corresponding observation window.
    - UNKNOWN rows are retained for audit but do not receive a binary label.
    - Temporary API/RPC failures are not saved, so a later run can retry them.

Liquidity and price histories are fetched at token level and reused across
the three experiments. Only the experiment-specific calculations are repeated for q1, q2, and q3.
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGS
# ============================================================

# A reproducible random subset is used for the current run.
SAMPLE_FRACTION = 0.0973
RANDOM_SEED = 42

# Delay and retry settings reduce rate-limit pressure on the APIs.
SLEEP_BETWEEN_API_CALLS = 0.5
MAX_RETRIES = 5

# Labels detect liquidity and price collapse during the first 24 hours after token creation.
OUTCOME_WINDOW_SECONDS = 24 * 60 * 60

# One-minute histories provide the highest available temporal resolution for locating the liquidity and price-drop events.
LIQUIDITY_RESOLUTION = "1m"
PRICE_RESOLUTION = "1m"

# C is the fixed post-event interval used to confirm that the liquidity loss persists. Use 15, 30, or 60 for sensitivity runs.
PERSISTENCE_CONFIRMATION_MINUTES = 30
PERSISTENCE_CONFIRMATION_SECONDS = PERSISTENCE_CONFIRMATION_MINUTES * 60

# The observation-window coefficients are taken from Table IV of the "From Hype to Collapse" paper. Each coefficient is multiplied by one day to obtain the experiment duration.
OBSERVATION_WINDOWS = [
    {
        "experiment_id": "q1",
        "fraction_of_day": 0.008681  # Approximately 12.5 minutes
    },
    {
        "experiment_id": "q2",
        "fraction_of_day": 0.024618  # Approximately 35.5 minutes
    },
    {
        "experiment_id": "q3",
        "fraction_of_day": 0.064641  # Approximately 93.1 minutes
    }
]

INPUT_FILE = "2024.csv"

OUTPUT_FILE = "rule_based_labels.csv"
PROGRESS_FILE = "processed_rule_labels.txt"

BIRDEYE_API_KEY = os.environ.get("BIRDEYE_API_KEY", "").strip()  # To set: $env:BIRDEYE_API_KEY="***"
BIRDEYE_BASE_URL = "https://public-api.birdeye.so"

RPC_URL = "https://api.mainnet-beta.solana.com"

# One output row is generated for each token-experiment pair.
# VALID rows receive a binary rug_pull_label, while in-window and unknown rows keep the label empty and record the reason in label_status.
OUTPUT_COLUMNS = [
    "lp_address",
    "token_mint",
    "token_creation_time",
    "experiment_id",
    "observation_window_seconds",
    "observation_end_time",
    "outcome_end_time",

    "liquidity_peak_before_drop",
    "liquidity_at_drop",
    "liquidity_drop_ratio",
    "liquidity_drop_flag",
    "liquidity_drop_factor_time",

    "price_peak_before_drop",
    "price_peak_time",
    "price_min_after_drop",
    "price_min_time",
    "price_drop_ratio",
    "price_drop_flag",
    "price_drop_factor_time",

    "persistence_confirmation_seconds",
    "persistence_end_time",
    "persistence_liquidity_threshold",
    "persistence_low_liquidity_seconds",
    "persistence_ratio",
    "liquidity_at_persistence_end",
    "persistent_liquidity_depletion_flag",
    "persistent_liquidity_depletion_factor_time",

    "label_available_time",
    "label_status",
    "rug_pull_label"
]

# Token-level liquidity and price histories are fetched once and reused across all eligible observation-window experiments.
LIQUIDITY_HISTORY_CACHE = {}
PRICE_OHLCV_CACHE = {}
TOKEN_CREATION_TIME_CACHE = {}

# ============================================================
# RULE THRESHOLDS
# ============================================================

# A candidate liquidity rug requires a drop greater than 80% from the running liquidity peak.
LIQUIDITY_DROP_THRESHOLD = 0.80

# Price must fall by more than 90% after the detected liquidity-drop event.
PRICE_DROP_THRESHOLD = 0.90

# Persistence requires liquidity to remain at or below 20% of the pre-event peak for at least 80% of the confirmation period.
PERSISTENCE_LOW_LIQUIDITY_RATIO = 0.20
PERSISTENCE_REQUIRED_RATIO = 0.80

# Birdeye returns at most 100 liquidity-history records per request, so the 24-hour one-minute history is collected through forward pagination.
BIRDEYE_LIQUIDITY_PAGE_LIMIT = 100

# ============================================================
# FILE / PROGRESS HELPERS
# ============================================================

def load_and_sample_input_file(input_file, required_columns):
    """
    Load and clean the 2024 input file used for labeling.
    """

    try:
        df = pd.read_csv(input_file, dtype=str)

    except Exception as e:
        raise RuntimeError(f"Could not read input file {input_file}: {e}")

    missing_columns = [ column for column in required_columns if column not in df.columns ]

    if missing_columns:
        raise ValueError(f"Input file {input_file} is missing columns: {missing_columns}")

    df = df.copy()

    df["LIQUIDITY_POOL_ADDRESS"] = (
        df["LIQUIDITY_POOL_ADDRESS"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["MINT"] = (
        df["MINT"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Rows without a valid token mint or pool address cannot be labeled and are removed from the input cohort.
    df = df[ (df["LIQUIDITY_POOL_ADDRESS"] != "") & (df["MINT"] != "") ].copy()

    print(f"2024 cleaned rows loaded={len(df)}")

    return df


def deduplicate_by_mint(df):
    """
    Keep one labeling input row for each unique token mint.
    """

    df = df.copy()

    # Birdeye histories are requested at token level, so each mint is processed only once even if it appears with multiple pools.
    df = (
        df.sort_values(
            by=[
                "MINT",
                "LIQUIDITY_POOL_ADDRESS"
            ],
            kind="stable"
        ).drop_duplicates(
            subset=["MINT"],
            keep="first"
        ).reset_index(drop=True)
    )

    return df


def make_label_key(token_mint, experiment_id):
    """
    Progress is tracked separately for each token and experiment because the same token produces three independent label rows.
    """

    return f"{token_mint}|{experiment_id}"


def load_processed_labels():
    """
    The progress file allows an interrupted run to continue without repeating already completed token-experiment pairs.
    """

    if not os.path.exists(PROGRESS_FILE):
        return set()

    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set( line.strip() for line in f if line.strip() and not line.lstrip().startswith("#") )

    except Exception as e:
        print(f"Could not read progress file: {e}")
        return set()


def mark_label_as_processed(token_mint, experiment_id):
    label_key = make_label_key(
        token_mint=token_mint,
        experiment_id=experiment_id
    )

    try:
        with open(PROGRESS_FILE, "a", encoding="utf-8") as file:
            file.write(label_key + "\n")
            file.flush()
            os.fsync(file.fileno())

        return True

    except Exception as e:
        print(f"Could not update progress file for label {label_key}: {e}")
        return False


def ensure_output_file_exists():
    try:
        if not os.path.exists(OUTPUT_FILE):
            pd.DataFrame(
                columns=OUTPUT_COLUMNS
            ).to_csv(
                OUTPUT_FILE,
                index=False
            )

            return

        existing_columns = list(
            pd.read_csv(
                OUTPUT_FILE,
                nrows=0
            ).columns
        )

        if existing_columns != OUTPUT_COLUMNS:
            raise RuntimeError(
                f"Output file {OUTPUT_FILE} has an old or incompatible header. Delete it before running this version.\nExpected: {OUTPUT_COLUMNS}\nFound: {existing_columns}")

    except RuntimeError:
        raise

    except Exception as e:
        raise RuntimeError(f"Could not validate output file: {e}")


def append_row_to_output(row):
    try:
        df_row = pd.DataFrame(
            [row], 
            columns=OUTPUT_COLUMNS
        )

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
# GENERIC HELPERS
# ============================================================

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

    numeric_value = safe_float(value)

    if numeric_value is None:
        return None

    try:
        return int(numeric_value)
    
    except Exception:
        return None


def clean_string(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
        
    except Exception:
        pass

    return str(value).strip()


def safe_get_from_candidates(data, candidates):
    """
    Return the first available value from multiple possible field aliases.
    Birdeye response field names may differ between endpoints or API versions, so parsers accept a prioritized list of equivalent keys.
    """
        
    if not isinstance(data, dict):
        return None

    for key in candidates:
        value = data.get(key)
        if value is not None:
            return value

    return None

# ============================================================
# HTTP / BIRDEYE HELPERS
# ============================================================

def http_get_json(url, headers=None, params=None, service_name="HTTP", allow_forbidden=False):
    """
    Execute a GET request with centralized retry and error handling.
    Rate limits and transient transport failures are retried with exponential backoff. Authentication failures are fatal because continuing without valid Birdeye access would produce systematically incomplete labels.
    A forbidden response may be tolerated only when allow_forbidden=True and the caller has a complete alternative source, such as Solana RPC for creation-time discovery.
    """

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=20
            )

            if response.status_code == 429:
                wait_time = 2 ** attempt
                print(f" {service_name} rate limited. Sleeping {wait_time}s.")
                time.sleep(wait_time)
                continue

            # Invalid API requests are not saved as permanent labels.
            if response.status_code in [400, 422]:
                print(f" {service_name} bad request with status={response.status_code}, url={url}")
                print(f" Response body: {response.text[:300]}")

                return None, True

            if response.status_code == 401:
                raise SystemExit(f"\nFATAL: {service_name} returned 401. The Birdeye API key is missing or invalid.")

            if response.status_code == 403:
                if allow_forbidden:
                    # This endpoint may not be included in the current Birdeye plan, so the caller may use another source.
                    return None, False

                raise SystemExit(f"\nFATAL: {service_name} returned 403.\nURL: {url}\nResponse: {response.text[:300]}")

            # A missing endpoint result is treated as unavailable data rather than a transport failure; the caller determines whether it implies UNKNOWN or whether a fallback is available.
            if response.status_code == 404:
                return None, False

            response.raise_for_status()
            return response.json(), False

        except Exception as e:
            wait_time = 2 ** attempt
            print(f" {service_name} failed attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            time.sleep(wait_time)

    print(f" {service_name} failed after retries, url={url}")
    return None, True


def get_birdeye_headers():
    """
    Construct the common Birdeye headers for Solana requests and attach the API key only when it has been configured.
    """

    headers = {
        "accept": "application/json",
        "x-chain": "solana"
    }

    if BIRDEYE_API_KEY:
        headers["X-API-KEY"] = BIRDEYE_API_KEY

    return headers


def birdeye_get(path, params=None, allow_forbidden=False):
    """
    Call one Birdeye endpoint using shared authentication, retry handling, and a pacing delay that reduces rate-limit pressure.
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


def rpc_call(method, params):
    """
    Execute a Solana JSON-RPC request with exponential backoff.
    The returned error flag distinguishes an RPC failure from a legitimate empty result. RPC calls are used only when the preferred Birdeye creation-time source is unavailable.
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

            # Back off exponentially instead of immediately repeating a rate-limited request against the public Solana RPC endpoint.
            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()
            data = response.json()

            if "error" in data:
                error_text = str(data["error"])

                if "rate" in error_text.lower():
                    time.sleep(2 ** attempt)
                    continue

                print(f" RPC error in {method}: {error_text}")
                return None, True

            return data.get("result"), False

        except Exception:
            time.sleep(2 ** attempt)

    return None, True


def get_oldest_signature_time(address):
    """
    Estimate token creation time from the oldest available on-chain signature when Birdeye does not provide a creation timestamp.
    Solana returns signatures newest-first. Pagination follows the 'before' cursor backward until the oldest available signature is reached.
    The signature's blockTime is preferred. If it is absent, the complete transaction is fetched and its blockTime is used. This is a fallback estimate based on the earliest available activity for the mint address.
    """

    oldest_item = None
    before = None

    while True:
        options = {"limit": 1000}

        if before:
            options["before"] = before

        result, had_error = rpc_call(
            "getSignaturesForAddress",
            [address, options]
        )

        if had_error:
            return None, True

        if not result:
            break

        # The final item of a newest-first page is the oldest signature observed on that page and becomes the cursor for the next backward request.
        oldest_item = result[-1]
        before = oldest_item.get("signature")

        if len(result) < 1000 or not before:
            break

    if not oldest_item:
        return None, False

    block_time = safe_int(oldest_item.get("blockTime"))

    if block_time is not None:
        return block_time, False

    signature = clean_string(oldest_item.get("signature"))

    if not signature:
        return None, False

    # Fetch the full transaction only when the signature-list record itself does not contain a usable blockTime.
    transaction, had_error = rpc_call(
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

    if had_error:
        return None, True

    return (
        safe_int(transaction.get("blockTime"))
        if isinstance(transaction, dict)
        else None
    ), False


def get_token_creation_time(token_mint):
    """
    Determine the temporal anchor used by every experiment of one token.
    The dedicated Birdeye creation-info endpoint is queried first. When it does not return a timestamp, the oldest available Solana signature is used as a fallback estimate.
    The result is cached because the q1, q2, and q3 experiments share the same creation time. This timestamp defines both observation-window starts and the beginning of the fixed 24-hour outcome horizon.
    """

    # Reuse the timestamp already obtained for another experiment of this mint.
    if token_mint in TOKEN_CREATION_TIME_CACHE:
        return TOKEN_CREATION_TIME_CACHE[token_mint], False

    # Prefer Birdeye because it may provide creation time directly without scanning a potentially large on-chain signature history.
    data, had_error = birdeye_get(
        "/defi/token_creation_info",
        params={"address": token_mint},
        allow_forbidden=True
    )

    if had_error:
        return None, True

    creation_data = data.get("data") if isinstance(data, dict) else None

    creation_time = None

    if isinstance(creation_data, dict):
        creation_time = safe_int(
            safe_get_from_candidates(
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

    # Fall back to the oldest on-chain signature when Birdeye does not provide the creation timestamp.
    if creation_time is None:
        creation_time, had_error = get_oldest_signature_time(token_mint)

        if had_error:
            return None, True

    TOKEN_CREATION_TIME_CACHE[token_mint] = creation_time
    return creation_time, False

# ============================================================
# BIRDEYE LIQUIDITY AND PRICE HISTORY HELPERS
# ============================================================

def extract_history_items(data):
    """
    Extract the record list from a Birdeye response.
    """

    if not isinstance(data, dict):
        return []

    inner = data.get("data")

    if isinstance(inner, list):
        return inner

    if isinstance(inner, dict):
        for key in ["items", "list", "records", "data"]:
            value = inner.get(key)

            if isinstance(value, list):
                return value

    for key in ["items", "list", "records", "data"]:
        value = data.get(key)

        if isinstance(value, list):
            return value

    return []


def extract_history_timestamp(item):
    """
    Extract a Unix timestamp from a history record.
    """

    if not isinstance(item, dict):
        return None

    return safe_int(
        safe_get_from_candidates(
            item,
            [
                "unix_time",
                "unixTime",
                "timestamp",
                "time",
                "block_unix_time",
                "blockUnixTime"
            ]
        )
    )


def extract_total_liquidity(item):
    """
    Extract Birdeye's token-level total liquidity value.
    """

    if not isinstance(item, dict):
        return None

    direct_value = safe_get_from_candidates(
        item,
        [
            "total_liquidity",
            "totalLiquidity",
            "total_liquidity_usd",
            "totalLiquidityUsd",
            "liquidity",
            "liquidity_usd",
            "liquidityUsd"
        ]
    )

    direct_value = safe_float(direct_value)

    if direct_value is not None:
        return direct_value

    liquidity_object = item.get("liquidity")

    if isinstance(liquidity_object, dict):
        nested_value = safe_get_from_candidates(
            liquidity_object,
            [
                "total",
                "total_liquidity",
                "totalLiquidity",
                "usd",
                "value"
            ]
        )

        return safe_float(nested_value)

    return None


def normalize_liquidity_items(items, start_time, end_time):
    """
    Convert raw liquidity records into chronological timestamp/value pairs.
    Records with invalid timestamps, negative liquidity, or timestamps outside the inclusive [start_time, end_time] interval are discarded.
    Duplicate timestamps are collapsed to one final value before sorting.
    The inclusive endpoint allows the explicitly requested snapshot at the 24-hour boundary to support coverage and boundary checks.
    """

    # Invalid values and records outside the requested interval are discarded, and duplicate timestamps are reduced to one value.
    points_by_time = {}

    for item in items:
        item_time = extract_history_timestamp(item)
        liquidity = extract_total_liquidity(item)

        if item_time is None or liquidity is None or liquidity < 0 or item_time < start_time or item_time > end_time:
            continue

        # When duplicate timestamps occur, the last returned value replaces the earlier value so only one liquidity state represents each minute.
        points_by_time[item_time] = liquidity

    return sorted(
        points_by_time.items(),
        key=lambda value: value[0]
    )


def get_birdeye_liquidity_history(token_mint, start_time, end_time):
    """
    Retrieve all available one-minute token-level liquidity observations covering the first 24-hour outcome horizon.
    Birdeye returns at most BIRDEYE_LIQUIDITY_PAGE_LIMIT records per request.
    Pages are therefore requested forward from creation time until the endpoint is reached, the API returns a short/empty page, or the calculated safety limit is reached.
    A final backward request at end_time adds the closest known state at or before the 24-hour boundary. The function collects all available records and it does not synthesize missing one-minute observations.
    """

    cache_key = f"{token_mint}|{start_time}|{end_time}|{LIQUIDITY_RESOLUTION}"

    # Reuse the same complete first-day history across all three experiments.
    if cache_key in LIQUIDITY_HISTORY_CACHE:
        return LIQUIDITY_HISTORY_CACHE[cache_key]

    # All available liquidity snapshots are collected before any experiment-specific drop calculation is performed.
    all_items = []
    anchor_time = start_time
    previous_latest_time = None
    pagination_complete = False

    # A full 24-hour one-minute series contains approximately 1,440 points. The safety margin permits partially filled or overlapping API pages while still detecting stalled or unexpectedly unbounded pagination.
    maximum_pages = ((end_time - start_time) // 60 // BIRDEYE_LIQUIDITY_PAGE_LIMIT) + 5

    for _ in range(maximum_pages):
        data, had_error = birdeye_get(
            "/defi/v3/liquidity/history/token",
            params={
                "address": token_mint,
                "resolution": LIQUIDITY_RESOLUTION,
                "time": anchor_time,
                "direction": "forward",
                "count": BIRDEYE_LIQUIDITY_PAGE_LIMIT
            }
        )

        if had_error:
            return (
                [],
                True,
                False,
                "liquidity_api_error"
            )

        if isinstance(data, dict) and data.get("success") is False:
            result = (
                [],
                False,
                False,
                "liquidity_history_unavailable"
            )

            LIQUIDITY_HISTORY_CACHE[cache_key] = result
            return result

        page_items = extract_history_items(data)

        # No returned records means that no additional liquidity candles are available after this cursor.
        if not page_items:
            pagination_complete = True
            break

        page_times = [ extract_history_timestamp(item) for item in page_items ]

        page_times = [ item_time for item_time in page_times if item_time is not None ]

        if not page_times:
            # An unexpected response structure should not be saved as a permanent UNKNOWN label.
            return (
                [],
                True,
                False,
                "liquidity_response_shape_error"
            )

        all_items.extend(page_items)

        latest_page_time = max(page_times)

        # The current page has reached or crossed the end of the first-day outcome window.
        if latest_page_time >= end_time:
            pagination_complete = True
            break

        # Pagination must advance strictly forward. A repeated or older final timestamp indicates a stalled cursor and is treated as retryable.
        if previous_latest_time is not None and latest_page_time <= previous_latest_time:
            return (
                [],
                True,
                False,
                "liquidity_pagination_stalled"
            )

        previous_latest_time = latest_page_time

        # A short page means Birdeye has no more available liquidity candles after the returned records.
        if len(page_items) < BIRDEYE_LIQUIDITY_PAGE_LIMIT:
            pagination_complete = True
            break

        # Continue from the next one-minute candle.
        anchor_time = latest_page_time + 60

    if not pagination_complete:
        result = (
            [],
            False,
            False,
            "liquidity_pagination_limit_reached"
        )

        # Complete means pagination terminated normally and produced at least two usable observations; it does not mean that every minute has a candle.
        LIQUIDITY_HISTORY_CACHE[cache_key] = result
        return result

    # Add the closest snapshot at or before the 24-hour boundary so the last known liquidity state is included in the history.
    end_data, end_error = birdeye_get(
        "/defi/v3/liquidity/history/token",
        params={
            "address": token_mint,
            "resolution": LIQUIDITY_RESOLUTION,
            "time": end_time,
            "direction": "back",
            "count": 1
        }
    )

    if end_error:
        return (
            [],
            True,
            False,
            "liquidity_api_error"
        )

    if not (isinstance(end_data, dict) and end_data.get("success") is False):
        all_items.extend(extract_history_items(end_data))

    points = normalize_liquidity_items(
        items=all_items,
        start_time=start_time,
        end_time=end_time
    )

    if not points:
        result = (
            [],
            False,
            False,
            "missing_liquidity_history"
        )

        LIQUIDITY_HISTORY_CACHE[cache_key] = result
        return result

    if len(points) < 2:
        result = (
            [],
            False,
            False,
            "insufficient_liquidity_history"
        )

        LIQUIDITY_HISTORY_CACHE[cache_key] = result
        return result

    # Successful pagination means all available liquidity candles for the first-day interval were collected.
    result = (
        points,
        False,
        True,
        None
    )

    LIQUIDITY_HISTORY_CACHE[cache_key] = result
    return result


def extract_ohlcv_items(data):
    """
    Extract OHLCV candles from a Birdeye response.
    """

    return extract_history_items(data)


def extract_ohlcv_high(item):
    """
    Extract the highest recorded price from one OHLCV candle.
    """
        
    if not isinstance(item, dict):
        return None

    return safe_float(
        safe_get_from_candidates(
            item,
            [
                "h",
                "high",
                "high_price",
                "highPrice"
            ]
        )
    )


def extract_ohlcv_low(item):
    """
    Extract the lowest recorded price from one OHLCV candle.
    """
        
    if not isinstance(item, dict):
        return None

    return safe_float(
        safe_get_from_candidates(
            item,
            [
                "l",
                "low",
                "low_price",
                "lowPrice"
            ]
        )
    )


def get_birdeye_price_ohlcv(token_mint, start_time, end_time):
    """
    Retrieve one-minute USD OHLCV candles for the first 24-hour outcome horizon.
    Price history is fetched at most once per token and is requested only when at least one experiment has a valid liquidity-drop and persistence result requiring Factor 2.
    Padding is disabled so the API does not create synthetic candles.
    Recorded outliers are retained because the labeling rule deliberately uses the raw pre-event maximum and post-event minimum.
    The normalized candle interval is [start_time, end_time).
    """

    cache_key = (f"{token_mint}|{start_time}|{end_time}|{PRICE_RESOLUTION}")

    # Every eligible experiment reuses the same first-day price history.
    if cache_key in PRICE_OHLCV_CACHE:
        return PRICE_OHLCV_CACHE[cache_key]

    # One-minute USD OHLCV candles are collected for the entire first-day outcome window. Padding is disabled so synthetic candles are not introduced into the peak/minimum calculation.
    params = {
        "address": token_mint,
        "type": PRICE_RESOLUTION,
        "currency": "usd",
        "time_from": start_time,
        "time_to": end_time,
        "mode": "range",
        "padding": "false",
        "outlier": "true"       # Keep recorded outliers for the raw max/min rule.
    }

    data, had_error = birdeye_get(
        "/defi/v3/ohlcv",
        params=params
    )

    if had_error:
        return (
            [],
            True,
            False,
            "price_api_error"
        )

    if (isinstance(data, dict) and data.get("success") is False):
        result = (
            [],
            False,
            False,
            "price_history_unavailable"
        )

        PRICE_OHLCV_CACHE[cache_key] = result
        return result

    # Duplicate timestamps are collapsed so every minute contributes only one OHLCV candle to the subsequent price-drop calculation.
    points_by_time = {}

    for item in extract_ohlcv_items(data):
        item_time = extract_history_timestamp(item)
        high_price = extract_ohlcv_high(item)
        low_price = extract_ohlcv_low(item)

        if item_time is None or high_price is None or low_price is None or high_price <= 0 or low_price <= 0 or item_time < start_time or item_time >= end_time:
            continue

        points_by_time[item_time] = {
            "time": item_time,
            "high": high_price,
            "low": low_price
        }

    price_points = [ points_by_time[item_time] for item_time in sorted(points_by_time) ]

    if not price_points:
        result = (
            [],
            False,
            False,
            "missing_price_history"
        )

        PRICE_OHLCV_CACHE[cache_key] = result
        return result

    result = (
        price_points,
        False,
        True,
        None
    )

    PRICE_OHLCV_CACHE[cache_key] = result
    return result

# ============================================================
# LABEL FACTOR 1: LIQUIDITY DROP
# ============================================================

def compute_liquidity_drop_for_experiment(liquidity_points, token_creation_time, observation_end_time, outcome_end_time):
    """
    Evaluate LiquidityDrop for one observation-window experiment.
    Liquidity observations are scanned chronologically while maintaining the highest strictly preceding liquidity value as the running peak.
    The first loss strictly greater than 80% from that peak is selected as the candidate liquidity-rug event.
    The same token history can produce different experiment statuses because q1, q2, and q3 have different observation cutoffs:
        - IN_WINDOW_EVENT: the event is already visible in model input;
        - VALID: the event occurs safely after the cutoff;
        - UNKNOWN_*: one-minute resolution or an invalid baseline prevents a defensible label.
    If no qualifying event is found, the factor is False and the largest post-observation drop is retained for audit.
    """

    # For each time t:
    # L_peak(t) = max liquidity observed before t
    # LiquidityDrop(t) = 1 - liquidity(t) / L_peak(t)
    #
    # The first time LiquidityDrop(t) exceeds 0.80 is treated as the candidate liquidity-rug time t_rug.

    result = {
        "liquidity_peak_before_drop": None,
        "liquidity_at_drop": None,
        "liquidity_drop_ratio": None,
        "liquidity_drop_flag": None,
        "liquidity_drop_factor_time": None,
        "_liquidity_status": None
    }

    # The running peak is updated chronologically and always represents liquidity observed strictly before the next point.
    running_peak = None
    maximum_observed_drop = 0.0

    # Liquidity history is scanned once in chronological order while maintaining the running pre-event peak required by the LiquidityDrop definition.
    for point_time, liquidity in liquidity_points:
        if point_time < token_creation_time:
            continue

        # The outcome window is [creation, creation + 24 hours).
        if point_time > outcome_end_time:
            break

        if liquidity is None or liquidity < 0:
            continue

        # The first positive observation establishes the initial baseline but cannot itself be compared with a preceding peak.
        if running_peak is None:
            if liquidity > 0:
                running_peak = liquidity
            continue

        if running_peak > 0:
            # Measure the current loss relative to the highest liquidity level observed before this timestamp.
            drop_ratio = max(0.0, 1.0 - (liquidity / running_peak))

            # For negative samples, retain the largest post-observation drop even when it does not cross the 80% rug threshold.
            if observation_end_time < point_time < outcome_end_time:
                maximum_observed_drop = max(maximum_observed_drop, drop_ratio)

            # Stop at the first threshold-crossing event, because t_rug is defined as the earliest liquidity drop greater than 80%.
            if drop_ratio > LIQUIDITY_DROP_THRESHOLD:
                result.update(
                    {
                        "liquidity_peak_before_drop": running_peak,
                        "liquidity_at_drop": liquidity,
                        "liquidity_drop_ratio": drop_ratio,
                        "liquidity_drop_flag": True,
                        "liquidity_drop_factor_time": point_time
                    }
                )

                # One-minute snapshots do not reveal the exact transaction second. Events touching an observation or outcome boundary are therefore marked ambiguous instead of forcing a label.
                if point_time == outcome_end_time:
                    # A drop first visible exactly at the 24-hour boundary may have occurred during the preceding minute.
                    result["_liquidity_status"] = "UNKNOWN_OUTCOME_BOUNDARY_AMBIGUOUS"

                # A rug already visible inside the observation window would leak the target event into model input and is excluded.
                elif point_time <= observation_end_time:
                    result["_liquidity_status"] = "IN_WINDOW_EVENT"

                elif point_time - 60 <= observation_end_time:
                    # With one-minute snapshots, the exact event may have occurred just before or just after the boundary.
                    result["_liquidity_status"] = "UNKNOWN_OBSERVATION_BOUNDARY_AMBIGUOUS"

                else:
                    result["_liquidity_status"] = "VALID"

                return result

        # Update the peak only after evaluating the current point, ensuring the denominator always represents liquidity observed strictly before it.
        if liquidity > running_peak:
            running_peak = liquidity

    if running_peak is None or running_peak <= 0:
        result["_liquidity_status"] = "UNKNOWN_INVALID_LIQUIDITY_BASELINE"

        return result

    # No qualifying liquidity event was found after the observation window, so this factor is negative and no t_rug is defined.
    result.update(
        {
            "liquidity_drop_ratio": maximum_observed_drop,
            "liquidity_drop_flag": False,
            "_liquidity_status": "VALID"
        }
    )

    return result


# ============================================================
# LABEL FACTOR 2: PRICE DROP
# ============================================================

def compute_price_drop(price_points, token_creation_time, liquidity_drop_time, outcome_end_time):
    """
    Evaluate PriceDrop relative to the detected liquidity event.
    P_peak is the highest price from token creation through the final fully completed candle before t_rug.
    P_min is the lowest price among complete candles beginning at the first full minute at or after t_rug and ending before the 24-hour outcome boundary.
    PriceDrop is True only when the relative decline is strictly greater than 90%. price_drop_factor_time records the timestamp of the selected post-event minimum, not the first instant at which the 90% threshold may have been crossed.
    """

    # P_peak is the maximum price from token creation until before t_rug. P_min is the minimum price from t_rug until the end of the 24-hour outcome window.
    #
    # PriceDrop = 1 - P_min / P_peak

    result = {
        "price_peak_before_drop": None,
        "price_peak_time": None,
        "price_min_after_drop": None,
        "price_min_time": None,
        "price_drop_ratio": None,
        "price_drop_flag": None,
        "price_drop_factor_time": None,
        "_price_status": None
    }

    # Only fully completed one-minute candles are used before t_rug so a candle crossing the event cannot leak post-event price information into P_peak.
    pre_event_points = [ point for point in price_points if (point["time"] >= token_creation_time and point["time"] + 60 <= liquidity_drop_time) ]

    # Round upward to the next minute boundary so no candle that partially overlaps the liquidity event is used as a purely post-event candle.
    first_post_event_minute = (liquidity_drop_time + 60 - 1) // 60 * 60

    post_event_points = [ point for point in price_points if (point["time"] >= first_post_event_minute and point["time"] < outcome_end_time) ]

    if not pre_event_points:
        result["_price_status"] = "UNKNOWN_MISSING_PRE_EVENT_PRICE"

        return result

    if not post_event_points:
        result["_price_status"] = "UNKNOWN_MISSING_POST_EVENT_PRICE"

        return result

    # The rule uses global extrema over the valid pre-event and post-event regions rather than the price immediately adjacent to t_rug.
    peak_point = max(pre_event_points, key=lambda point: point["high"])

    minimum_point = min(post_event_points, key=lambda point: point["low"])

    peak_price = peak_point["high"]
    minimum_price = minimum_point["low"]

    if peak_price is None or peak_price <= 0:
        result["_price_status"] = "UNKNOWN_INVALID_PRE_EVENT_PRICE"

        return result

    if minimum_price is None or minimum_price <= 0:
        result["_price_status"] = "UNKNOWN_INVALID_POST_EVENT_PRICE"

        return result

    # Measure the relative price collapse from the pre-event peak to the minimum price observed after the liquidity event.
    price_drop_ratio = max(0.0, 1.0 - (minimum_price / peak_price))

    result.update(
        {
            "price_peak_before_drop": peak_price,
            "price_peak_time": peak_point["time"],
            "price_min_after_drop": minimum_price,
            "price_min_time": minimum_point["time"],
            "price_drop_ratio": price_drop_ratio,

            # The rule uses a strict threshold: > 0.90.
            "price_drop_flag": price_drop_ratio > PRICE_DROP_THRESHOLD,

            "price_drop_factor_time": minimum_point["time"],
            "_price_status": "VALID"
        }
    )

    return result


# ============================================================
# LABEL FACTOR 3: PERSISTENT LIQUIDITY DEPLETION
# ============================================================

def compute_persistent_liquidity_depletion(liquidity_points, liquidity_peak_before_drop, liquidity_at_drop, liquidity_drop_time, outcome_end_time):
    """
    Confirm that the liquidity collapse remains persistent after t_rug.
    The low-liquidity threshold is 20% of the pre-drop peak. Starting from the liquidity observed at t_rug, each value is carried forward as a step function until the next one-minute observation.
    The function integrates the number of seconds spent at or below the threshold during the fixed confirmation interval.
    Persistence is True only when:
        1. at least 80% of the interval is spent below the threshold.
        2. liquidity is still below the threshold at the interval endpoint.
    The complete confirmation interval and historical coverage must remain inside the fixed 24-hour outcome horizon.
    """

    result = {
        "persistence_confirmation_seconds": PERSISTENCE_CONFIRMATION_SECONDS,
        "persistence_end_time": None,
        "persistence_liquidity_threshold": None,
        "persistence_low_liquidity_seconds": None,
        "persistence_ratio": None,
        "liquidity_at_persistence_end": None,
        "persistent_liquidity_depletion_flag": None,
        "persistent_liquidity_depletion_factor_time": None,
        "_persistence_status": None
    }

    if liquidity_peak_before_drop is None or liquidity_peak_before_drop <= 0:
        result["_persistence_status"] = "UNKNOWN_INVALID_PERSISTENCE_BASELINE"
        return result

    if liquidity_at_drop is None or liquidity_at_drop < 0 or liquidity_drop_time is None:
        result["_persistence_status"] = "UNKNOWN_INVALID_PERSISTENCE_START"
        return result

    persistence_end_time = liquidity_drop_time + PERSISTENCE_CONFIRMATION_SECONDS

    low_liquidity_threshold = PERSISTENCE_LOW_LIQUIDITY_RATIO * liquidity_peak_before_drop

    result["persistence_end_time"] = persistence_end_time

    result["persistence_liquidity_threshold"] = low_liquidity_threshold

    # The complete confirmation period must remain inside the fixed 24-hour outcome window.
    if persistence_end_time > outcome_end_time:
        result["_persistence_status"] = "UNKNOWN_INSUFFICIENT_PERSISTENCE_HORIZON"
        return result

    # Historical coverage must extend through the end of the confirmation interval. Otherwise persistence cannot be reconstructed reliably.
    if not liquidity_points or liquidity_points[-1][0] < persistence_end_time:
        result["_persistence_status"] = "UNKNOWN_INSUFFICIENT_PERSISTENCE_COVERAGE"
        return result

    # Factor 1 already provides the liquidity value at t_rug. Between later one-minute snapshots, the latest observed value is carried forward as a step function.
    current_liquidity = liquidity_at_drop
    cursor_time = liquidity_drop_time
    low_liquidity_seconds = 0

    # Persistence is measured by integrating the amount of time that liquidity remains below the persistence threshold after the liquidity-drop event.
    for (point_time, liquidity) in liquidity_points:
        if point_time <= liquidity_drop_time:
            continue

        if point_time > persistence_end_time:
            break

        if liquidity is None or liquidity < 0:
            result["_persistence_status"] = "UNKNOWN_INVALID_PERSISTENCE_VALUE"
            return result

        segment_seconds = point_time - cursor_time

        if segment_seconds < 0:
            result["_persistence_status"] = "UNKNOWN_INVALID_PERSISTENCE_INTERVAL"
            return result

        if current_liquidity <= low_liquidity_threshold:
            low_liquidity_seconds += segment_seconds

        current_liquidity = liquidity
        cursor_time = point_time

    final_segment_seconds =  persistence_end_time - cursor_time

    if final_segment_seconds < 0:
        result["_persistence_status"] = "UNKNOWN_INVALID_PERSISTENCE_INTERVAL"
        return result

    if current_liquidity <= low_liquidity_threshold:
        low_liquidity_seconds += final_segment_seconds

    persistence_ratio = low_liquidity_seconds / PERSISTENCE_CONFIRMATION_SECONDS
    persistence_ratio = min(1.0, max(0.0, persistence_ratio))

    # The endpoint condition prevents a temporary low-liquidity period followed by full recovery from being classified as persistent depletion.
    persistent_flag = persistence_ratio >= PERSISTENCE_REQUIRED_RATIO and current_liquidity <= low_liquidity_threshold

    # Factor 3 becomes observable only at persistence_end_time, after the full confirmation interval has elapsed.
    result.update(
        {
            "persistence_low_liquidity_seconds": low_liquidity_seconds,
            "persistence_ratio": persistence_ratio,
            "liquidity_at_persistence_end": current_liquidity,
            "persistent_liquidity_depletion_flag": persistent_flag,
            "persistent_liquidity_depletion_factor_time": persistence_end_time,
            "_persistence_status": "VALID"
        }
    )

    return result

# ============================================================
# EXPERIMENT LABEL BUILDING
# ============================================================

def make_base_label_row(lp_address, token_mint, token_creation_time, experiment, outcome_end_time):
    """
    Initialize one output row for one token-experiment pair.
    The experiment coefficient is converted into an integer number of seconds, and the observation cutoff is anchored to token creation.
    Factor fields remain None until their corresponding computations are valid and applicable.
    _temporary_error is an internal execution flag and is not written to the public CSV.
    """

    # Convert the paper coefficient into seconds and derive the observation boundary relative to token creation time.
    observation_window_seconds = int(round(experiment["fraction_of_day"] * OUTCOME_WINDOW_SECONDS))

    observation_end_time = None

    if token_creation_time is not None:
        observation_end_time = token_creation_time + observation_window_seconds

    # Explicit initialization keeps all output rows schema-compatible, including UNKNOWN and IN_WINDOW_EVENT rows with unavailable factors.
    return {
        "lp_address": lp_address,
        "token_mint": token_mint,
        "token_creation_time": token_creation_time,
        "experiment_id": experiment["experiment_id"],
        "observation_window_seconds": observation_window_seconds,
        "observation_end_time": observation_end_time,
        "outcome_end_time": outcome_end_time,

        "liquidity_peak_before_drop": None,
        "liquidity_at_drop": None,
        "liquidity_drop_ratio": None,
        "liquidity_drop_flag": None,
        "liquidity_drop_factor_time": None,

        "price_peak_before_drop": None,
        "price_peak_time": None,
        "price_min_after_drop": None,
        "price_min_time": None,
        "price_drop_ratio": None,
        "price_drop_flag": None,
        "price_drop_factor_time": None,

        "persistence_confirmation_seconds": PERSISTENCE_CONFIRMATION_SECONDS,
        "persistence_end_time": None,
        "persistence_liquidity_threshold": None,
        "persistence_low_liquidity_seconds": None,
        "persistence_ratio": None,
        "liquidity_at_persistence_end": None,
        "persistent_liquidity_depletion_flag": None,
        "persistent_liquidity_depletion_factor_time": None,

        "label_available_time": None,
        "label_status": None,
        "rug_pull_label": None,

        "_temporary_error": False
    }


def build_label_rows(input_row, pending_experiments):
    """
    Build all pending experiment labels for one token.
    Token-level liquidity history is fetched once and reused by LiquidityDrop and PersistentLiquidityDepletion across all observation-window experiments.
    Price history is fetched at most once and shared by every experiment that has valid LiquidityDrop and persistence results.
    Label construction follows this pipeline:
        1. Fetch token creation time.
        2. Fetch token-level liquidity history once.
        3. Compute LiquidityDrop for every observation window.
        4. Compute PersistentLiquidityDepletion only when a valid liquidity event exists.
        5. Fetch price history once if required.
        6. Compute PriceDrop.
        7. Apply the final labeling rule.
    """

    lp_address = clean_string(input_row.get("LIQUIDITY_POOL_ADDRESS"))

    token_mint = clean_string(input_row.get("MINT"))

    token_creation_time, creation_error = get_token_creation_time(token_mint)

    # A transport or RPC/API failure is temporary and no row is saved and every pending experiment remains eligible for retry in a later execution.
    if creation_error:
        rows = []

        for experiment in pending_experiments:
            row = make_base_label_row(
                lp_address=lp_address,
                token_mint=token_mint,
                token_creation_time=None,
                experiment=experiment,
                outcome_end_time=None
            )

            row["_temporary_error"] = True
            rows.append(row)

        return rows

    # A successful request with no recoverable creation timestamp is a data limitation rather than a transient failure. Save auditable UNKNOWN rows.
    if token_creation_time is None:
        rows = []

        for experiment in pending_experiments:
            row = make_base_label_row(
                lp_address=lp_address,
                token_mint=token_mint,
                token_creation_time=None,
                experiment=experiment,
                outcome_end_time=None
            )

            row["label_status"] = "UNKNOWN_MISSING_TOKEN_CREATION_TIME"

            rows.append(row)

        return rows

    # All experiments share the same fixed outcome boundary, exactly 24 hours after token creation.
    outcome_end_time = token_creation_time + OUTCOME_WINDOW_SECONDS

    rows_by_experiment = {}

    for experiment in pending_experiments:
        experiment_id = experiment["experiment_id"]

        rows_by_experiment[experiment_id] = (
            make_base_label_row(
                lp_address=lp_address,
                token_mint=token_mint,
                token_creation_time=token_creation_time,
                experiment=experiment,
                outcome_end_time=outcome_end_time
            )
        )

    # Retrieve one token-level first-day liquidity history and reuse it for every pending observation-window experiment.
    (liquidity_points, liquidity_error, liquidity_coverage_complete, liquidity_reason) = get_birdeye_liquidity_history(
                                                                                            token_mint=token_mint,
                                                                                            start_time=token_creation_time,
                                                                                            end_time=outcome_end_time
                                                                                        )

    if liquidity_error:
        for row in rows_by_experiment.values():
            row["_temporary_error"] = True

        return list(rows_by_experiment.values())

    # Missing or structurally incomplete historical coverage cannot support a defensible binary target, so every affected experiment is saved UNKNOWN.
    if not liquidity_points or not liquidity_coverage_complete:
        status = "UNKNOWN_"+ (liquidity_reason or "LIQUIDITY_HISTORY").upper()

        for row in rows_by_experiment.values():
            row["label_status"] = status

        return list(rows_by_experiment.values())

    liquidity_results = {}
    persistence_results = {}
    requires_price_history = False

    # Factors 1 and 3 reuse the same token-level liquidity history. No additional liquidity API request is needed.
    for experiment in pending_experiments:
        experiment_id = experiment["experiment_id"]

        row = rows_by_experiment[experiment_id]

        liquidity_result = (
            compute_liquidity_drop_for_experiment(
                liquidity_points=liquidity_points,
                token_creation_time=token_creation_time,
                observation_end_time=(row["observation_end_time"]),
                outcome_end_time=outcome_end_time
            )
        )

        liquidity_results[experiment_id] = liquidity_result

        row.update({ key: value for (key, value) in liquidity_result.items() if not key.startswith("_") })

        # Persistence applies only when Factor 1 found a valid post-observation liquidity-drop event.
        if liquidity_result["_liquidity_status"] == "VALID" and liquidity_result["liquidity_drop_flag"] is True:
            persistence_result = (
                compute_persistent_liquidity_depletion(
                    liquidity_points=liquidity_points,
                    liquidity_peak_before_drop=(liquidity_result["liquidity_peak_before_drop"]),
                    liquidity_at_drop=(liquidity_result["liquidity_at_drop"]),
                    liquidity_drop_time=(liquidity_result["liquidity_drop_factor_time"]),
                    outcome_end_time=outcome_end_time
                )
            )

            persistence_results[experiment_id] = persistence_result

            row.update({ key: value for (key, value) in persistence_result.items() if not key.startswith("_") })

            # PriceDrop is calculated whenever Factors 1 and 3 were both computable, even when persistence is False. This keeps all three factor values available for audit.
            if persistence_result["_persistence_status"] == "VALID":
                requires_price_history = True

    price_points = []
    price_error = False
    price_coverage_complete = False
    price_reason = None

    # Price history is fetched at most once per token and shared across every experiment that requires Factor 2.
    if requires_price_history:
        (price_points, price_error, price_coverage_complete, price_reason) = get_birdeye_price_ohlcv(
                                                                                token_mint=token_mint,
                                                                                start_time=token_creation_time,
                                                                                end_time=outcome_end_time
                                                                            )

    # Finalize each experiment independently. The same token may be VALID under one cutoff and IN_WINDOW_EVENT or UNKNOWN under another cutoff.
    for experiment in pending_experiments:
        experiment_id = experiment["experiment_id"]

        row = rows_by_experiment[experiment_id]

        liquidity_result = liquidity_results[experiment_id]

        liquidity_status = liquidity_result["_liquidity_status"]

        # The target event is already visible before or at this experiment's feature cutoff. Preserve the row for audit but do not assign a binary supervised-learning label.
        if liquidity_status == "IN_WINDOW_EVENT":
            row["label_status"] = "IN_WINDOW_EVENT"

            row["label_available_time"] = liquidity_result["liquidity_drop_factor_time"]

            continue

        if liquidity_status != "VALID":
            row["label_status"] = liquidity_status
            continue

        # Without LiquidityDrop, the final three-factor AND rule is false. PriceDrop and persistence are not applicable.
        if liquidity_result["liquidity_drop_flag"] is False:
            row["label_available_time"] = outcome_end_time

            row["label_status"] = "VALID"
            row["rug_pull_label"] = 0

            continue

        persistence_result = persistence_results.get(experiment_id)

        if persistence_result is None:
            row["label_status"] = "UNKNOWN_MISSING_PERSISTENCE_RESULT"
            continue

        if persistence_result["_persistence_status"] != "VALID":
            row["label_status"] = persistence_result["_persistence_status"]
            continue

        # A failed price request is temporary and must not become a permanent UNKNOWN label so leave the experiment unprocessed for a future retry.
        if price_error:
            row["_temporary_error"] = True
            continue

        if not price_points or not price_coverage_complete:
            row["label_status"] = "UNKNOWN_" + ( price_reason or "PRICE_HISTORY").upper()
            continue

        # Factor 2 is evaluated only after a valid liquidity-drop event because the price decline is defined relative to the detected liquidity event.
        price_result = compute_price_drop(
            price_points=price_points,
            token_creation_time=token_creation_time,
            liquidity_drop_time=liquidity_result["liquidity_drop_factor_time"],
            outcome_end_time=outcome_end_time
        )

        row.update(
            { key: value for (key, value) in price_result.items() if not key.startswith("_") }
        )

        if price_result["_price_status"] != "VALID":
            row["label_status"] = price_result["_price_status"]
            continue

        # Final rule:
        # LiquidityDrop AND PriceDrop AND PersistentLiquidityDepletion
        row["rug_pull_label"] = int(
            liquidity_result["liquidity_drop_flag"]
            and price_result["price_drop_flag"]
            and persistence_result["persistent_liquidity_depletion_flag"]
        )

        # Factor 2 examines the complete first-day outcome period, so the final three-factor label is available at 24 hours.
        row["label_available_time"] = outcome_end_time

        row["label_status"] = "VALID"

    return list(rows_by_experiment.values())

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Coordinate the complete resumable rule-based labeling workflow.
    The procedure:
        1. validates and cleans the raw 2024 cohort;
        2. deduplicates rows at token-mint level;
        3. applies reproducible sampling;
        4. identifies unfinished token-experiment pairs;
        5. builds and saves valid, excluded, and unknown rows;
        6. leaves temporary failures pending for retry;
        7. releases per-token caches after processing.
    """
        
    print("=" * 70)
    print("STARTED RULE-BASED LABEL BUILDING FOR 2024")
    print("=" * 70)

    # Fail before reading or modifying dataset files when Birdeye authentication is not configured.
    if not BIRDEYE_API_KEY:
        raise SystemExit("BIRDEYE_API_KEY is empty.")

    df = load_and_sample_input_file(
        input_file=INPUT_FILE,
        required_columns=[
            "LIQUIDITY_POOL_ADDRESS",
            "MINT"
        ]
    )

    # Labeling is token-level, so eliminate repeated mint rows before random sampling or external history requests.
    df = deduplicate_by_mint(df)

    # Sampling is performed after deduplication so every unique token mint has the same selection probability.
    if SAMPLE_FRACTION < 1.0:
        df = df.sample(
            frac=SAMPLE_FRACTION,
            random_state=RANDOM_SEED
        ).reset_index(drop=True)

    print(f"Unique 2024 token mints selected: {len(df)}")

    # Validate the public CSV schema before loading resume information.
    ensure_output_file_exists()

    processed_labels = load_processed_labels()

    # Build a token-level queue containing only the experiments that have not already been marked complete.
    pending_tokens = []
    total_pending_experiments = 0

    for (_, input_row) in df.iterrows():
        token_mint = clean_string(input_row.get("MINT"))

        # Skip only experiment rows already recorded in the progress file; unfinished experiments for the same token remain pending.
        pending_experiments = [ experiment for experiment in OBSERVATION_WINDOWS if make_label_key(token_mint=token_mint, experiment_id=experiment["experiment_id"]) not in processed_labels ]

        if pending_experiments:
            pending_tokens.append((input_row, pending_experiments))

            total_pending_experiments += len(pending_experiments)

    print(f"Already processed experiment labels: {len(processed_labels)}")
    print(f"Remaining token mints: {len(pending_tokens)}")
    print(f"Remaining experiment labels: {total_pending_experiments}")

    print("-" * 70)

    saved_valid_labels = 0
    saved_excluded_rows = 0
    skipped_temporary_errors = 0

    # Tokens are processed independently so a temporary failure for one token does not interrupt labeling of the remaining dataset.
    for (input_row, pending_experiments) in tqdm(pending_tokens, desc="Building 2024 labels"):
        token_mint = clean_string(input_row.get("MINT"))

        try:
            rows = build_label_rows(
                input_row=input_row,
                pending_experiments=pending_experiments
            )

            for row in rows:
                experiment_id = row["experiment_id"]

                label_key = make_label_key(
                    token_mint=token_mint,
                    experiment_id=experiment_id
                )

                # Temporary failures are neither written nor marked as processed, allowing the same token-experiment pair to be retried later.
                if row.get("_temporary_error"):
                    skipped_temporary_errors += 1

                    print(f" Temporary API error; not saved: {label_key}")

                    continue

                # Internal control fields such as _temporary_error are excluded before the public label row is appended to the CSV.
                clean_row = { column: row.get(column) for column in OUTPUT_COLUMNS }

                if not append_row_to_output(clean_row):
                    print(f" Could not save label row: {label_key}")
                    continue

                # Mark the token-experiment pair only after its output row has been written successfully.
                if not mark_label_as_processed(token_mint=token_mint, experiment_id=experiment_id):
                    print(f" Row was saved, but progress was not updated: {label_key}")
                    continue

                # Count binary supervised labels separately from auditable excluded or unknown rows.
                if row.get("label_status") == "VALID":
                    saved_valid_labels += 1
                else:
                    saved_excluded_rows += 1

                print(f" Label done: {label_key}; status={row.get('label_status')}; rug_pull_label= {row.get('rug_pull_label')}")

        except Exception as e:
            print(f" Error while processing token {token_mint}: {e}")

        finally:
            # Liquidity history is shared by Factors 1 and 3, while price history is shared across all experiments that require Factor 2.
            # Both caches are released after the current token is completed.
            LIQUIDITY_HISTORY_CACHE.clear()
            PRICE_OHLCV_CACHE.clear()
            TOKEN_CREATION_TIME_CACHE.pop(
                token_mint,
                None
            )

    print("-" * 70)
    print(f"Saved valid labels: {saved_valid_labels}")

    print(f"Saved excluded/unknown rows: {saved_excluded_rows}")
    print(f"Skipped temporary API errors: {skipped_temporary_errors}")

    print("=" * 70)
    print("FINISHED RULE-BASED LABEL BUILDING")
    print("=" * 70)


if __name__ == "__main__":
    main()