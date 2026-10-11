"""Bounded loopback HTTP benchmark. Does not claim production capacity or peak RSS."""

import argparse
import json
import math
import platform
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def request(url):
    start = time.perf_counter()
    try:
        with urlopen(url, timeout=10) as response:
            payload = json.load(response)
            valid = (
                isinstance(payload, list)
                and bool(payload)
                and all("prediction" in row and "model_version" in row for row in payload)
            )
            status = str(response.status) if valid else "invalid_payload"
    except HTTPError as error:
        status = str(error.code)
    except (URLError, TimeoutError, ValueError, OSError):
        status = "request_error"
    return (time.perf_counter() - start) * 1000, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/forecasts/items/ITEM_001?store_id=STORE_1&horizon=7"
    )
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if urlparse(args.url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("This diagnostic permits loopback targets only")
    if (
        not 1 <= args.requests <= 10000
        or not 1 <= args.concurrency <= 32
        or not 0 <= args.warmup <= 100
    ):
        parser.error("Use 1..10000 requests, 1..32 concurrency, 0..100 warmup")
    for _ in range(args.warmup):
        if request(args.url)[1] != "200":
            raise SystemExit("Warmup failed; no benchmark recorded")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(request, [args.url] * args.requests))
    elapsed = time.perf_counter() - start
    statuses = Counter(status for _, status in results)
    latencies = [ms for ms, _ in results]
    print(
        json.dumps(
            {
                "measured_at": datetime.now(UTC).isoformat(),
                "environment": "localhost Docker Compose; synthetic data; one repeated item query",
                "client_platform": platform.platform(),
                "client_python": platform.python_version(),
                "url": args.url,
                "requests": args.requests,
                "concurrency": args.concurrency,
                "warmup_requests_excluded": args.warmup,
                "connection_reuse": False,
                "duration_seconds": elapsed,
                "requests_per_second": args.requests / elapsed,
                "error_rate": 1 - statuses["200"] / args.requests,
                "statuses": dict(statuses),
                "latency_ms": {
                    f"p{int(q * 100)}": percentile(latencies, q) for q in [0.5, 0.95, 0.99]
                },
                "scope": "Closed-loop warm query test; not maximum capacity or an SLO",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
