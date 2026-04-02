#!/usr/bin/env python3
"""Download public espresso shots from Visualizer.coffee API.

Usage:
    python calibration/fetch_shots.py [--pages N] [--output DIR] [--resume]

Rate limited to ~50 req/min with 2-3 concurrent requests via token bucket.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp


API_BASE = "https://visualizer.coffee/api"
HEADERS = {
    "User-Agent": "coffee-sim-calibration/0.1 (research; github.com/cxc/coffee-sim)",
    "Accept": "application/json",
}

# Rate limiting: 50 requests per 60 seconds
RATE_LIMIT = 50
RATE_WINDOW = 60.0
MAX_CONCURRENT = 3


class TokenBucket:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: float, window: float):
        self.rate = rate
        self.window = window
        self.tokens = rate
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate / self.window)
            self.last_refill = now

            if self.tokens < 1:
                wait = (1 - self.tokens) * self.window / self.rate
                await asyncio.sleep(wait)
                self.tokens = 0
                self.last_refill = time.monotonic()
            else:
                self.tokens -= 1


async def fetch_page(
    session: aiohttp.ClientSession,
    page: int,
    bucket: TokenBucket,
) -> list[dict]:
    """Fetch a page of shot listings (minimal: id, clock, updated_at)."""
    await bucket.acquire()
    url = f"{API_BASE}/shots"
    params = {"items": 100, "page": page}
    async with session.get(url, params=params, headers=HEADERS) as resp:
        if resp.status != 200:
            print(f"  Page {page}: HTTP {resp.status}")
            return []
        data = await resp.json()
        return data.get("data", [])


async def fetch_shot_detail(
    session: aiohttp.ClientSession,
    shot_id: str,
    bucket: TokenBucket,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Fetch full shot detail by ID."""
    async with semaphore:
        await bucket.acquire()
        url = f"{API_BASE}/shots/{shot_id}"
        try:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"  Error fetching {shot_id}: {e}")
            return None


async def main(pages: int, output_dir: Path, resume: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket = TokenBucket(RATE_LIMIT, RATE_WINDOW)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Collect existing IDs for --resume
    existing_ids: set[str] = set()
    if resume:
        for f in output_dir.glob("*.json"):
            existing_ids.add(f.stem)
        if existing_ids:
            print(f"Resume mode: {len(existing_ids)} shots already on disk")

    async with aiohttp.ClientSession() as session:
        # Phase 1: Collect shot IDs from listing pages
        print(f"Fetching {pages} listing pages...")
        all_ids: list[str] = []

        for page in range(1, pages + 1):
            entries = await fetch_page(session, page, bucket)
            if not entries:
                print(f"  Page {page}: empty or error, stopping pagination")
                break
            page_ids = [e["id"] for e in entries if "id" in e]
            all_ids.extend(page_ids)
            print(f"  Page {page}: {len(page_ids)} shots")

        print(f"Total IDs collected: {len(all_ids)}")

        # Filter out already-downloaded shots
        if resume:
            new_ids = [sid for sid in all_ids if sid not in existing_ids]
            print(f"New shots to fetch: {len(new_ids)} (skipping {len(all_ids) - len(new_ids)})")
        else:
            new_ids = all_ids

        # Phase 2: Fetch full details for each shot
        print(f"Fetching {len(new_ids)} shot details (concurrency={MAX_CONCURRENT})...")
        fetched = 0
        errors = 0

        tasks = []
        for shot_id in new_ids:
            task = fetch_shot_detail(session, shot_id, bucket, semaphore)
            tasks.append((shot_id, task))

        # Process in batches to show progress
        batch_size = 50
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start : batch_start + batch_size]
            results = await asyncio.gather(*(t for _, t in batch), return_exceptions=True)

            for (shot_id, _), result in zip(batch, results):
                if isinstance(result, Exception):
                    errors += 1
                    continue
                if result is None:
                    errors += 1
                    continue

                out_path = output_dir / f"{shot_id}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f)
                fetched += 1

            total_done = batch_start + len(batch)
            print(f"  Progress: {total_done}/{len(tasks)} ({fetched} saved, {errors} errors)")

    total_on_disk = len(list(output_dir.glob("*.json")))
    print(f"\nDone. {fetched} new shots saved. {total_on_disk} total on disk.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch shots from Visualizer.coffee")
    parser.add_argument("--pages", type=int, default=20, help="Number of listing pages to fetch (100 shots/page)")
    parser.add_argument("--output", type=str, default="calibration/dataset", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Skip shots already downloaded")
    args = parser.parse_args()

    asyncio.run(main(args.pages, Path(args.output), args.resume))
