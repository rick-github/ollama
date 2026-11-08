#!/usr/bin/env python3
"""
Ollama pricing scraper for Prometheus.

Scrapes https://ollama.com/pricing periodically and exposes per-model token
prices as Prometheus gauges on a /metrics endpoint. Designed to run in its
own Docker container alongside the Ollama monitoring stack.

Exposes:
  ollama_cloud_token_price{model,tier,token_type} <price_per_million_tokens>

  tier:       "peak" or "off_peak"
  token_type: "prompt", "cached_prompt", or "eval"

Prometheus scrapes this endpoint and Grafana joins the price gauges with
the ollama server's token count gauges to compute actual cost.

Usage:
  python3 pricing_scraper.py [--interval 3600] [--port 9100]

Environment:
  OLLAMA_PRICING_URL     URL of the pricing page (default: https://ollama.com/pricing)
  OLLAMA_PRICING_INTERVAL  Seconds between scrapes (default: 3600)
  OLLAMA_PRICING_PORT     Port for /metrics endpoint (default: 9100)
"""

import argparse
import calendar
import datetime
import html
import re
import sys
import time
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

PRICING_URL = "https://ollama.com/pricing"

# Pricing data: { model: { "off_peak": { "prompt": X, "cached_prompt": Y, "eval": Z },
#                          "peak":     { "prompt": X, "cached_prompt": Y, "eval": Z } } }
pricing_data: dict[str, dict[str, dict[str, float]]] = {}
pricing_timestamp: float = 0

# Peak schedule parsed from the pricing page.
# peak_start_hour, peak_end_hour: UTC hours (e.g. 12, 18)
# peak_days: set of weekday numbers (0=Monday ... 6=Sunday)
peak_start_hour: int = 12
peak_end_hour: int = 18
peak_days: set[int] = {0, 1, 2, 3, 4}  # Mon-Fri, updated from scraped data


def parse_price(s: str) -> float | None:
    """Parse a price string like '$0.15' or '-' (no caching)."""
    s = s.strip()
    if s == "-" or s == "" or s == "—":
        return None
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_model_name(href: str) -> str:
    """Extract the model name from a library link like /library/glm-5.2."""
    m = re.search(r"/library/([^/]+)", href)
    if m:
        return m.group(1)
    return ""


WEEKDAY_NAMES = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


def parse_peak_schedule(body: str) -> tuple[int, int, set[int]]:
    """Parse the peak pricing schedule from the page text.

    Expects text like "Peak pricing applies between 12:00 and 18:00 UTC,
    Monday to Friday." Returns (start_hour, end_hour, weekdays_set).
    Falls back to (12, 18, Mon-Fri) if parsing fails.
    """
    # Remove HTML tags and decode entities to get plain text
    text = html.unescape(re.sub(r"<[^>]+>", " ", body))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Find the peak schedule sentence
    m = re.search(r"between\s+(\d{1,2}):(\d{2})\s+and\s+(\d{1,2}):(\d{2})\s+UTC", text, re.IGNORECASE)
    start_hour = 12
    end_hour = 18
    if m:
        start_hour = int(m.group(1))
        end_hour = int(m.group(3))

    # Find day range: "Monday to Friday", "Mon-Fri", "Monday through Friday"
    days: set[int] = set()
    # Try "X to Y" or "X through Y" or "X-Y"
    m = re.search(
        r"(monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thurs|friday|fri|saturday|sat|sunday|sun)"
        r"\s*(?:to|through|-|–|—)\s*"
        r"(monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thurs|friday|fri|saturday|sat|sunday|sun)",
        text,
        re.IGNORECASE,
    )
    if m:
        start_day = WEEKDAY_NAMES.get(m.group(1).lower(), 0)
        end_day = WEEKDAY_NAMES.get(m.group(2).lower(), 4)
        # Build the range (wrapping if needed)
        d = start_day
        while True:
            days.add(d)
            if d == end_day:
                break
            d = (d + 1) % 7
            if len(days) >= 7:
                break

    if not days:
        # Try comma-separated or "and"-separated list
        m = re.search(
            r"(monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thurs|friday|fri|saturday|sat|sunday|sun)"
            r"(?:\s*(?:,|and)\s*"
            r"(monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thurs|friday|fri|saturday|sat|sunday|sun))*",
            text,
            re.IGNORECASE,
        )
        if m:
            for word in re.findall(
                r"(monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thurs|friday|fri|saturday|sat|sunday|sun)",
                text[m.start():m.end()],
                re.IGNORECASE,
            ):
                d = WEEKDAY_NAMES.get(word.lower())
                if d is not None:
                    days.add(d)

    if not days:
        days = {0, 1, 2, 3, 4}  # Default: Mon-Fri

    return start_hour, end_hour, days


def scrape_pricing(url: str) -> dict[str, dict[str, dict[str, float]]]:
    """Scrape the pricing page and return structured pricing data."""
    global peak_start_hour, peak_end_hour, peak_days
    req = urllib.request.Request(url, headers={"User-Agent": "ollama-pricing-scraper/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")

    # Parse the peak schedule from the page text
    peak_start_hour, peak_end_hour, peak_days = parse_peak_schedule(body)

    data: dict[str, dict[str, dict[str, float]]] = {}

    # Find all model pricing tables. The page has two tables:
    # 1. Off-peak (default) pricing
    # 2. Peak pricing (header: "Peak pricing")
    #
    # Each table row has:
    #   <td><a href="/library/MODEL">...</a></td>
    #   <td>$X.XX</td>  (input)
    #   <td>$X.XX</td>  (cached input, may be "-")
    #   <td>$X.XX</td>  (output)

    # Split on the "Peak pricing" heading to separate peak from off-peak tables
    peak_split = re.split(r"<h[23][^>]*>\s*Peak pricing", body, flags=re.IGNORECASE)
    off_peak_html = peak_split[0]
    peak_html = peak_split[1] if len(peak_split) > 1 else ""

    def parse_table(html_str: str, tier: str) -> None:
        # Find table rows with model links
        rows = re.findall(
            r"<tr[^>]*>.*?<a\s+href=\"(/library/[^\"]+)\"[^>]*>(.*?)</a>.*?</tr>",
            html_str,
            re.DOTALL,
        )
        if not rows:
            # Try a more lenient pattern
            rows = re.findall(
                r"<tr[^>]*>(.*?)<a\s+href=\"(/library/[^\"]+)\"[^>]*>(.*?)</a>(.*?)</tr>",
                html_str,
                re.DOTALL,
            )
            rows = [(href, text + rest) for _, href, text, rest in rows]

        for href, _text in rows:
            model = extract_model_name(href)
            if not model:
                continue

            # Get the full row HTML to extract price cells
            row_match = re.search(
                r"<tr[^>]*>.*?" + re.escape(f'href="{href}"') + r".*?</tr>",
                html_str,
                re.DOTALL,
            )
            if not row_match:
                continue
            row_html = row_match.group(0)

            # Extract all <td> cell contents
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
            if len(cells) < 4:
                continue

            prompt_price = parse_price(html.unescape(re.sub(r"<[^>]+>", "", cells[1]).strip()))
            cached_price = parse_price(html.unescape(re.sub(r"<[^>]+>", "", cells[2]).strip()))
            eval_price = parse_price(html.unescape(re.sub(r"<[^>]+>", "", cells[3]).strip()))

            if model not in data:
                data[model] = {}
            data[model][tier] = {
                "prompt": prompt_price or 0.0,
                "cached_prompt": cached_price if cached_price is not None else 0.0,
                "eval": eval_price or 0.0,
            }

    parse_table(off_peak_html, "off_peak")
    parse_table(peak_html, "peak")

    return data


def update_pricing(url: str) -> None:
    """Fetch and update the pricing data."""
    global pricing_data, pricing_timestamp
    try:
        new_data = scrape_pricing(url)
        if new_data:
            pricing_data = new_data
            pricing_timestamp = time.time()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Scraped {len(pricing_data)} models from {url}", file=sys.stderr)
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Warning: no models parsed from {url}", file=sys.stderr)
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error scraping {url}: {e}", file=sys.stderr)


def is_peak_hours(ts: float | None = None) -> bool:
    """Check if the given timestamp (or now) falls in peak pricing hours.

    Uses the peak schedule parsed from the pricing page (peak_start_hour,
    peak_end_hour, peak_days). Falls back to 12:00-18:00 UTC Mon-Fri
    if the schedule hasn't been parsed yet.
    """
    if ts is None:
        ts = time.time()
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    if dt.weekday() not in peak_days:
        return False
    return peak_start_hour <= dt.hour < peak_end_hour


def format_metrics() -> str:
    """Format the pricing data as Prometheus text exposition format."""
    lines = []
    lines.append("# HELP ollama_cloud_token_price Price per million tokens for cloud models.")
    lines.append("# TYPE ollama_cloud_token_price gauge")

    peak = is_peak_hours()

    for model, tiers in sorted(pricing_data.items()):
        for tier, prices in sorted(tiers.items()):
            for token_type, price in sorted(prices.items()):
                lines.append(
                    f'ollama_cloud_token_price{{model="{model}",tier="{tier}",token_type="{token_type}"}} {price}'
                )

    # Current pricing: the applicable price right now (peak or off_peak).
    current_tier = "peak" if peak else "off_peak"
    lines.append("# HELP ollama_cloud_current_token_price Currently applicable price per million tokens (peak or off-peak).")
    lines.append("# TYPE ollama_cloud_current_token_price gauge")

    for model, tiers in sorted(pricing_data.items()):
        prices = tiers.get(current_tier)
        if not prices:
            # If no peak pricing data exists for this model, fall back to off_peak.
            prices = tiers.get("off_peak", {})
        for token_type, price in sorted(prices.items()):
            lines.append(
                f'ollama_cloud_current_token_price{{model="{model}",token_type="{token_type}"}} {price}'
            )

    lines.append("# HELP ollama_pricing_peak_active Whether peak pricing is currently active (1=yes, 0=no).")
    lines.append("# TYPE ollama_pricing_peak_active gauge")
    lines.append(f"ollama_pricing_peak_active {1 if peak else 0}")

    lines.append("# HELP ollama_pricing_peak_start_hour UTC hour when peak pricing starts.")
    lines.append("# TYPE ollama_pricing_peak_start_hour gauge")
    lines.append(f"ollama_pricing_peak_start_hour {peak_start_hour}")

    lines.append("# HELP ollama_pricing_peak_end_hour UTC hour when peak pricing ends.")
    lines.append("# TYPE ollama_pricing_peak_end_hour gauge")
    lines.append(f"ollama_pricing_peak_end_hour {peak_end_hour}")

    lines.append("# HELP ollama_pricing_peak_day Whether peak pricing applies on this weekday (1=yes, 0=no).")
    lines.append("# TYPE ollama_pricing_peak_day gauge")
    day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for d in range(7):
        lines.append(f'ollama_pricing_peak_day{{day="{day_names[d]}"}} {1 if d in peak_days else 0}')

    if pricing_timestamp > 0:
        lines.append("# HELP ollama_pricing_scraper_last_scrape Timestamp of the last successful scrape.")
        lines.append("# TYPE ollama_pricing_scraper_last_scrape gauge")
        lines.append(f"ollama_pricing_scraper_last_scrape {pricing_timestamp}")

    lines.append("# HELP ollama_pricing_scraper_model_count Number of models in the last scrape.")
    lines.append("# TYPE ollama_pricing_scraper_model_count gauge")
    lines.append(f"ollama_pricing_scraper_model_count {len(pricing_data)}")

    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = format_metrics().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ollama pricing scraper\n")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {fmt % args}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Ollama pricing scraper for Prometheus")
    parser.add_argument("--url", default=PRICING_URL, help="Pricing page URL")
    parser.add_argument("--interval", type=int, default=3600, help="Scrape interval in seconds")
    parser.add_argument("--port", type=int, default=9100, help="Metrics endpoint port")
    args = parser.parse_args()

    url = args.url
    interval = args.interval
    port = args.port

    # Initial scrape
    update_pricing(url)

    # Start a background thread for periodic scraping
    import threading

    def scrape_loop():
        while True:
            time.sleep(interval)
            update_pricing(url)

    t = threading.Thread(target=scrape_loop, daemon=True)
    t.start()

    # Start the HTTP server
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Serving /metrics on port {port}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()