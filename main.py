"""Entry for make.fun's gta6-twitch-ccu-forecast-01 bounty.

Forecast Twitch concurrent-viewer quantiles hour by hour for a release day,
starting from hour 1's known print, by transferring the release-day shape of
the k nearest past launches.

Model: k-nearest-launch shape transfer (k=4, level weight 0.4).

  * hour_key(day_start, h) = the ISO hour stamp at day_start + (h - 1) hours.
  * For each comparable launch, build a "from" -> viewers map.  v1 is the
    hour-1 viewers (skip the launch if absent).  ratios[h] = ln(viewers_h / v1)
    for h in 2..24 where present.
  * shape = log signature of the last 48 pre-release hours: take viewers at
    hour_key(day_start, 1 - h) for h in 48..1, keep values >= 1, require at
    least 12 points, normalize by their geometric mean, return ln(v / g).
  * The target launch's shape is built the same way from case "history"
    (points keyed by "at").
  * sim(a, b) = mean squared difference of the last min(len) elements.
  * Rank comparables by sim(target_shape, comp_shape) + 0.4 * (ln(v1) - ln(hour1))^2
    and keep the 4 nearest.
  * For each hour h in 2..24, gather the neighbors' ratios[h] (sorted); if
    none, skip the hour.  mu = mean, sd = sample std.  base = ln(v0) + mu with
    v0 = max(hour1, 1).  Quantile levels [0.05, 0.25, 0.5, 0.75, 0.95] use
    Z = {-1.6449, -0.6745, 0.0, 0.6745, 1.6449}:
    q = max(int(round(exp(base + Z * sd))), 1).

Quantile widths use 0.9 times the neighbors' residual standard deviation with
a floor of 0.4 log units, tuned by leave-one-out backtest.

Quantiles are clamped to [1, 1000000000] because the grader rejects values
outside that range.

Deterministic: standard library only, no randomness, no network; both runs
write the same bytes.
"""

from datetime import datetime, timedelta
import json
import math
import os


Z = {
    0.05: -1.6449,
    0.25: -0.6745,
    0.5: 0.0,
    0.75: 0.6745,
    0.95: 1.6449,
}
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
K = 4
LEVEL_WEIGHT = 0.4


def parse_ts(value):
    """Parse an ISO UTC timestamp into an aware datetime."""
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def hour_key(day_start, h):
    """ISO hour stamp for hour ``h`` (1-indexed) of a launch day."""
    return (day_start + timedelta(hours=h - 1)).strftime("%Y-%m-%dT%H:00:00Z")


def log_signature(day_start, viewers):
    """Scale-free log signature of the last 48 pre-release hours."""
    vals = []
    for h in range(48, 0, -1):
        v = viewers.get(hour_key(day_start, 1 - h))
        if v is not None and v >= 1:
            vals.append(v)
    if len(vals) < 12:
        return None
    g = math.exp(sum(math.log(v) for v in vals) / len(vals))
    return [math.log(v / g) for v in vals]


def sim(a, b):
    """Mean squared difference of the last min(len) elements."""
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    return sum((x - y) ** 2 for x, y in zip(a[-n:], b[-n:])) / n


def _clamp(x):
    """Clamp x into the quantile range [1, 1_000_000_000]."""
    return min(max(x, 1), 1_000_000_000)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    in_path = None
    for name in sorted(os.listdir(input_dir)):
        if name.endswith(".in"):
            in_path = os.path.join(input_dir, name)
            break
    if in_path is None:
        raise FileNotFoundError("no .in file in " + input_dir)

    with open(in_path, "r", encoding="utf-8") as f:
        case = json.load(f)

    day_start = parse_ts(case["day_start"])
    hour1 = case["known"][0]["viewers"]

    # Target shape from the launch's own pre-release history.
    history = {p["at"]: p["viewers"] for p in case["history"]}
    target_shape = log_signature(day_start, history)

    # Score each comparable launch by shape proximity and viewer level.
    scored = []
    for comp in case["comparables"]:
        comp_day = parse_ts(comp["day_start"])
        viewers = {p["from"]: p["viewers"] for p in comp["hourly"]}

        v1 = viewers.get(hour_key(comp_day, 1))
        if v1 is None or v1 <= 0:
            continue

        ratios = {}
        for h in range(2, 25):
            v = viewers.get(hour_key(comp_day, h))
            if v is not None and v > 0:
                ratios[h] = math.log(v / v1)

        shape = log_signature(comp_day, viewers)
        if shape is None:
            continue

        level_term = LEVEL_WEIGHT * (math.log(v1) - math.log(max(hour1, 1))) ** 2
        score = sim(target_shape, shape) + level_term
        scored.append((score, ratios))

    scored.sort(key=lambda item: item[0])
    neighbors = [ratios for _, ratios in scored[:K]]

    # Transfer the median release-day ratio path onto hour 1's print.
    hours = []
    for h in range(2, 25):
        vals = sorted(r[h] for r in neighbors if h in r)
        if not vals:
            continue
        mu = sum(vals) / len(vals)
        sd = max(math.sqrt(sum((x - mu) ** 2 for x in vals) / max(len(vals) - 1, 1)) * 0.9, 0.4)
        base = math.log(max(hour1, 1)) + mu
        q = [_clamp(int(round(math.exp(base + Z[t] * sd)))) for t in LEVELS]
        hours.append({"hour": h, "q": q})

    result = {"hours": hours}

    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.basename(in_path)[:-3] + ".out"
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
