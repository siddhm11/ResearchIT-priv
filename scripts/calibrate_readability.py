"""Derive the readability band thresholds from the real corpus.

A scale where 90% of papers land in one band tells a reader nothing. These
thresholds are the terciles of the actual score distribution, so the label
discriminates by construction. Re-run if the corpus changes materially.

    .venv/bin/python scripts/calibrate_readability.py
"""
import sqlite3
import sys

sys.path.insert(0, ".")
from app import config, readability  # noqa: E402


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    rows = [r[0] for r in conn.execute(
        "SELECT abstract FROM paper_metadata "
        "WHERE abstract IS NOT NULL AND length(abstract) > 200")]
    if not rows:
        print("no abstracts cached; run the app first")
        return

    scores = sorted(readability.score(a) for a in rows)
    n = len(scores)
    t1, t2 = scores[n // 3], scores[2 * n // 3]

    print(f"n = {n} abstracts")
    print(f"  min {scores[0]:.3f}  median {scores[n // 2]:.3f}  max {scores[-1]:.3f}")
    print(f"  tercile 1 (technical  at) = {t1:.3f}")
    print(f"  tercile 2 (specialist at) = {t2:.3f}")
    print()
    print(f"  currently _TECHNICAL_AT  = {readability._TECHNICAL_AT}")
    print(f"           _SPECIALIST_AT = {readability._SPECIALIST_AT}")
    print()
    from collections import Counter
    dist = Counter(readability.level(a) for a in rows)
    for lvl in readability.LEVELS:
        pct = dist[lvl] / n * 100
        print(f"  {lvl:<12} {dist[lvl]:>4}  {pct:>5.1f}%")


if __name__ == "__main__":
    main()
