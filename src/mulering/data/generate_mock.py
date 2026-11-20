"""Generate synthetic mock AML data matching IBM HI-Small format for local testing."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_mock_data(
    output_dir: str | Path,
    n_normal: int = 15000,
    n_accounts: int = 300,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    banks = [f"{i:03d}" for i in range(1, 6)]
    acct_ids = [f"ACC{i:05d}" for i in range(1, n_accounts + 1)]
    accounts = [(rng.choice(banks), acct) for acct in acct_ids]

    start_time = pd.Timestamp("2022-09-01 00:00:00")
    # 14 days of timestamps
    ts_seconds = rng.integers(0, 14 * 86400, size=n_normal)
    normal_ts = [start_time + pd.Timedelta(seconds=int(s)) for s in ts_seconds]
    normal_ts.sort()

    normal_rows = []
    currencies = ["US Dollar", "Euro", "Yen", "Swiss Franc"]
    pay_formats = ["ACH", "Wire", "Credit Card", "Cheque", "Cash"]

    for t in normal_ts:
        src_bank, src_acct = accounts[rng.integers(0, len(accounts))]
        dst_bank, dst_acct = accounts[rng.integers(0, len(accounts))]
        while src_bank == dst_bank and src_acct == dst_acct:
            dst_bank, dst_acct = accounts[rng.integers(0, len(accounts))]

        amt = float(np.round(rng.lognormal(mean=4.5, sigma=1.2), 2))
        amt = max(5.0, min(amt, 50000.0))
        cur = rng.choice(currencies, p=[0.7, 0.15, 0.1, 0.05])
        fmt = rng.choice(pay_formats)
        normal_rows.append({
            "ts": t.strftime("%Y/%m/%d %H:%M"),
            "src_bank": src_bank,
            "src_acct": src_acct,
            "dst_bank": dst_bank,
            "dst_acct": dst_acct,
            "amount_recv": amt,
            "cur_recv": cur,
            "amount_sent": amt,
            "cur_sent": cur,
            "pay_format": fmt,
            "label": 0,
        })

    # Plant laundering rings
    laundering_rows = []
    pattern_blocks = []

    # Ring 1: CYCLE (4 accounts: A -> B -> C -> D -> A)
    cycle_accts = [
        ("001", "CYC001"), ("002", "CYC002"), ("003", "CYC003"), ("001", "CYC004")
    ]
    cycle_time = start_time + pd.Timedelta(days=2, hours=10)
    cycle_block = ["BEGIN LAUNDERING ATTEMPT - CYCLE"]
    for i in range(len(cycle_accts)):
        sb, sa = cycle_accts[i]
        db, da = cycle_accts[(i + 1) % len(cycle_accts)]
        t_str = (cycle_time + pd.Timedelta(hours=i * 3)).strftime("%Y/%m/%d %H:%M")
        row_str = f"{t_str},{sb},{sa},{db},{da},9500.0,US Dollar,9500.0,US Dollar,Wire,1"
        cycle_block.append(row_str)
        laundering_rows.append({
            "ts": t_str, "src_bank": sb, "src_acct": sa, "dst_bank": db, "dst_acct": da,
            "amount_recv": 9500.0, "cur_recv": "US Dollar",
            "amount_sent": 9500.0, "cur_sent": "US Dollar",
            "pay_format": "Wire", "label": 1
        })
    cycle_block.append("END LAUNDERING ATTEMPT - CYCLE")
    pattern_blocks.append(cycle_block)

    # Ring 2: FAN-OUT (1 Boss -> 5 Mules)
    boss = ("003", "BOSS01")
    fanout_mules = [("001", f"MULE_OUT_{i}") for i in range(1, 6)]
    fanout_time = start_time + pd.Timedelta(days=5, hours=8)
    fanout_block = ["BEGIN LAUNDERING ATTEMPT - FAN-OUT"]
    for i, (mb, ma) in enumerate(fanout_mules):
        t_str = (fanout_time + pd.Timedelta(minutes=i * 20)).strftime("%Y/%m/%d %H:%M")
        row_str = f"{t_str},{boss[0]},{boss[1]},{mb},{ma},4900.0,US Dollar,4900.0,US Dollar,ACH,1"
        fanout_block.append(row_str)
        laundering_rows.append({
            "ts": t_str, "src_bank": boss[0], "src_acct": boss[1], "dst_bank": mb, "dst_acct": ma,
            "amount_recv": 4900.0, "cur_recv": "US Dollar",
            "amount_sent": 4900.0, "cur_sent": "US Dollar",
            "pay_format": "ACH", "label": 1
        })
    fanout_block.append("END LAUNDERING ATTEMPT - FAN-OUT")
    pattern_blocks.append(fanout_block)

    # Ring 3: FAN-IN (5 Mules -> 1 Collector)
    collector = ("004", "COLLECT01")
    fanin_mules = [("002", f"MULE_IN_{i}") for i in range(1, 6)]
    fanin_time = start_time + pd.Timedelta(days=8, hours=14)
    fanin_block = ["BEGIN LAUNDERING ATTEMPT - FAN-IN"]
    for i, (mb, ma) in enumerate(fanin_mules):
        t_str = (fanin_time + pd.Timedelta(minutes=i * 15)).strftime("%Y/%m/%d %H:%M")
        row_str = f"{t_str},{mb},{ma},{collector[0]},{collector[1]},4800.0,US Dollar,4800.0,US Dollar,Wire,1"
        fanin_block.append(row_str)
        laundering_rows.append({
            "ts": t_str, "src_bank": mb, "src_acct": ma, "dst_bank": collector[0], "dst_acct": collector[1],
            "amount_recv": 4800.0, "cur_recv": "US Dollar",
            "amount_sent": 4800.0, "cur_sent": "US Dollar",
            "pay_format": "Wire", "label": 1
        })
    fanin_block.append("END LAUNDERING ATTEMPT - FAN-IN")
    pattern_blocks.append(fanin_block)

    # Ring 4: SCATTER-GATHER (Origin -> 3 Intermediaries -> Sink)
    origin = ("005", "SCAT_ORIG")
    layer = [("001", f"SG_MID_{i}") for i in range(1, 4)]
    sink = ("002", "SCAT_SINK")
    sg_time = start_time + pd.Timedelta(days=11, hours=9)
    sg_block = ["BEGIN LAUNDERING ATTEMPT - SCATTER-GATHER"]
    # Scatter phase
    for i, (mb, ma) in enumerate(layer):
        t_str = (sg_time + pd.Timedelta(minutes=i * 10)).strftime("%Y/%m/%d %H:%M")
        row_str = f"{t_str},{origin[0]},{origin[1]},{mb},{ma},7500.0,US Dollar,7500.0,US Dollar,Wire,1"
        sg_block.append(row_str)
        laundering_rows.append({
            "ts": t_str, "src_bank": origin[0], "src_acct": origin[1], "dst_bank": mb, "dst_acct": ma,
            "amount_recv": 7500.0, "cur_recv": "US Dollar",
            "amount_sent": 7500.0, "cur_sent": "US Dollar",
            "pay_format": "Wire", "label": 1
        })
    # Gather phase
    for i, (mb, ma) in enumerate(layer):
        t_str = (sg_time + pd.Timedelta(hours=4, minutes=i * 10)).strftime("%Y/%m/%d %H:%M")
        row_str = f"{t_str},{mb},{ma},{sink[0]},{sink[1]},7400.0,US Dollar,7400.0,US Dollar,Wire,1"
        sg_block.append(row_str)
        laundering_rows.append({
            "ts": t_str, "src_bank": mb, "src_acct": ma, "dst_bank": sink[0], "dst_acct": sink[1],
            "amount_recv": 7400.0, "cur_recv": "US Dollar",
            "amount_sent": 7400.0, "cur_sent": "US Dollar",
            "pay_format": "Wire", "label": 1
        })
    sg_block.append("END LAUNDERING ATTEMPT - SCATTER-GATHER")
    pattern_blocks.append(sg_block)

    # Combine all transactions and sort by timestamp
    all_df = pd.DataFrame(normal_rows + laundering_rows)
    all_df["ts_dt"] = pd.to_datetime(all_df["ts"], format="%Y/%m/%d %H:%M")
    all_df = all_df.sort_values("ts_dt", kind="stable").drop(columns=["ts_dt"]).reset_index(drop=True)

    # Save HI-Small_Trans.csv
    trans_path = out / "HI-Small_Trans.csv"
    all_df.to_csv(trans_path, index=False)

    # Save HI-Small_Patterns.txt
    patterns_path = out / "HI-Small_Patterns.txt"
    with open(patterns_path, "w") as f:
        for block in pattern_blocks:
            f.write("\n".join(block) + "\n\n")

    # Save HI-Small_accounts.csv
    all_accts = set(zip(all_df["src_bank"], all_df["src_acct"])) | set(zip(all_df["dst_bank"], all_df["dst_acct"]))
    accts_df = pd.DataFrame(list(all_accts), columns=["Bank", "Account"])
    accts_path = out / "HI-Small_accounts.csv"
    accts_df.to_csv(accts_path, index=False)

    n_illicit = int(all_df["label"].sum())
    print(f"Generated mock data in {out}:")
    print(f"  - Total transactions: {len(all_df):,}")
    print(f"  - Laundering transactions: {n_illicit:,} ({n_illicit/len(all_df)*100:.2f}%)")
    print(f"  - Planted rings: {len(pattern_blocks)}")
    print(f"  - Files: {trans_path.name}, {patterns_path.name}, {accts_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic mock AML dataset.")
    parser.add_argument("--output-dir", default="data/raw/ibm")
    parser.add_argument("--n-normal", type=int, default=15000)
    parser.add_argument("--n-accounts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_mock_data(args.output_dir, args.n_normal, args.n_accounts, args.seed)
