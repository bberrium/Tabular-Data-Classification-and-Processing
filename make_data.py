"""Generate a synthetic, messy, imbalanced customer-churn dataset.

It mimics the problems of real tabular data: missing values, mixed numeric and
categorical columns, skewed features and a rare positive class (~5%).
Swap it for a real dataset (see README) once you've seen the pipeline run.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_dataset(n=40000, pos_rate=0.05, seed=42):
    rng = np.random.default_rng(seed)

    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.40, 0.15])
    payment = rng.choice(["card", "bank_transfer", "e-check", "mail"], n, p=[0.35, 0.30, 0.25, 0.10])
    region = rng.choice([f"region_{i}" for i in range(1, 9)], n)
    tenure = np.clip(rng.gamma(2.0, 14.0, n), 1, 72).round()
    age = np.clip(rng.normal(42, 14, n), 18, 90).round()
    monthly = np.clip(rng.normal(65, 25, n) + 18 * (internet == "fiber") - 25 * (internet == "none"), 15, 150).round(2)
    total = (monthly * tenure * rng.uniform(0.9, 1.1, n)).round(2)
    support_calls = rng.poisson(1.2, n)
    late_payments = rng.poisson(0.5, n)
    data_gb = rng.lognormal(3.0, 1.1, n).round(1)  # heavy right tail, large scale

    logit = (
        1.3 * (contract == "month-to-month")
        - 0.9 * (contract == "two-year")
        + 0.35 * support_calls
        + 0.55 * late_payments
        - 0.035 * tenure
        + 0.012 * (monthly - 65)
        + 0.9 * ((internet == "fiber") & (contract == "month-to-month"))  # interaction
        + 0.5 * (payment == "e-check")
        + 1.4 * ((support_calls >= 3) & (tenure < 12))  # non-linear rule
        + 1.0 * ((monthly > 90) & (internet == "fiber") & (late_payments >= 1))  # 3-way interaction
        + 0.9 * (data_gb > 60) * (contract != "two-year")  # threshold effect
        - 0.008 * (age - 42)
        + rng.normal(0, 0.3, n)
    )

    # Shift the intercept so that the positive rate matches pos_rate.
    lo, hi = -15.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(logit + mid)))).mean()
        lo, hi = (mid, hi) if rate < pos_rate else (lo, mid)
    p = 1 / (1 + np.exp(-(logit + mid)))
    churned = (rng.uniform(size=n) < p).astype(int)

    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": age, "tenure_months": tenure, "monthly_charges": monthly, "total_charges": total,
        "support_calls": support_calls, "late_payments": late_payments, "avg_data_gb": data_gb,
        "contract": contract, "internet_service": internet, "payment_method": payment, "region": region,
        "churned": churned,
    })

    # Inject missing values (some informative: heavy data users are more often missing).
    for col, frac in [("age", 0.08), 
                      ("avg_data_gb", 0.12), 
                      ("total_charges", 0.03),
                      ("payment_method", 0.02), 
                      ("region", 0.01)]:
        df.loc[rng.uniform(size=n) < frac, col] = np.nan

    missing_probability = np.where(data_gb > 60,0.30,0.08)
    mask = rng.uniform(size=n) < missing_probability
    df.loc[mask, "avg_data_gb"] = np.nan
    
    return df
    
    



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/customers.csv")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--pos-rate", type=float, default=0.05)
    args = ap.parse_args()
    df = make_dataset(args.n, args.pos_rate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {df.shape}, positive rate {df['churned'].mean():.2%}")