"""Step 3: score every account for fraud risk.

Combines (a) transparent rules with reason codes and (b) an Isolation Forest
anomaly score. Writes a risk_scores table to fraud.duckdb and prints how well
it caught the planted patterns.
"""
import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

con = duckdb.connect("fraud.duckdb")

# ---------- Feature engineering (plain SQL: your analyst superpower) ----------
features = con.execute("""
WITH t AS (
    SELECT account_id,
           COUNT(*) FILTER (WHERE type = 'deposit')              AS n_deposits,
           COALESCE(SUM(amount) FILTER (WHERE type = 'deposit'), 0) AS total_deposit,
           COUNT(*) FILTER (WHERE type = 'trade')                AS n_trades,
           COUNT(*) FILTER (WHERE type = 'withdrawal')           AS n_withdrawals,
           COUNT(*) FILTER (WHERE type = 'deposit' AND amount < 50) AS n_small_deposits,
           MIN(ts) FILTER (WHERE type = 'deposit')               AS first_deposit,
           MIN(ts) FILTER (WHERE type = 'withdrawal')            AS first_withdrawal
    FROM transactions GROUP BY account_id
),
pay AS (SELECT payment_method_id, COUNT(*) AS shared_payment FROM accounts GROUP BY 1),
dev AS (SELECT device_id, COUNT(*) AS shared_device FROM accounts GROUP BY 1)
SELECT a.account_id, a.affiliate_id, a.is_fraud, a.fraud_type, a.segment,
       COALESCE(t.n_deposits, 0) n_deposits, COALESCE(t.total_deposit, 0) total_deposit,
       COALESCE(t.n_trades, 0) n_trades, COALESCE(t.n_withdrawals, 0) n_withdrawals,
       COALESCE(t.n_small_deposits, 0) n_small_deposits,
       COALESCE(date_diff('minute', t.first_deposit, t.first_withdrawal), 100000) AS mins_to_withdrawal,
       pay.shared_payment, dev.shared_device
FROM accounts a
LEFT JOIN t USING (account_id)
JOIN pay USING (payment_method_id)
JOIN dev USING (device_id)
""").df()

# Accounts behaving identically within the same affiliate (same deposit total + same trade count)
features["signature_size"] = features.groupby(
    ["affiliate_id", features.total_deposit.round(0), "n_trades"]
)["account_id"].transform("count")
features["trades_per_deposit"] = features.n_trades / features.n_deposits.clip(lower=1)

# ---------- Rules with reason codes and weights ----------
# Strong rules = clear red flags (45 pts). Soft rules = weaker hints (20 pts).
rules = [
    ("QUICK_CASHOUT", 45,
     (features.mins_to_withdrawal <= 120) & (features.n_trades <= 3)),
    ("FAST_WITHDRAWAL_LOW_ACTIVITY", 20,
     (features.mins_to_withdrawal <= 600) & (features.n_trades <= 15) & (features.total_deposit >= 500)
     & (features.mins_to_withdrawal > 120)),
    ("SHARED_PAYMENT_OR_DEVICE", 45,
     (features.shared_payment >= 3) | (features.shared_device >= 3)),
    ("PAIRED_PAYMENT", 20,
     (features.shared_payment == 2)),
    ("AFFILIATE_PATTERN", 45,
     features.signature_size >= 10),
    ("STRUCTURING", 45,
     (features.n_small_deposits >= 10) & (features.n_trades <= 10)),
    ("FREQUENT_SMALL_DEPOSITS", 20,
     (features.n_small_deposits >= 5) & (features.n_small_deposits < 10) & (features.n_trades <= 10)),
]
rule_points = np.zeros(len(features))
reasons = [[] for _ in range(len(features))]
for name, points, mask in rules:
    m = mask.to_numpy()
    rule_points += m * points
    for i in np.where(m)[0]:
        reasons[i].append(name)
features["rule_score"] = np.minimum(rule_points, 100)

# ---------- Isolation Forest (unsupervised anomaly score) ----------
cols = ["n_deposits", "total_deposit", "n_trades", "n_small_deposits",
        "mins_to_withdrawal", "shared_payment", "shared_device", "signature_size", "trades_per_deposit"]
iso = IsolationForest(n_estimators=200, contamination=0.08, random_state=42)
iso.fit(features[cols])
raw = -iso.score_samples(features[cols])
features["anomaly_score"] = 100 * (raw - raw.min()) / (raw.max() - raw.min())

# ---------- Final risk score ----------
features["risk_score"] = np.minimum(features.rule_score + 0.3 * features.anomaly_score, 100).round(1)
features["reason_codes"] = [", ".join(r) if r else "none" for r in reasons]
THRESHOLD = 20   # the analyst-tunable dial: lower = catch more fraud, more false alarms
HIGH = 40        # at or above this: high-risk (escalate). Between THRESHOLD and HIGH: review queue
features["flagged"] = features.risk_score >= THRESHOLD
features["tier"] = np.where(features.risk_score >= HIGH, "high", np.where(features.flagged, "review", "clear"))

con.execute("CREATE OR REPLACE TABLE risk_scores AS SELECT * FROM features")

# ---------- How well did we do? (we know the answer key) ----------
def metrics(flag):
    tp = int((flag & features.is_fraud).sum())
    fp = int((flag & ~features.is_fraud).sum())
    fn = int((~flag & features.is_fraud).sum())
    return tp, fp, fn, tp / max(tp + fp, 1), tp / max(tp + fn, 1)

tp, fp, fn, precision, recall = metrics(features.flagged)
print(f"Flagged: {int(features.flagged.sum())} of {len(features)} accounts (threshold {THRESHOLD})")
print(f"Precision: {precision:.1%}   Recall: {recall:.1%}   (TP={tp}, FP={fp}, FN={fn})")

_, _, _, hp, hr = metrics(features.risk_score >= HIGH)
print(f"High-risk tier only (score >= {HIGH}): {int((features.risk_score >= HIGH).sum())} accounts, precision {hp:.0%}, recall {hr:.0%}")
print(f"Review queue (score {THRESHOLD}-{HIGH - 1}): {int((features.tier == 'review').sum())} accounts for a human analyst")

print("\nThreshold trade-off (the decision an analyst owns):")
print("threshold  flagged  precision  recall")
for t in [20, 30, 40, 50, 60]:
    _, _, _, p, r = metrics(features.risk_score >= t)
    print(f"{t:>9}  {int((features.risk_score >= t).sum()):>7}  {p:>8.0%}  {r:>6.0%}")

print("\nShare of accounts flagged, by segment:")
print(features.groupby("segment").flagged.mean().sort_values(ascending=False).map("{:.0%}".format).to_string())

print("\nTop 10 highest-risk accounts:")
print(features.sort_values("risk_score", ascending=False)
      [["account_id", "risk_score", "reason_codes", "segment"]].head(10).to_string(index=False))
con.close()
