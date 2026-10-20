"""Step 2: generate a synthetic trading-platform dataset with planted fraud patterns.

Output: fraud.duckdb with two tables
  accounts(account_id, signup_date, country, affiliate_id, device_id, payment_method_id, is_fraud, fraud_type, segment)
  transactions(txn_id, account_id, ts, type, amount)

All data is synthetic. is_fraud / fraud_type are the hidden answer key, used only to
check how well the detector works.
"""
import numpy as np
import pandas as pd
import duckdb

rng = np.random.default_rng(42)
START = pd.Timestamp("2026-07-01")
DAYS = 90
COUNTRIES = ["AE", "ZA", "NG", "IN", "BR", "ID", "KE", "GH", "MY", "TH"]

accounts, txns = [], []
_txn = [0]


def add_txn(acc, ts, kind, amount):
    _txn[0] += 1
    txns.append((_txn[0], acc, ts, kind, round(float(amount), 2)))


def new_account(i, signup, affiliate, device, payment, fraud_type=None, segment=None):
    accounts.append({
        "account_id": f"A{i:05d}",
        "signup_date": signup,
        "country": rng.choice(COUNTRIES),
        "affiliate_id": affiliate,
        "device_id": device,
        "payment_method_id": payment,
        "is_fraud": fraud_type is not None,
        "fraud_type": fraud_type or "none",
        "segment": segment or fraud_type or "normal",
    })
    return f"A{i:05d}"


idx = 0

# ---------- 1. Normal accounts ----------
for _ in range(3000):
    idx += 1
    signup = START + pd.Timedelta(days=int(rng.integers(0, 60)), hours=int(rng.integers(0, 24)))
    aff = f"AFF{rng.integers(1, 21):02d}"
    acc = new_account(idx, signup, aff, f"D{idx:05d}", f"P{idx:05d}")
    for _ in range(int(rng.integers(1, 5))):
        add_txn(acc, signup + pd.Timedelta(days=float(rng.uniform(0, 30))), "deposit", rng.lognormal(5.0, 0.8))
    for _ in range(int(rng.integers(15, 250))):
        add_txn(acc, signup + pd.Timedelta(days=float(rng.uniform(0, 80))), "trade", rng.lognormal(3.5, 0.9))
    if rng.random() < 0.35:
        add_txn(acc, signup + pd.Timedelta(days=float(rng.uniform(5, 85))), "withdrawal", rng.lognormal(5.0, 0.9))

# ---------- 2. Fraud: quick cash-out (deposit, almost no trading, fast withdrawal) ----------
for _ in range(60):
    idx += 1
    signup = START + pd.Timedelta(days=int(rng.integers(0, 80)), hours=int(rng.integers(0, 24)))
    acc = new_account(idx, signup, f"AFF{rng.integers(1, 21):02d}", f"D{idx:05d}", f"P{idx:05d}", "quick_cashout")
    dep = signup + pd.Timedelta(minutes=int(rng.integers(5, 60)))
    amt = rng.uniform(500, 3000)
    add_txn(acc, dep, "deposit", amt)
    for _ in range(int(rng.integers(0, 3))):
        add_txn(acc, dep + pd.Timedelta(minutes=int(rng.integers(2, 20))), "trade", rng.uniform(5, 30))
    add_txn(acc, dep + pd.Timedelta(minutes=int(rng.integers(20, 90))), "withdrawal", amt * rng.uniform(0.9, 0.99))

# ---------- 3. Fraud: shared payment / device ring ----------
for ring in range(8):
    shared_pay, shared_dev = f"PRING{ring:02d}", f"DRING{ring:02d}"
    ring_start = START + pd.Timedelta(days=int(rng.integers(0, 70)))
    for _ in range(int(rng.integers(5, 9))):
        idx += 1
        signup = ring_start + pd.Timedelta(hours=int(rng.integers(0, 72)))
        acc = new_account(idx, signup, f"AFF{rng.integers(1, 21):02d}", shared_dev, shared_pay, "shared_ring")
        add_txn(acc, signup + pd.Timedelta(hours=1), "deposit", rng.uniform(200, 800))
        for _ in range(int(rng.integers(1, 6))):
            add_txn(acc, signup + pd.Timedelta(hours=float(rng.uniform(2, 20))), "trade", rng.uniform(10, 60))
        add_txn(acc, signup + pd.Timedelta(days=float(rng.uniform(1, 3))), "withdrawal", rng.uniform(150, 700))

# ---------- 4. Fraud: affiliate farm (identical behaviour, one affiliate) ----------
farm_start = START + pd.Timedelta(days=40)
for _ in range(70):
    idx += 1
    signup = farm_start + pd.Timedelta(hours=int(rng.integers(0, 96)))
    acc = new_account(idx, signup, "AFF99", f"D{idx:05d}", f"P{idx:05d}", "affiliate_farm")
    add_txn(acc, signup + pd.Timedelta(minutes=30), "deposit", 100.0)
    for k in range(3):
        add_txn(acc, signup + pd.Timedelta(hours=2 + k), "trade", 10.0)

# ---------- 5. Fraud: structuring (many small deposits, then withdraw) ----------
for _ in range(40):
    idx += 1
    signup = START + pd.Timedelta(days=int(rng.integers(0, 80)))
    acc = new_account(idx, signup, f"AFF{rng.integers(1, 21):02d}", f"D{idx:05d}", f"P{idx:05d}", "structuring")
    t = signup + pd.Timedelta(hours=1)
    total = 0.0
    for _ in range(int(rng.integers(15, 35))):
        a = rng.uniform(40, 49)
        total += a
        t += pd.Timedelta(minutes=int(rng.integers(5, 40)))
        add_txn(acc, t, "deposit", a)
    add_txn(acc, t + pd.Timedelta(hours=3), "withdrawal", total * 0.95)


# =====================================================================
# REALISM UPGRADE: legitimate look-alikes (cause false positives) and
# disguised fraud (causes misses). Real data is messy, so ours should be too.
# =====================================================================
def normal_behaviour(acc, signup):
    for _ in range(int(rng.integers(1, 4))):
        add_txn(acc, signup + pd.Timedelta(days=float(rng.uniform(0, 30))), "deposit", rng.lognormal(5.0, 0.8))
    for _ in range(int(rng.integers(15, 150))):
        add_txn(acc, signup + pd.Timedelta(days=float(rng.uniform(0, 80))), "trade", rng.lognormal(3.5, 0.9))
    if rng.random() < 0.3:
        add_txn(acc, signup + pd.Timedelta(days=float(rng.uniform(5, 85))), "withdrawal", rng.lognormal(5.0, 0.9))


def rand_signup(max_day=75):
    return START + pd.Timedelta(days=int(rng.integers(0, max_day)), hours=int(rng.integers(0, 24)))


# ---------- 6. LEGIT look-alikes ----------
# 6a. Impatient genuine traders: deposit, a few trades, quick withdrawal
for _ in range(40):
    idx += 1
    signup = rand_signup()
    acc = new_account(idx, signup, f"AFF{rng.integers(1, 21):02d}", f"D{idx:05d}", f"P{idx:05d}",
                      segment="legit_fast_withdrawer")
    dep = signup + pd.Timedelta(minutes=30)
    amt = rng.uniform(100, 1500)
    add_txn(acc, dep, "deposit", amt)
    for _ in range(int(rng.integers(0, 4))):
        add_txn(acc, dep + pd.Timedelta(minutes=int(rng.integers(2, 25))), "trade", rng.uniform(5, 40))
    add_txn(acc, dep + pd.Timedelta(minutes=int(rng.integers(40, 110))), "withdrawal", amt * rng.uniform(0.85, 1.0))

# 6b. Households / internet cafes: 3-4 genuine accounts on one device
for h in range(25):
    dev = f"DHOME{h:02d}"
    for _ in range(int(rng.integers(3, 5))):
        idx += 1
        signup = rand_signup()
        acc = new_account(idx, signup, f"AFF{rng.integers(1, 21):02d}", dev, f"P{idx:05d}",
                          segment="legit_shared_device")
        normal_behaviour(acc, signup)

# 6c. Micro-depositors: many small deposits but genuinely active traders
for _ in range(25):
    idx += 1
    signup = rand_signup(50)
    acc = new_account(idx, signup, f"AFF{rng.integers(1, 21):02d}", f"D{idx:05d}", f"P{idx:05d}",
                      segment="legit_micro_depositor")
    for _ in range(int(rng.integers(10, 20))):
        add_txn(acc, signup + pd.Timedelta(days=float(rng.uniform(0, 30))), "deposit", rng.uniform(10, 45))
    for _ in range(int(rng.integers(40, 180))):
        add_txn(acc, signup + pd.Timedelta(days=float(rng.uniform(0, 80))), "trade", rng.lognormal(3.0, 0.7))

# 6d. Genuine promo campaign: one affiliate, similar new users (look like a farm)
for _ in range(20):
    idx += 1
    signup = START + pd.Timedelta(days=20, hours=int(rng.integers(0, 120)))
    acc = new_account(idx, signup, "AFF07", f"D{idx:05d}", f"P{idx:05d}", segment="legit_promo_campaign")
    add_txn(acc, signup + pd.Timedelta(hours=1), "deposit", 100.0)
    for k in range(5):
        add_txn(acc, signup + pd.Timedelta(days=float(rng.uniform(0.2, 10))), "trade", rng.uniform(8, 25))

# ---------- 7. DISGUISED fraud ----------
# 7a. Stealth cash-out: makes some trades and waits a few hours
for _ in range(30):
    idx += 1
    signup = rand_signup()
    acc = new_account(idx, signup, f"AFF{rng.integers(1, 21):02d}", f"D{idx:05d}", f"P{idx:05d}", "quick_cashout_stealth")
    dep = signup + pd.Timedelta(minutes=30)
    amt = rng.uniform(500, 3000)
    add_txn(acc, dep, "deposit", amt)
    for _ in range(int(rng.integers(6, 15))):
        add_txn(acc, dep + pd.Timedelta(minutes=int(rng.integers(5, 150))), "trade", rng.uniform(5, 40))
    add_txn(acc, dep + pd.Timedelta(hours=float(rng.uniform(3, 8))), "withdrawal", amt * rng.uniform(0.9, 0.99))

# 7b. Paired accounts sharing only a payment method (stays under the 3+ rule)
for p in range(10):
    shared_pay = f"PPAIR{p:02d}"
    for _ in range(2):
        idx += 1
        signup = rand_signup()
        acc = new_account(idx, signup, f"AFF{rng.integers(1, 21):02d}", f"D{idx:05d}", shared_pay, "shared_pair")
        add_txn(acc, signup + pd.Timedelta(hours=1), "deposit", rng.uniform(200, 800))
        for _ in range(int(rng.integers(2, 8))):
            add_txn(acc, signup + pd.Timedelta(hours=float(rng.uniform(2, 30))), "trade", rng.uniform(10, 60))
        add_txn(acc, signup + pd.Timedelta(days=float(rng.uniform(1, 3))), "withdrawal", rng.uniform(150, 700))

# 7c. Light structuring: fewer small deposits
for _ in range(15):
    idx += 1
    signup = rand_signup()
    acc = new_account(idx, signup, f"AFF{rng.integers(1, 21):02d}", f"D{idx:05d}", f"P{idx:05d}", "structuring_light")
    t = signup + pd.Timedelta(hours=1)
    total = 0.0
    for _ in range(int(rng.integers(6, 10))):
        a = rng.uniform(40, 49)
        total += a
        t += pd.Timedelta(minutes=int(rng.integers(10, 90)))
        add_txn(acc, t, "deposit", a)
    for _ in range(int(rng.integers(0, 4))):
        add_txn(acc, t + pd.Timedelta(minutes=int(rng.integers(5, 60))), "trade", rng.uniform(5, 20))
    add_txn(acc, t + pd.Timedelta(hours=6), "withdrawal", total * 0.95)

accounts_df = pd.DataFrame(accounts)
txn_df = pd.DataFrame(txns, columns=["txn_id", "account_id", "ts", "type", "amount"])

con = duckdb.connect("fraud.duckdb")
con.execute("CREATE OR REPLACE TABLE accounts AS SELECT * FROM accounts_df")
con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM txn_df")
con.close()

print(f"Accounts: {len(accounts_df):,}  (fraud: {int(accounts_df.is_fraud.sum())})")
print(f"Transactions: {len(txn_df):,}")
print(accounts_df.segment.value_counts().to_string())
print("Saved to fraud.duckdb")
