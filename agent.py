"""Step 5: the fraud analyst agent. Free and local, using Ollama.

An agent = a language model + tools + a loop:
  1. We send the question and a list of tools (the Python functions below).
  2. The model replies either with an answer or with "call this tool".
  3. We run the tool, hand the result back, and repeat until it answers.

Try it from the terminal:
    python agent.py "Why is account A03211 flagged?"
"""
import json
import os
import re
import sys

import duckdb

DB = "fraud.duckdb"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")   # use qwen2.5:3b on a low-RAM laptop
HIGH, REVIEW = 40, 20                                   # same tiers as the dashboard

SYSTEM_PROMPT = f"""You are a fraud analyst assistant for a trading platform.
Rules:
- ALWAYS use the tools to get facts. Never invent numbers, account ids or reasons.
- Risk tiers: score >= {HIGH} is high-risk, {REVIEW} to {HIGH - 1} is review, below {REVIEW} is clear.
- You explain and summarise. A human analyst makes every decision. Never say an account IS fraud;
  say it matches patterns that need review.
- Be concise: short paragraphs or a few bullet points, quote the key numbers.
- If a tool returns an error or nothing, say so plainly instead of guessing."""


# ------------------------------------------------------------------ helpers
def _query(sql, params=None):
    con = duckdb.connect(DB, read_only=True)   # read-only: the agent can never change data
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def _tier(score):
    return "high-risk" if score >= HIGH else "review" if score >= REVIEW else "clear"


# ------------------------------------------------------------------ tools
REASON_MEANINGS = {
    "QUICK_CASHOUT": "withdrew within 2 hours of the first deposit with 3 or fewer trades",
    "FAST_WITHDRAWAL_LOW_ACTIVITY": "withdrew within 10 hours of the first deposit with 15 or fewer trades and 500+ deposited",
    "SHARED_PAYMENT_OR_DEVICE": "3 or more accounts share one payment method or device",
    "PAIRED_PAYMENT": "exactly 2 accounts share one payment method",
    "AFFILIATE_PATTERN": "10 or more accounts from one affiliate have the same deposit total and trade count",
    "STRUCTURING": "10 or more deposits under 50 with 10 or fewer trades",
    "FREQUENT_SMALL_DEPOSITS": "5 to 9 deposits under 50 with 10 or fewer trades",
}


def get_account_summary(account_id: str) -> str:
    """Get the risk score, risk tier, reason codes and behaviour numbers for one account.

    Args:
        account_id: The account id, for example A03211.

    Returns:
        A text summary of the account.
    """
    df = _query(
        "SELECT r.*, a.country, a.signup_date FROM risk_scores r "
        "JOIN accounts a USING (account_id) WHERE account_id = ?",
        [account_id.strip().upper()],
    )
    if df.empty:
        return f"No account found with id {account_id}."
    r = df.iloc[0]
    mins = "no withdrawal yet" if r.mins_to_withdrawal >= 100000 else f"{int(r.mins_to_withdrawal)} minutes"
    meanings = "\n".join(f"- {c}: {REASON_MEANINGS[c]}" for c in r.reason_codes.split(", ") if c in REASON_MEANINGS)
    return (
        f"Account {r.account_id}: risk score {r.risk_score:.0f} ({_tier(r.risk_score)}).\n"
        f"Reason codes: {r.reason_codes}\nWhat the codes mean:\n{meanings}\n"
        f"Deposits: {int(r.n_deposits)} totalling {r.total_deposit:,.0f} "
        f"({int(r.n_small_deposits)} of them under 50). Trades: {int(r.n_trades)}. "
        f"Withdrawals: {int(r.n_withdrawals)}.\n"
        f"Time from first deposit to first withdrawal: {mins}.\n"
        f"Other accounts sharing its payment method: {int(r.shared_payment) - 1}. "
        f"Other accounts sharing its device: {int(r.shared_device) - 1}.\n"
        f"Affiliate: {r.affiliate_id} (other accounts from this affiliate with the same deposit total and trade count: "
        f"{int(r.signature_size) - 1}). "
        f"Country: {r.country}. Signed up: {str(r.signup_date)[:10]}."
    )


def get_linked_accounts(account_id: str) -> str:
    """List other accounts that share a payment method or device with this account.

    Args:
        account_id: The account id, for example A03211.

    Returns:
        A table of linked accounts with their risk scores.
    """
    df = _query(
        """
        SELECT b.account_id,
               CASE WHEN b.payment_method_id = a.payment_method_id AND b.device_id = a.device_id THEN 'payment+device'
                    WHEN b.payment_method_id = a.payment_method_id THEN 'payment' ELSE 'device' END AS shared,
               r.risk_score, r.reason_codes
        FROM accounts a
        JOIN accounts b ON (a.payment_method_id = b.payment_method_id OR a.device_id = b.device_id)
                       AND a.account_id <> b.account_id
        JOIN risk_scores r ON r.account_id = b.account_id
        WHERE a.account_id = ? ORDER BY r.risk_score DESC LIMIT 15
        """,
        [account_id.strip().upper()],
    )
    if df.empty:
        return f"Account {account_id} shares no payment method or device with other accounts."
    return f"{len(df)} linked accounts (max 15 shown):\n" + df.to_string(index=False)


def list_high_risk_accounts(min_score: int = HIGH, affiliate_id: str = "", limit: int = 10) -> str:
    """List the highest-risk accounts, optionally for one affiliate.

    Args:
        min_score: Only accounts with at least this risk score (default 40).
        affiliate_id: Optional affiliate id such as AFF99. Leave empty for all affiliates.
        limit: How many accounts to return (default 10, max 25).

    Returns:
        A table of accounts sorted by risk score.
    """
    limit = max(1, min(int(limit), 25))
    sql = "SELECT account_id, risk_score, reason_codes, affiliate_id FROM risk_scores WHERE risk_score >= ?"
    params = [float(min_score)]
    if affiliate_id:
        sql += " AND affiliate_id = ?"
        params.append(affiliate_id.strip().upper())
    df = _query(sql + f" ORDER BY risk_score DESC LIMIT {limit}", params)
    return "No accounts match." if df.empty else df.to_string(index=False)


def get_portfolio_stats() -> str:
    """Get overall numbers: total accounts, how many are high-risk or in review, top reason codes and top affiliates.

    Returns:
        A text summary of the whole portfolio.
    """
    df = _query("SELECT account_id, affiliate_id, risk_score, reason_codes FROM risk_scores")
    high = int((df.risk_score >= HIGH).sum())
    review = int(((df.risk_score >= REVIEW) & (df.risk_score < HIGH)).sum())
    codes = (df[df.risk_score >= REVIEW].reason_codes.str.split(", ").explode().value_counts().head(6))
    aff = (df.assign(flagged=df.risk_score >= REVIEW).groupby("affiliate_id")
           .agg(accounts=("account_id", "count"), flagged=("flagged", "sum")))
    aff["flag_rate"] = (aff.flagged / aff.accounts).round(2)
    aff = aff[aff.accounts >= 10].sort_values("flag_rate", ascending=False).head(5)
    return (f"{len(df)} accounts. High-risk: {high}. Review queue: {review}.\n"
            f"Top reason codes among flagged accounts:\n{codes.to_string()}\n"
            f"Affiliates with the highest flag rate:\n{aff.to_string()}")


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|call|set)\b", re.I)
_LABELS = re.compile(r"\b(is_fraud|fraud_type|segment)\b", re.I)


def run_sql(query: str) -> str:
    """Run a read-only SQL SELECT query. Tables: accounts(account_id, signup_date, country, affiliate_id,
    device_id, payment_method_id), transactions(txn_id, account_id, ts, type, amount) where type is
    'deposit', 'trade' or 'withdrawal', and risk_scores(account_id, affiliate_id, risk_score, reason_codes,
    n_deposits, total_deposit, n_trades, n_withdrawals, n_small_deposits, mins_to_withdrawal,
    shared_payment, shared_device, signature_size). Max 25 rows are returned.

    Args:
        query: A single SELECT statement.

    Returns:
        The result rows as text, or an error message.
    """
    q = query.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", q, re.I):
        return "Query rejected: only SELECT queries are allowed."
    if ";" in q or _FORBIDDEN.search(q):
        return "Query rejected: only a single read-only SELECT is allowed."
    if _LABELS.search(q):
        return "Query rejected: those columns are not available."
    try:
        df = _query(f"SELECT * FROM ({q}) LIMIT 25")
    except Exception as e:
        return f"SQL error: {e}"
    return "No rows." if df.empty else df.to_string(index=False)


TOOLS = [get_account_summary, get_linked_accounts, list_high_risk_accounts, get_portfolio_stats, run_sql]
TOOL_FUNCS = {f.__name__: f for f in TOOLS}


# ------------------------------------------------------------------ the agent loop
def run_agent(question, history=None, model=MODEL, max_steps=6):
    """Answer a question. Returns (answer_text, trace). The trace lists every tool the agent used."""
    import ollama  # imported here so the tools above can be tested without Ollama installed

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history or [])
    messages.append({"role": "user", "content": question})
    trace = []

    for _ in range(max_steps):
        response = ollama.chat(model=model, messages=messages, tools=TOOLS)
        msg = response.message
        messages.append(msg)
        if not msg.tool_calls:                      # no tool requested: this is the final answer
            return msg.content, trace
        for call in msg.tool_calls:                 # run each requested tool and feed the result back
            name, args = call.function.name, dict(call.function.arguments)
            func = TOOL_FUNCS.get(name)
            try:
                result = func(**args) if func else f"Unknown tool: {name}"
            except Exception as e:
                result = f"Tool error: {e}"
            trace.append({"tool": name, "args": args, "result": str(result)[:1500]})
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})

    return "I could not finish within the step limit. Try a more specific question.", trace


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Give me an overview of the portfolio and the riskiest affiliate."
    answer, steps = run_agent(q)
    for s in steps:
        print(f"[tool] {s['tool']}({s['args']})")
    print("\n" + answer)
