"""Fraud Analyst Copilot - dashboard (Step 4).

Run with:  streamlit run app.py
Reads fraud.duckdb, which is built by generate_data.py and detect.py.
All data is synthetic.
"""
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from agent import MODEL, run_agent

st.set_page_config(page_title="Fraud Analyst Copilot", page_icon="🛡️", layout="wide")
DB = "fraud.duckdb"


@st.cache_data
def load_scores() -> pd.DataFrame:
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("SELECT * FROM risk_scores").df()
    con.close()
    return df


@st.cache_data
def load_transactions(account_id: str) -> pd.DataFrame:
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(
        "SELECT ts, type, amount FROM transactions WHERE account_id = ? ORDER BY ts", [account_id]
    ).df()
    con.close()
    return df


def precision_recall(df: pd.DataFrame, flag: pd.Series):
    tp = int((flag & df.is_fraud).sum())
    fp = int((flag & ~df.is_fraud).sum())
    fn = int((~flag & df.is_fraud).sum())
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), tp, fp, fn


try:
    scores = load_scores()
except Exception:
    st.error("Could not find fraud.duckdb. Run generate_data.py and then detect.py first.")
    st.stop()

# ---------------- Sidebar controls ----------------
st.sidebar.header("Risk thresholds")
review_cut = st.sidebar.slider("Flag for review at score >=", 0, 100, 20, 5)
high_cut = st.sidebar.slider("High-risk (escalate) at score >=", 0, 100, 40, 5)
if high_cut < review_cut:
    high_cut = review_cut
st.sidebar.caption("Lower thresholds catch more fraud but send more genuine customers to review.")

scores["flagged"] = scores.risk_score >= review_cut
scores["tier"] = "clear"
scores.loc[scores.flagged, "tier"] = "review"
scores.loc[scores.risk_score >= high_cut, "tier"] = "high"

prec, rec, tp, fp, fn = precision_recall(scores, scores.flagged)

# ---------------- Header and KPIs ----------------
st.title("🛡️ Fraud Analyst Copilot")
st.caption("Synthetic data only. Precision and recall are measured against planted patterns, so real-world results would be lower.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Accounts", f"{len(scores):,}")
k2.metric("Flagged", f"{int(scores.flagged.sum()):,}", f"{scores.flagged.mean():.1%} of accounts", delta_color="off")
k3.metric("High-risk", f"{int((scores.tier == 'high').sum()):,}")
k4.metric("Precision", f"{prec:.0%}", help="Of flagged accounts, how many were really fraud")
k5.metric("Recall", f"{rec:.0%}", help="Of all fraud accounts, how many we caught")

tab1, tab2, tab3 = st.tabs(["Overview", "Case review", "Ask the agent"])

# ---------------- Overview ----------------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        by_seg = (scores.groupby("segment").agg(accounts=("account_id", "count"), flagged=("flagged", "sum"))
                  .assign(flag_rate=lambda d: d.flagged / d.accounts).reset_index()
                  .sort_values("flag_rate", ascending=False))
        by_seg["kind"] = by_seg.segment.apply(
            lambda s: "legitimate" if s.startswith("legit_") or s == "normal" else "fraud")
        fig = px.bar(by_seg, x="flag_rate", y="segment", color="kind", orientation="h",
                     color_discrete_map={"fraud": "#d62728", "legitimate": "#1f77b4"},
                     title="Share of accounts flagged, by segment")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%", height=420)
        st.plotly_chart(fig, width="stretch")
        st.caption("Blue bars are genuine customers: these are the false alarms.")

    with c2:
        sweep = []
        for t in range(0, 101, 5):
            p, r, *_ = precision_recall(scores, scores.risk_score >= t)
            sweep.append({"threshold": t, "Precision": p, "Recall": r})
        sweep_df = pd.DataFrame(sweep).melt("threshold", var_name="metric", value_name="value")
        fig = px.line(sweep_df, x="threshold", y="value", color="metric", markers=True,
                      title="Precision vs recall as the threshold moves")
        fig.add_vline(x=review_cut, line_dash="dash", annotation_text="current")
        fig.update_layout(yaxis_tickformat=".0%", height=420)
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        reasons = (scores[scores.flagged].assign(reason=lambda d: d.reason_codes.str.split(", "))
                   .explode("reason"))
        reasons = reasons[reasons.reason != "none"].reason.value_counts().reset_index()
        reasons.columns = ["reason", "accounts"]
        fig = px.bar(reasons, x="accounts", y="reason", orientation="h", title="Top risk drivers (flagged accounts)")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380)
        st.plotly_chart(fig, width="stretch")
    with c4:
        aff = (scores.groupby("affiliate_id").agg(accounts=("account_id", "count"), flagged=("flagged", "sum"))
               .assign(flag_rate=lambda d: d.flagged / d.accounts).reset_index())
        aff = aff[aff.accounts >= 10].sort_values("flag_rate", ascending=False).head(10)
        fig = px.bar(aff, x="flag_rate", y="affiliate_id", orientation="h", title="Affiliates with the highest flag rate")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%", height=380)
        st.plotly_chart(fig, width="stretch")

# ---------------- Case review ----------------
with tab2:
    st.subheader("Flagged accounts")
    f1, f2 = st.columns(2)
    tier_filter = f1.multiselect("Tier", ["high", "review"], default=["high", "review"])
    all_reasons = sorted({r for s in scores.reason_codes for r in s.split(", ") if r != "none"})
    reason_filter = f2.multiselect("Reason code", all_reasons)

    view = scores[scores.tier.isin(tier_filter)]
    if reason_filter:
        view = view[view.reason_codes.apply(lambda s: any(r in s for r in reason_filter))]
    view = view.sort_values("risk_score", ascending=False)
    st.dataframe(view[["account_id", "risk_score", "tier", "reason_codes", "affiliate_id",
                       "total_deposit", "n_trades", "mins_to_withdrawal"]],
                 width="stretch", hide_index=True, height=300)

    if len(view):
        st.subheader("Account drill-down")
        acc_id = st.selectbox("Account", view.account_id.tolist())
        row = scores[scores.account_id == acc_id].iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Risk score", f"{row.risk_score:.0f}")
        d2.metric("Deposits", f"{int(row.n_deposits)} (total {row.total_deposit:,.0f})")
        d3.metric("Trades", f"{int(row.n_trades)}")
        d4.metric("Accounts sharing payment", f"{int(row.shared_payment)}")
        st.write(f"**Reason codes:** {row.reason_codes}")

        tx = load_transactions(acc_id)
        fig = px.scatter(tx, x="ts", y="amount", color="type", title="Transaction timeline",
                         color_discrete_map={"deposit": "#2ca02c", "trade": "#7f7f7f", "withdrawal": "#d62728"})
        st.plotly_chart(fig, width="stretch")
        with st.expander("Raw transactions"):
            st.dataframe(tx, width="stretch", hide_index=True)

# ---------------- Ask the agent ----------------
with tab3:
    st.subheader("Ask the agent")
    st.caption(f"Runs locally with Ollama (model: {MODEL}). It explains and summarises; a human analyst decides.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    def ask(question: str):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[-6:]]
        with st.spinner("The agent is working..."):
            try:
                answer, trace = run_agent(question, history=history)
            except Exception as e:
                answer, trace = (f"Could not reach the local model ({e}). Is Ollama running and the model "
                                 f"pulled? Try: ollama pull {MODEL}"), []
        st.session_state.chat.append({"role": "user", "content": question})
        st.session_state.chat.append({"role": "assistant", "content": answer, "trace": trace})

    q1, q2 = st.columns([2, 1])
    case_options = scores[scores.tier == "high"].sort_values("risk_score", ascending=False).account_id.tolist()
    case_acc = q1.selectbox("Draft a case summary for", case_options or ["(no high-risk accounts)"])
    if q2.button("Draft escalation summary", disabled=not case_options):
        ask(f"Draft a short escalation case summary for account {case_acc}: overview, key evidence with numbers, "
            f"any linked accounts, and a suggested next step for a human analyst.")

    typed = st.chat_input("Ask about accounts, affiliates or patterns, e.g. 'Which affiliate looks riskiest?'")
    if typed:
        ask(typed)

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("trace"):
                with st.expander(f"Tools used ({len(m['trace'])})"):
                    for t in m["trace"]:
                        st.code(f"{t['tool']}({t['args']})\n\n{t['result']}")
