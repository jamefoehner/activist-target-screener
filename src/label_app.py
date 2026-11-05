"""
LLM evaluation project, step 2: hand-label activist demands in Item 4 text.

Run:   python3 -m pip install streamlit pandas
       streamlit run label_app.py

Pass 1 labels every filing in data/label_sample.csv.
Pass 2 (a week later) re-labels a blind subset in a different order, without
showing your pass-1 answers, so you can measure how consistent your own labels are.
Labels save to data/labels_pass1.csv / data/labels_pass2.csv after every click on Save.
Read LABELING_GUIDE.md before starting.
"""

import random
from pathlib import Path

import pandas as pd
import streamlit as st

SAMPLE = Path("data/label_sample.csv")
CATEGORIES = {
    "board_seats": "Board seats / board representation",
    "sale_or_review": "Sale of the company or a strategic review",
    "breakup_spin": "Breakup, spin-off, or divestiture",
    "capital_return": "Buybacks, dividends, or other capital return",
    "operations_costs": "Operational improvements or cost cuts",
    "management_change": "CEO / management change",
    "governance": "Governance changes (declassify board, bylaws, etc.)",
    "oppose_deal": "Oppose a pending deal or demand better terms",
    "capital_discipline": "Capital discipline: stop acquisitions, cut capex, reduce debt, improve returns on capital",
    "engagement_only": "Engagement only: no specific demand stated",
}
TONES = ["cooperative", "neutral", "confrontational"]
PASS2_SIZE = 20

st.set_page_config(page_title="13D demand labeling", layout="wide")


@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE)


def labels_path(p):
    return Path(f"data/labels_pass{p}.csv")


def load_labels(p):
    f = labels_path(p)
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["item_id"])


def order_for(p, ids):
    if p == 1:
        return list(ids)
    # pass 2 and the second labeler use the same 20 items, drawn only from filings
    # that pass 1 did not mark "EXCLUDE" (non-campaigns can't test demand labels)
    pass1 = load_labels(1)
    excluded = set()
    if len(pass1) and "notes" in pass1:
        excluded = set(pass1.loc[pass1["notes"].fillna("").str.upper().str.startswith("EXCLUDE"), "item_id"])
    eligible = [i for i in ids if i not in excluded]
    rng = random.Random(2027)
    sub = rng.sample(eligible, min(PASS2_SIZE, len(eligible)))
    rng.shuffle(sub)
    return sub


sample = load_sample()
st.sidebar.title("13D demand labeling")
p = st.sidebar.radio("Labeling pass", [1, 2, "B"],
                     format_func=lambda x: {1: "Pass 1", 2: "Pass 2 (self re-label)", "B": "Second labeler"}[x],
                     help="Pass 2: your blind re-label a week later. Second labeler: a different person labels the same 20 filings.")
order = order_for(p, sample["item_id"])
labels = load_labels(p)
done = set(labels["item_id"].astype(int)) if len(labels) else set()
st.sidebar.progress(len(done & set(order)) / len(order), text=f"{len(done & set(order))} of {len(order)} labeled")

if "pos" not in st.session_state or st.session_state.get("active_pass") != p:
    st.session_state["active_pass"] = p
    todo = [i for i, x in enumerate(order) if x not in done]
    st.session_state.pos = todo[0] if todo else 0

col_prev, col_next, col_jump = st.sidebar.columns(3)
if col_prev.button("◀ Prev"):
    st.session_state.pos = max(0, st.session_state.pos - 1)
if col_next.button("Next ▶"):
    st.session_state.pos = min(len(order) - 1, st.session_state.pos + 1)
if col_jump.button("Next unlabeled"):
    todo = [i for i, x in enumerate(order) if x not in done]
    if todo:
        st.session_state.pos = todo[0]

jump = st.sidebar.number_input("Go to item #", min_value=0, max_value=int(max(order)), value=0, step=1,
                               help="Enter an item number and press Enter (0 = stay here)")
if jump != st.session_state.get("last_jump", 0):          # only act when the box changes
    st.session_state["last_jump"] = jump
    if jump in order:
        st.session_state.pos = order.index(jump)

item_id = order[st.session_state.pos]
row = sample[sample["item_id"] == item_id].iloc[0]
prev = labels[labels["item_id"] == item_id].iloc[0] if item_id in done else None

st.markdown(f"**Item {item_id}** · position {st.session_state.pos + 1} of {len(order)} · "
            f"{row['activist']} · {row['filed_date']} · [filing]({row['url']})")
if p == 1:
    st.caption(f"Company: {row['subject_name']}")

left, right = st.columns([3, 2])
with left:
    st.text_area("Item 4. Purpose of Transaction", row["item4_text"], height=620, disabled=True)

with right:
    with st.form(f"form_{p}_{item_id}"):
        st.write("**Demands stated** (specific intentions only; ignore boilerplate lists of possible actions)")
        picks = {k: st.checkbox(v, value=bool(prev.get(k, 0)) if prev is not None and pd.notna(prev.get(k, 0)) else False,
                                key=f"{k}_{p}_{item_id}")
                 for k, v in CATEGORIES.items()}
        seats = st.number_input("Board seats requested (0 if none or not stated)", 0, 20,
                                int(prev["seats"]) if prev is not None else 0)
        proxy = st.radio("Proxy fight threatened?", ["no", "yes"],
                         index=["no", "yes"].index(prev["proxy_fight"]) if prev is not None else 0, horizontal=True)
        tone = st.radio("Tone", TONES, index=TONES.index(prev["tone"]) if prev is not None else 1, horizontal=True)
        unclear = st.checkbox("Genuinely ambiguous (flag for review)",
                              value=bool(prev["ambiguous"]) if prev is not None else False)
        notes = st.text_input("Notes", prev["notes"] if prev is not None and isinstance(prev["notes"], str) else "")
        if st.form_submit_button("Save", type="primary"):
            rec = {"item_id": item_id, **{k: int(v) for k, v in picks.items()}, "seats": int(seats),
                   "proxy_fight": proxy, "tone": tone, "ambiguous": int(unclear), "notes": notes}
            labels = labels[labels["item_id"] != item_id]
            labels = pd.concat([labels, pd.DataFrame([rec])], ignore_index=True)
            labels.to_csv(labels_path(p), index=False)
            st.session_state.pos = min(len(order) - 1, st.session_state.pos + 1)
            st.rerun()
