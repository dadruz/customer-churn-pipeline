"""Streamlit dashboard for the customer churn pipeline.

Reads the scored export produced by `src/evaluate.py` and the model comparison
written by `src/train.py`, and presents them as a single-page dashboard.

    streamlit run dashboards/app.py

The scored CSV is the single source of truth: `src/evaluate.py` writes customer
attributes into it alongside the probabilities, so nothing here needs the DuckDB
file. That matters for deployment -- Streamlit Cloud only has the repository's
committed files, and the database is gitignored as a rebuildable artifact.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SCORED_PATH = ROOT / "data" / "processed" / "scored_customers.csv"
COMPARISON_PATH = ROOT / "models" / "model_comparison.md"

# Matches the operating point chosen in src/evaluate.py: the highest-recall
# threshold whose precision still clears 0.50 on the training folds.
DEFAULT_THRESHOLD = 0.24

BLUE = "#2a78d6"
ORANGE = "#eb6834"


# ----------------------------------------------------------------- data


@st.cache_data
def load_scored():
    """Load the scored export written by src/evaluate.py.

    The file carries probabilities, outcomes and customer attributes together,
    so it is the app's only data input.
    """
    return pd.read_csv(SCORED_PATH)


@st.cache_data
def load_comparison():
    """Read the model comparison markdown, if train.py has written it."""
    if not COMPARISON_PATH.exists():
        return None
    return COMPARISON_PATH.read_text()


@st.cache_data
def to_csv_bytes(frame):
    """Encode a frame for the download button."""
    return frame.to_csv(index=False).encode("utf-8")


def metrics_at(frame, threshold):
    """Precision, recall and flagged count at one decision threshold."""
    flagged = frame["churn_probability"] >= threshold
    churned = frame["actual_churn"] == 1

    tp = int((flagged & churned).sum())
    fp = int((flagged & ~churned).sum())
    fn = int((~flagged & churned).sum())

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "flagged": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


# ----------------------------------------------------------------- charts


def ink_colours():
    """Text and grid colours that stay legible in either Streamlit theme."""
    try:
        dark = st.context.theme.type == "dark"
    except Exception:                      # older Streamlit, or no theme context
        dark = False
    return ("#d6d5d0", "#4a4a47") if dark else ("#52514e", "#e5e4e0")


def show(fig):
    """Render a figure and release it.

    Every slider step triggers a full script rerun, so figures would otherwise
    accumulate in matplotlib's global registry for the life of the session.
    """
    st.pyplot(fig)
    plt.close(fig)


def blank_figure(width, height):
    """A transparent figure, so it sits on the app background in either theme."""
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    return fig, ax


def rate_bar_chart(frame, column, title):
    """Horizontal bar chart of actual churn rate by one categorical column."""
    ink, grid = ink_colours()

    summary = (frame.groupby(column, observed=True)
                    .agg(n=("actual_churn", "size"), rate=("actual_churn", "mean"))
                    .sort_values("rate"))
    overall = frame["actual_churn"].mean() * 100

    fig, ax = blank_figure(6.2, 2.9)
    ax.barh(summary.index.astype(str), summary["rate"] * 100,
            color=BLUE, height=0.6)

    for y, (rate, n) in enumerate(zip(summary["rate"] * 100, summary["n"])):
        ax.text(rate + 0.8, y, f"{rate:.1f}%  (n={n:,})",
                va="center", fontsize=8.5, color=ink)

    ax.axvline(overall, color=ink, linestyle="--", linewidth=1.1, zorder=1)
    ax.set_xlim(0, 68)
    ax.set_xlabel("churn rate (%)", fontsize=9, color=ink)
    ax.set_title(title, fontsize=10.5, color=ink, loc="left", pad=8)
    ax.tick_params(colors=ink, labelsize=9)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(grid)

    fig.tight_layout()
    return fig


def probability_histogram(frame, threshold):
    """Distribution of predicted probability, split by what actually happened."""
    ink, grid = ink_colours()
    bins = [i / 50 for i in range(51)]

    fig, ax = blank_figure(12, 3.6)
    for label, colour, name in [(0, BLUE, "stayed"), (1, ORANGE, "churned")]:
        values = frame.loc[frame["actual_churn"] == label, "churn_probability"]
        ax.hist(values, bins=bins, color=colour, alpha=0.55, label=f"actually {name}")
        ax.hist(values, bins=bins, histtype="step", color=colour, linewidth=1.8)

    ax.axvline(threshold, color=ink, linestyle="--", linewidth=1.6)
    ax.text(threshold + 0.008, ax.get_ylim()[1] * 0.94,
            f"threshold {threshold:.2f}", fontsize=9, color=ink, va="top")

    ax.set_xlim(0, 1)
    ax.set_xlabel("predicted churn probability", fontsize=9, color=ink)
    ax.set_ylabel("customers", fontsize=9, color=ink)
    ax.tick_params(colors=ink, labelsize=9)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(grid)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(ink)

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- app

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📉",
    layout="wide",
)

if not SCORED_PATH.exists():
    st.error(
        f"No scored export at `{SCORED_PATH.relative_to(ROOT)}`. "
        "Run `python3 src/evaluate.py` (or trigger the Airflow DAG) first."
    )
    st.stop()

data = load_scored()

st.title("Customer Churn Dashboard")
st.caption(
    "Telco customer churn — DuckDB ingestion, a scored risk model, and the "
    "retention shortlist it produces."
)

# Filled in after the threshold slider is read further down the page, so the
# headline numbers reflect whatever operating point is currently selected.
kpi_row = st.container()

st.divider()

# --- Churn drivers ---------------------------------------------------------

st.subheader("Churn drivers")

st.caption(
    "Observed churn rate by segment, from actual outcomes. The dashed line "
    "marks the overall rate."
)
left, right = st.columns(2)
with left:
    show(rate_bar_chart(data, "Contract", "By contract type"))
with right:
    show(rate_bar_chart(data, "PaymentMethod", "By payment method"))

st.divider()

# --- Risk segments ---------------------------------------------------------

st.subheader("Risk segments")
st.caption(
    "Move the threshold to trade precision against recall. Lower catches more "
    "churners and flags more customers who would have stayed."
)

threshold = st.slider(
    "Decision threshold",
    min_value=0.05, max_value=0.95,
    value=DEFAULT_THRESHOLD, step=0.01,
    help="Customers scoring at or above this probability are flagged as at risk.",
)

scores = metrics_at(data, threshold)

slider_metrics = st.columns(3)
slider_metrics[0].metric("Precision", f"{scores['precision']:.1%}",
                         help="Share of flagged customers who actually churned.")
slider_metrics[1].metric("Recall", f"{scores['recall']:.1%}",
                         help="Share of all churners that the model flags.")
slider_metrics[2].metric("Customers flagged", f"{scores['flagged']:,}",
                         help=f"{scores['flagged'] / len(data):.1%} of the base.")

show(probability_histogram(data, threshold))

if threshold != DEFAULT_THRESHOLD:
    st.caption(
        f"Model default is {DEFAULT_THRESHOLD:.2f} — the highest-recall "
        "threshold whose precision still clears 0.50 on the training folds."
    )

st.divider()

# --- High-risk customers ---------------------------------------------------

st.subheader("High-risk customers")

DISPLAY_COLUMNS = [
    "customerID", "churn_probability",
    "Contract", "PaymentMethod", "InternetService", "tenure", "MonthlyCharges",
    "predicted_churn", "actual_churn", "split",
]

at_risk = data[data["churn_probability"] >= threshold]

contracts = sorted(data["Contract"].dropna().unique())
chosen = st.multiselect(
    "Filter by contract type", contracts, default=contracts,
    help="Leave all selected to see the whole shortlist.",
)
at_risk = at_risk[at_risk["Contract"].isin(chosen)]

at_risk = at_risk.sort_values("churn_probability", ascending=False)

st.caption(
    f"{len(at_risk):,} customers at or above the {threshold:.2f} threshold. "
    "Showing the 200 highest-risk; the download covers the full filtered set."
)

st.dataframe(
    at_risk[DISPLAY_COLUMNS].head(200),
    width="stretch",
    hide_index=True,
    column_config={
        "customerID": st.column_config.TextColumn("Customer"),
        "churn_probability": st.column_config.ProgressColumn(
            "Churn probability", min_value=0.0, max_value=1.0, format="%.3f",
        ),
        "tenure": st.column_config.NumberColumn("Tenure (mo)"),
        "MonthlyCharges": st.column_config.NumberColumn(
            "Monthly charges", format="$%.2f"),
        "predicted_churn": st.column_config.NumberColumn("Flagged"),
        "actual_churn": st.column_config.NumberColumn("Churned"),
        "split": st.column_config.TextColumn("Split"),
    },
)

st.download_button(
    f"Download all {len(at_risk):,} filtered customers (CSV)",
    data=to_csv_bytes(at_risk[DISPLAY_COLUMNS]),
    file_name=f"high_risk_customers_threshold_{threshold:.2f}.csv",
    mime="text/csv",
)

st.divider()

# --- Model provenance ------------------------------------------------------

st.subheader("Model")

comparison = load_comparison()
if comparison:
    # Drop the leading H1 so it does not compete with the page title.
    body = comparison.split("\n", 1)[1] if comparison.startswith("# ") else comparison
    st.markdown(body)
else:
    st.info(
        "No `models/model_comparison.md` yet. Run `python3 src/train.py` to "
        "generate it."
    )

test_rows = int((data["split"] == "test").sum())
train_rows = int((data["split"] == "train").sum())
st.caption(
    f"Every prediction shown is out-of-sample. The {train_rows:,} training rows "
    f"carry out-of-fold cross-validated probabilities and the {test_rows:,} "
    "held-out rows were scored by the saved model, which never saw them; the "
    "`split` column records which path produced each row. Filter on it before "
    "using this data to measure anything."
)
