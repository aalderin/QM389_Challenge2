from pathlib import Path
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "NR_dataset.xlsx"
SEGMENT_ORDER = ["Promising", "Growth", "Stable", "Decline", "Unclassified"]


st.set_page_config(
    page_title="NovaRetail Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.name}")
    df = pd.read_excel(path, sheet_name="data")
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply only conservative, meaning-preserving preparation needed for analysis."""
    required = {
        "label", "CustomerID", "TransactionID", "TransactionDate", "ProductCategory",
        "PurchaseAmount", "CustomerAgeGroup", "CustomerGender", "CustomerRegion",
        "CustomerSatisfaction", "RetailChannel"
    }
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError("Missing required columns: " + ", ".join(missing_required))

    out = df.copy()
    out["TransactionDate"] = pd.to_datetime(out["TransactionDate"], errors="coerce")
    out["PurchaseAmount"] = pd.to_numeric(out["PurchaseAmount"], errors="coerce")
    out["CustomerSatisfaction"] = pd.to_numeric(out["CustomerSatisfaction"], errors="coerce")
    out["Segment"] = out["label"].fillna("Unclassified").astype(str).str.strip()

    issues = []
    if out["TransactionDate"].isna().any():
        issues.append(f"{int(out['TransactionDate'].isna().sum())} invalid/missing transaction date(s)")
    if out["PurchaseAmount"].isna().any():
        issues.append(f"{int(out['PurchaseAmount'].isna().sum())} invalid/missing purchase amount(s)")
    if out["CustomerSatisfaction"].isna().any():
        issues.append(f"{int(out['CustomerSatisfaction'].isna().sum())} invalid/missing satisfaction value(s)")
    if (out["PurchaseAmount"] < 0).any():
        issues.append("Negative purchase amounts detected")
    if (~out["CustomerSatisfaction"].dropna().between(1, 5)).any():
        issues.append("Customer satisfaction values outside the expected 1–5 range detected")

    return out, issues


def data_quality_summary(df: pd.DataFrame) -> dict:
    exact_dup = int(df.duplicated().sum())
    missing_segment = int(df["label"].isna().sum())
    reused_tx = int((df["TransactionID"].value_counts() > 1).sum())
    label_counts = (
        df.dropna(subset=["label"])
        .groupby("CustomerID")["label"]
        .nunique()
    )
    multi_segment_customers = int((label_counts > 1).sum())
    return {
        "rows": len(df),
        "unique_customers": int(df["CustomerID"].nunique()),
        "unique_transaction_ids": int(df["TransactionID"].nunique()),
        "missing_segment": missing_segment,
        "exact_duplicates": exact_dup,
        "reused_transaction_ids": reused_tx,
        "multi_segment_customers": multi_segment_customers,
    }


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    segment_options = [s for s in SEGMENT_ORDER if s in set(df["Segment"])]
    selected_segments = st.sidebar.multiselect(
        "Customer segment", segment_options, default=segment_options
    )

    regions = sorted(df["CustomerRegion"].dropna().astype(str).unique())
    selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

    categories = sorted(df["ProductCategory"].dropna().astype(str).unique())
    selected_categories = st.sidebar.multiselect("Product category", categories, default=categories)

    channels = sorted(df["RetailChannel"].dropna().astype(str).unique())
    selected_channels = st.sidebar.multiselect("Retail channel", channels, default=channels)

    min_date = df["TransactionDate"].min().date()
    max_date = df["TransactionDate"].max().date()
    selected_dates = st.sidebar.date_input(
        "Transaction date", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates if not isinstance(selected_dates, tuple) else min_date

    mask = (
        df["Segment"].isin(selected_segments)
        & df["CustomerRegion"].astype(str).isin(selected_regions)
        & df["ProductCategory"].astype(str).isin(selected_categories)
        & df["RetailChannel"].astype(str).isin(selected_channels)
        & df["TransactionDate"].dt.date.between(start_date, end_date)
    )
    return df.loc[mask].copy()


def calculate_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "revenue": 0.0, "customers": 0, "avg_purchase": 0.0,
            "avg_satisfaction": float("nan"), "decline_share": 0.0,
        }
    revenue = float(df["PurchaseAmount"].sum())
    decline_revenue = float(df.loc[df["Segment"] == "Decline", "PurchaseAmount"].sum())
    return {
        "revenue": revenue,
        "customers": int(df["CustomerID"].nunique()),
        "avg_purchase": float(df["PurchaseAmount"].mean()),
        "avg_satisfaction": float(df["CustomerSatisfaction"].mean()),
        "decline_share": decline_revenue / revenue if revenue else 0.0,
    }


def aggregate_revenue(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    return (
        df.groupby(dim, dropna=False, as_index=False)
        .agg(Revenue=("PurchaseAmount", "sum"), Records=("PurchaseAmount", "size"),
             AvgPurchase=("PurchaseAmount", "mean"), AvgSatisfaction=("CustomerSatisfaction", "mean"))
        .sort_values("Revenue", ascending=False)
    )


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def show_kpis(df: pd.DataFrame) -> None:
    k = calculate_kpis(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Purchase amount", format_currency(k["revenue"]))
    c2.metric("Unique customers", f"{k['customers']:,}")
    c3.metric("Avg. purchase / record", format_currency(k["avg_purchase"]))
    c4.metric("Avg. satisfaction", "—" if pd.isna(k["avg_satisfaction"]) else f"{k['avg_satisfaction']:.2f} / 5")
    c5.metric("Decline revenue share", f"{k['decline_share']:.1%}")


def current_insights(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if df.empty:
        return ["No records match the current filters."], ["Broaden the filters before making a commercial decision."]

    observations = []
    recommendations = []

    for dim, label in [
        ("Segment", "segment"), ("CustomerRegion", "region"),
        ("ProductCategory", "product category"), ("RetailChannel", "channel")
    ]:
        agg = aggregate_revenue(df, dim)
        if not agg.empty:
            top = agg.iloc[0]
            share = top["Revenue"] / df["PurchaseAmount"].sum()
            observations.append(
                f"Top {label}: **{top[dim]}**, with {format_currency(top['Revenue'])} "
                f"({share:.1%} of filtered purchase amount)."
            )

    decline = df[df["Segment"] == "Decline"]
    if not decline.empty:
        decline_share = decline["PurchaseAmount"].sum() / df["PurchaseAmount"].sum()
        decline_sat = decline["CustomerSatisfaction"].mean()
        observations.append(
            f"Decline-labeled records account for **{decline_share:.1%}** of filtered purchase amount "
            f"and average **{decline_sat:.2f}/5** satisfaction."
        )
        risk_region = aggregate_revenue(decline, "CustomerRegion").iloc[0]["CustomerRegion"]
        recommendations.append(
            f"Review retention activity in **{risk_region}** first because it contributes the most purchase amount among currently visible Decline records."
        )
    else:
        recommendations.append("No Decline-labeled records are visible under the current filters; prioritize growth analysis for this slice.")

    growth = df[df["Segment"].isin(["Growth", "Promising"])]
    if not growth.empty:
        top_growth_region = aggregate_revenue(growth, "CustomerRegion").iloc[0]["CustomerRegion"]
        recommendations.append(
            f"Consider testing incremental growth investment in **{top_growth_region}**, where visible Growth/Promising records generate the most purchase amount."
        )

    recommendations.append(
        "Treat these as prioritization signals from the sample, not causal evidence; validate with broader customer history before major investment changes."
    )
    return observations, recommendations


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Executive overview")
    left, right = st.columns(2)

    seg = aggregate_revenue(df, "Segment")
    if not seg.empty:
        fig = px.bar(seg, x="Segment", y="Revenue", text_auto="$.2s", title="Purchase Amount by Segment")
        fig.update_layout(yaxis_title="Purchase Amount ($)", xaxis_title="Segment")
        left.plotly_chart(fig, use_container_width=True)

    reg = aggregate_revenue(df, "CustomerRegion")
    if not reg.empty:
        fig = px.bar(reg, x="CustomerRegion", y="Revenue", text_auto="$.2s", title="Purchase Amount by Region")
        fig.update_layout(yaxis_title="Purchase Amount ($)", xaxis_title="Region")
        right.plotly_chart(fig, use_container_width=True)

    daily = (
        df.groupby("TransactionDate", as_index=False)["PurchaseAmount"].sum()
        .sort_values("TransactionDate")
    )
    if not daily.empty:
        fig = px.line(daily, x="TransactionDate", y="PurchaseAmount", markers=True, title="Purchase Amount Over Time")
        fig.update_layout(yaxis_title="Purchase Amount ($)", xaxis_title="Transaction Date")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### What the current filters show")
    observations, recommendations = current_insights(df)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Observed evidence**")
        for item in observations:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("**Possible actions to consider**")
        for item in recommendations:
            st.markdown(f"- {item}")


def render_growth(df: pd.DataFrame) -> None:
    st.subheader("Growth opportunities")
    growth = df[df["Segment"].isin(["Growth", "Promising"])].copy()
    if growth.empty:
        st.info("No Growth or Promising records match the current filters.")
        return

    c1, c2 = st.columns(2)
    region = aggregate_revenue(growth, "CustomerRegion")
    fig = px.bar(region, x="CustomerRegion", y="Revenue", title="Growth + Promising Purchase Amount by Region")
    fig.update_layout(yaxis_title="Purchase Amount ($)", xaxis_title="Region")
    c1.plotly_chart(fig, use_container_width=True)

    channel = aggregate_revenue(growth, "RetailChannel")
    fig = px.bar(channel, x="RetailChannel", y="Revenue", title="Growth + Promising Purchase Amount by Channel")
    fig.update_layout(yaxis_title="Purchase Amount ($)", xaxis_title="Retail Channel")
    c2.plotly_chart(fig, use_container_width=True)

    category = aggregate_revenue(growth, "ProductCategory").head(12).sort_values("Revenue")
    fig = px.bar(category, x="Revenue", y="ProductCategory", orientation="h", title="Top Product Categories: Growth + Promising")
    fig.update_layout(xaxis_title="Purchase Amount ($)", yaxis_title="Product Category")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Product-category labels are shown exactly as supplied in the dataset; similar labels were not merged without a documented business taxonomy.")


def render_decline(df: pd.DataFrame) -> None:
    st.subheader("Customer decline & retention signals")
    decline = df[df["Segment"] == "Decline"].copy()
    if decline.empty:
        st.info("No Decline-labeled records match the current filters.")
        return

    k1, k2, k3 = st.columns(3)
    k1.metric("Decline purchase amount", format_currency(decline["PurchaseAmount"].sum()))
    k2.metric("Customers in Decline records", f"{decline['CustomerID'].nunique():,}")
    k3.metric("Avg. Decline satisfaction", f"{decline['CustomerSatisfaction'].mean():.2f} / 5")

    c1, c2 = st.columns(2)
    reg = aggregate_revenue(decline, "CustomerRegion")
    fig = px.bar(reg, x="CustomerRegion", y="Revenue", title="Decline Purchase Amount by Region")
    fig.update_layout(yaxis_title="Purchase Amount ($)", xaxis_title="Region")
    c1.plotly_chart(fig, use_container_width=True)

    ch = aggregate_revenue(decline, "RetailChannel")
    fig = px.bar(ch, x="RetailChannel", y="Revenue", title="Decline Purchase Amount by Channel")
    fig.update_layout(yaxis_title="Purchase Amount ($)", xaxis_title="Retail Channel")
    c2.plotly_chart(fig, use_container_width=True)

    cat = aggregate_revenue(decline, "ProductCategory").head(10).sort_values("Revenue")
    fig = px.bar(cat, x="Revenue", y="ProductCategory", orientation="h", title="Top Categories within Decline Records")
    fig.update_layout(xaxis_title="Purchase Amount ($)", yaxis_title="Product Category")
    st.plotly_chart(fig, use_container_width=True)

    customer = (
        decline.groupby("CustomerID", as_index=False)
        .agg(PurchaseAmount=("PurchaseAmount", "sum"), Records=("PurchaseAmount", "size"),
             AvgSatisfaction=("CustomerSatisfaction", "mean"), LastDate=("TransactionDate", "max"))
        .sort_values("PurchaseAmount", ascending=False)
    )
    customer["PurchaseAmount"] = customer["PurchaseAmount"].map(lambda x: f"${x:,.2f}")
    customer["AvgSatisfaction"] = customer["AvgSatisfaction"].map(lambda x: f"{x:.2f}")
    customer["LastDate"] = customer["LastDate"].dt.strftime("%Y-%m-%d")
    st.markdown("#### Decline-labeled customer detail")
    st.dataframe(customer, use_container_width=True, hide_index=True)


def render_customer_detail(df: pd.DataFrame) -> None:
    st.subheader("Customer detail")
    if df.empty:
        st.info("No records match the current filters.")
        return

    customer = (
        df.groupby("CustomerID", as_index=False)
        .agg(PurchaseAmount=("PurchaseAmount", "sum"), Records=("PurchaseAmount", "size"),
             AvgPurchase=("PurchaseAmount", "mean"), AvgSatisfaction=("CustomerSatisfaction", "mean"),
             FirstDate=("TransactionDate", "min"), LastDate=("TransactionDate", "max"))
        .sort_values("PurchaseAmount", ascending=False)
    )
    labels = (
        df.groupby("CustomerID")["Segment"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .rename("SegmentLabelsObserved")
        .reset_index()
    )
    customer = customer.merge(labels, on="CustomerID", how="left")

    fig = px.bar(customer.head(15).sort_values("PurchaseAmount"), x="PurchaseAmount", y=customer.head(15).sort_values("PurchaseAmount")["CustomerID"].astype(str),
                 orientation="h", title="Top Customers by Purchase Amount")
    fig.update_layout(xaxis_title="Purchase Amount ($)", yaxis_title="Customer ID")
    st.plotly_chart(fig, use_container_width=True)

    display = customer.copy()
    display["PurchaseAmount"] = display["PurchaseAmount"].map(lambda x: f"${x:,.2f}")
    display["AvgPurchase"] = display["AvgPurchase"].map(lambda x: f"${x:,.2f}")
    display["AvgSatisfaction"] = display["AvgSatisfaction"].map(lambda x: f"{x:.2f}")
    display["FirstDate"] = display["FirstDate"].dt.strftime("%Y-%m-%d")
    display["LastDate"] = display["LastDate"].dt.strftime("%Y-%m-%d")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("A customer's segment is not forced to one value because repeated CustomerIDs can carry different segment labels in the source data.")


def build_grounding_context(df: pd.DataFrame) -> str:
    if df.empty:
        return "The current dashboard filters return zero rows."
    k = calculate_kpis(df)
    lines = [
        f"Rows visible: {len(df)}",
        f"Unique customers visible: {df['CustomerID'].nunique()}",
        f"Total purchase amount: {k['revenue']:.2f}",
        f"Average purchase amount per record: {k['avg_purchase']:.2f}",
        f"Average satisfaction: {k['avg_satisfaction']:.3f}",
        f"Decline revenue share: {k['decline_share']:.4f}",
    ]
    for dim in ["Segment", "CustomerRegion", "ProductCategory", "RetailChannel"]:
        agg = aggregate_revenue(df, dim).head(15)
        lines.append(f"{dim} revenue table:\n" + agg.to_string(index=False))
    return "\n\n".join(lines)


def deterministic_answer(question: str, df: pd.DataFrame) -> str:
    q = question.lower().strip()
    if df.empty:
        return "The current filters return no records, so I cannot calculate an answer."

    if "segment" in q and any(w in q for w in ["most revenue", "highest revenue", "generates the most", "top"]):
        a = aggregate_revenue(df, "Segment").iloc[0]
        return f"{a['Segment']} generates the most purchase amount in the current view: {format_currency(a['Revenue'])}."

    if "region" in q and any(w in q for w in ["strongest", "most", "highest", "top", "sales", "revenue"]):
        a = aggregate_revenue(df, "CustomerRegion").iloc[0]
        return f"{a['CustomerRegion']} is the strongest region in the current view by purchase amount: {format_currency(a['Revenue'])}."

    if any(w in q for w in ["at risk", "retention", "decline"]):
        d = df[df["Segment"] == "Decline"]
        if d.empty:
            return "There are no Decline-labeled records in the current filtered view."
        region = aggregate_revenue(d, "CustomerRegion").iloc[0]
        return (
            f"Decline-labeled records total {format_currency(d['PurchaseAmount'].sum())} with average satisfaction "
            f"of {d['CustomerSatisfaction'].mean():.2f}/5. {region['CustomerRegion']} has the most purchase amount "
            f"within those Decline records ({format_currency(region['Revenue'])}), so it is a reasonable first place to investigate retention needs."
        )

    if "channel" in q:
        a = aggregate_revenue(df, "RetailChannel").iloc[0]
        return f"{a['RetailChannel']} leads the current view by purchase amount at {format_currency(a['Revenue'])}."

    if "product" in q or "category" in q:
        a = aggregate_revenue(df, "ProductCategory").iloc[0]
        return f"{a['ProductCategory']} is the top product category in the current view by purchase amount: {format_currency(a['Revenue'])}."

    customer_match = re.search(r"\b(\d{5})\b", q)
    if customer_match:
        cid = int(customer_match.group(1))
        c = df[df["CustomerID"] == cid]
        if c.empty:
            return f"Customer {cid} is not present under the current filters."
        labels = ", ".join(sorted(c["Segment"].unique()))
        return (
            f"Customer {cid} has {len(c)} visible record(s), {format_currency(c['PurchaseAmount'].sum())} in total purchase amount, "
            f"average satisfaction {c['CustomerSatisfaction'].mean():.2f}/5, and these segment labels in the visible data: {labels}."
        )

    k = calculate_kpis(df)
    return (
        f"In the current filtered view there are {len(df)} records and {k['customers']} unique customers, with "
        f"{format_currency(k['revenue'])} in purchase amount and average satisfaction of {k['avg_satisfaction']:.2f}/5. "
        "Try asking which segment, region, product category, or channel leads, or ask about decline/retention."
    )


def get_openai_key() -> str | None:
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    return key or os.getenv("OPENAI_API_KEY")


def ai_answer(question: str, df: pd.DataFrame) -> str:
    key = get_openai_key()
    if not key or OpenAI is None:
        return deterministic_answer(question, df)

    context = build_grounding_context(df)
    client = OpenAI(api_key=key)
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are NovaRetail's dashboard assistant. Answer only from the supplied filtered-dashboard context. "
                        "Do not invent values, do not make causal claims, and clearly say when the context is insufficient. "
                        "Keep answers concise and executive-friendly. Distinguish observed evidence from recommendations."
                    ),
                },
                {"role": "user", "content": f"FILTERED DASHBOARD CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
            ],
        )
        return response.output_text.strip()
    except Exception as exc:
        fallback = deterministic_answer(question, df)
        return f"The optional AI connection was unavailable, so I used the dashboard's grounded calculation mode instead.\n\n{fallback}"


def render_chatbot(df: pd.DataFrame) -> None:
    st.subheader("AI dashboard assistant")
    st.caption("Answers are grounded in the currently filtered dashboard data. An OpenAI API key enables broader natural-language interpretation; core dashboard questions still work without one.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about segments, regions, products, channels, customers, or decline signals…")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        answer = ai_answer(question, df)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    st.markdown("**Example questions**")
    st.markdown(
        "- Which segment generates the most purchase amount?\n"
        "- Which region is strongest?\n"
        "- Where should we investigate retention risk?\n"
        "- Which product category leads?\n"
        "- Tell me about customer 12351."
    )


def main() -> None:
    st.title("NovaRetail Customer Intelligence Dashboard")
    st.caption("Interactive decision support for customer value, growth opportunities, and decline signals")

    try:
        source = load_data(DATA_PATH)
        df, cleaning_issues = clean_data(source)
    except Exception as exc:
        st.error(f"The dashboard could not load the dataset: {exc}")
        st.stop()

    quality = data_quality_summary(df)
    if quality["multi_segment_customers"] > 0:
        st.warning(
            f"Data-quality note: {quality['multi_segment_customers']} repeated CustomerIDs carry more than one segment label. "
            "Segment charts therefore describe segment-labeled records; customer rankings do not assign a single inferred segment."
        )

    filtered = filter_data(df)
    st.caption(f"Showing {len(filtered):,} of {len(df):,} purchase records under the current filters.")
    show_kpis(filtered)

    if filtered.empty:
        st.info("No records match the current filter combination. Adjust the sidebar filters to continue.")
    else:
        tabs = st.tabs(["Overview", "Growth", "Decline & Retention", "Customer Detail", "AI Assistant"])
        with tabs[0]:
            render_overview(filtered)
        with tabs[1]:
            render_growth(filtered)
        with tabs[2]:
            render_decline(filtered)
        with tabs[3]:
            render_customer_detail(filtered)
        with tabs[4]:
            render_chatbot(filtered)

    with st.expander("Data quality & methodology"):
        st.markdown(
            f"- Source rows: **{quality['rows']}**\n"
            f"- Unique CustomerIDs: **{quality['unique_customers']}**\n"
            f"- Unique TransactionIDs: **{quality['unique_transaction_ids']}**\n"
            f"- Missing segment labels: **{quality['missing_segment']}**\n"
            f"- Exact duplicate rows: **{quality['exact_duplicates']}**\n"
            f"- Reused TransactionIDs: **{quality['reused_transaction_ids']}** IDs appear on multiple rows\n"
            f"- Customers with multiple observed segment labels: **{quality['multi_segment_customers']}**"
        )
        if cleaning_issues:
            st.markdown("**Additional validation notes:** " + "; ".join(cleaning_issues))
        st.markdown(
            "**Interpretation guardrails:** PurchaseAmount is treated as the available revenue/value proxy. "
            "No churn score, engagement score, or causal effect is manufactured. Product-category labels are preserved as supplied. "
            "Findings describe this sample and should be validated against broader customer history before major decisions."
        )


if __name__ == "__main__":
    main()
