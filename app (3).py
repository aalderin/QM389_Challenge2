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
INK = "#15263F"
TEAL = "#128A85"
CORAL = "#E67C60"
SEGMENT_COLORS = {"Promising": "#5C84D8", "Growth": "#128A85", "Stable": "#D7A443", "Decline": "#E67C60", "Unclassified": "#9CA3AF"}


st.set_page_config(
    page_title="NovaRetail Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_design():
    st.markdown("""
    <style>
    .stApp {background: #F6F8FA; color: #15263F;}
    section[data-testid="stSidebar"] {background: #EAF1F3;}
    .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1450px;}
    div[data-testid="stMetric"] {
      background:#FFFFFF; border:1px solid #E2E9ED; border-top:4px solid #128A85;
      padding:1rem 1.1rem; border-radius:13px;
      box-shadow:0 3px 12px rgba(18,46,67,.035); min-height:125px;
    }
    div[data-testid="stMetricLabel"] {color:#526778; font-size:.95rem;}
    div[data-testid="stMetricLabel"] > div {
      white-space:normal !important; overflow-wrap:anywhere; line-height:1.25;
    }
    div[data-testid="stMetricValue"] {color:#15263F; font-weight:750;}
    div[data-testid="stMetricValue"] > div {
      font-size:clamp(1.35rem, 2vw, 2rem) !important;
      line-height:1.15; white-space:normal !important; overflow-wrap:anywhere;
    }
    .filter-summary {color:#617486; font-size:.9rem; line-height:1.5;
      letter-spacing:.02em; margin:0 0 1rem; overflow-wrap:anywhere;}
    .hero {background:linear-gradient(105deg,#14283F 0%,#1E4E60 68%,#167E7B 100%);
      color:#fff;padding:29px 34px;border-radius:19px;margin-bottom:20px;}
    .hero .eyebrow {color:#A8E8DD;font-size:12px;font-weight:750;letter-spacing:2px;}
    .hero h1 {color:#fff;font-size:34px;letter-spacing:-.7px;margin:7px 0;line-height:1.15;}
    .hero p {color:#D5E9EC;font-size:15px;margin:3px 0;}
    .section-kicker {font-size:12px;letter-spacing:1.9px;font-weight:800;color:#16827B;margin-top:12px;}
    .insight-panel {background:#fff;border-left:4px solid #E67C60;border-radius:10px;
      padding:14px 19px;margin:11px 0 21px;}
    .subtle {color:#617486;font-size:13px;}
    div[data-testid="stPlotlyChart"] {background:#fff;border:1px solid #E5EBEF;
      border-radius:13px;padding:8px 10px;}
    </style>""", unsafe_allow_html=True)


def chart_style(fig, *, height=350):
    fig.update_layout(
        height=height, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial", color=INK, size=12),
        title=dict(font=dict(size=16, color=INK)),
        margin=dict(l=28, r=30, t=66, b=35),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#DCE5E9"),
        yaxis=dict(showgrid=True, gridcolor="#EDF1F3", zeroline=False),
        showlegend=False, hoverlabel=dict(bgcolor=INK, font_color="white"),
    )
    return fig


def currency_bar(data, dim, title, *, horizontal=False, height=350, color=TEAL):
    if horizontal:
        view = data.sort_values("Revenue", ascending=True).copy()
        view[dim] = view[dim].astype(str)
        fig = px.bar(view, x="Revenue", y=dim, orientation="h", title=title,
                     text="Revenue", color_discrete_sequence=[color],
                     category_orders={dim: view[dim].tolist()})
        fig.update_traces(texttemplate="$%{x:,.0f}", textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}<br>$%{x:,.2f}<extra></extra>")
        fig.update_layout(xaxis_title="Purchase amount ($)", yaxis_title=None)
        fig.update_yaxes(showgrid=False, type="category")
        fig.update_xaxes(showgrid=True, gridcolor="#EDF1F3", rangemode="tozero")
    else:
        fig = px.bar(data, x=dim, y="Revenue", title=title, text="Revenue",
                     color_discrete_sequence=[color])
        fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>")
        fig.update_layout(xaxis_title=None, yaxis_title="Purchase amount ($)")
    return chart_style(fig, height=height)


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


def format_compact_currency(value: float) -> str:
    """Readable KPI display; exact amounts are retained in the data and charts."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.2f}"


def show_kpis(df: pd.DataFrame) -> None:
    k = calculate_kpis(df)
    first_row = st.columns(3, gap="medium")
    second_row = st.columns(2, gap="medium")

    first_row[0].metric("Purchase amount", format_compact_currency(k["revenue"]))
    first_row[1].metric("Unique customers", f"{k['customers']:,}")
    first_row[2].metric("Avg. purchase / record", format_compact_currency(k["avg_purchase"]))
    second_row[0].metric(
        "Avg. satisfaction",
        "—" if pd.isna(k["avg_satisfaction"]) else f"{k['avg_satisfaction']:.2f} / 5",
    )
    second_row[1].metric("Decline revenue share", f"{k['decline_share']:.1%}")

    st.caption(
        f"Exact purchase amount: {format_currency(k['revenue'])} · "
        f"Exact average purchase per record: {format_currency(k['avg_purchase'])}"
    )


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
    st.markdown('<div class="section-kicker">01 / THE BIG PICTURE</div>', unsafe_allow_html=True)
    st.subheader("Where is customer value coming from?")
    left, right = st.columns([1.2, 1])
    seg = aggregate_revenue(df, "Segment")
    if not seg.empty:
        fig = px.bar(seg, x="Revenue", y="Segment", orientation="h", color="Segment",
                     color_discrete_map=SEGMENT_COLORS, text="Revenue",
                     title="Purchase amount by segment",
                     category_orders={"Segment": list(reversed(seg["Segment"].tolist()))})
        fig.update_traces(texttemplate="$%{x:,.0f}", textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}: $%{x:,.2f}<extra></extra>")
        fig.update_layout(xaxis_title="Purchase amount ($)", yaxis_title=None)
        fig.update_yaxes(type="category", showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor="#EDF1F3")
        left.plotly_chart(chart_style(fig, height=380), use_container_width=True)
    reg = aggregate_revenue(df, "CustomerRegion")
    if not reg.empty:
        right.plotly_chart(currency_bar(reg, "CustomerRegion", "Regional contribution", color="#4C7BA7", height=380), use_container_width=True)

    monthly = (df.dropna(subset=["TransactionDate"])
               .assign(Month=lambda x: x["TransactionDate"].dt.to_period("M").dt.to_timestamp()))
    monthly = monthly.groupby("Month", as_index=False)["PurchaseAmount"].sum()
    if len(monthly) > 1:
        fig = px.area(monthly, x="Month", y="PurchaseAmount", title="Purchase amount over time (monthly)")
        fig.update_traces(line_color=TEAL, fillcolor="rgba(18,138,133,0.17)",
                          hovertemplate="%{x|%b %Y}<br>$%{y:,.2f}<extra></extra>")
        fig.update_layout(xaxis_title=None, yaxis_title="Purchase amount ($)")
        st.plotly_chart(chart_style(fig, height=310), use_container_width=True)
    elif not monthly.empty:
        st.caption("A time-trend chart needs more than one distinct month in the selected data.")

    st.markdown("#### Insights → decisions")
    observations, recommendations = current_insights(df)
    a,b=st.columns(2)
    with a:
        st.markdown("**What the sample shows**")
        for v in observations: st.markdown("- "+v)
    with b:
        st.markdown("**Where to explore next**")
        for v in recommendations: st.markdown("- "+v)


def render_growth(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-kicker">02 / OPPORTUNITY SCAN</div>', unsafe_allow_html=True)
    st.subheader("Find the pockets worth exploring")
    growth = df[df["Segment"].isin(["Growth", "Promising"])].copy()
    if growth.empty:
        st.info("No Growth or Promising records match the current filters.")
        return
    c1,c2,c3 = st.columns(3)
    c1.metric("Growth + Promising amount", format_currency(growth["PurchaseAmount"].sum()))
    c2.metric("Share of filtered amount", f"{growth['PurchaseAmount'].sum() / df['PurchaseAmount'].sum():.1%}" if df['PurchaseAmount'].sum() else "—")
    c3.metric("Growth + Promising records", f"{len(growth):,}")
    left,right=st.columns([1,1])
    left.plotly_chart(currency_bar(aggregate_revenue(growth,"CustomerRegion"),"CustomerRegion","Regions contributing to growth",horizontal=True,color=TEAL),use_container_width=True)
    right.plotly_chart(currency_bar(aggregate_revenue(growth,"RetailChannel"),"RetailChannel","Channels contributing to growth",horizontal=True,color="#4C7BA7"),use_container_width=True)
    st.plotly_chart(currency_bar(aggregate_revenue(growth,"ProductCategory").head(10),"ProductCategory","Top product categories · Growth + Promising",horizontal=True,height=440,color="#D7A443"),use_container_width=True)
    st.caption("Labels match the source data; similar product category names have not been merged.")


def render_decline(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-kicker">03 / RETENTION WATCH</div>', unsafe_allow_html=True)
    st.subheader("Where are the decline signals concentrated?")
    decline = df[df["Segment"] == "Decline"].copy()
    if decline.empty:
        st.info("No Decline-labeled records match the current filters.")
        return
    a,b,c=st.columns(3)
    a.metric("Decline-labeled amount",format_currency(decline['PurchaseAmount'].sum()))
    b.metric("Customers in Decline records", f"{decline['CustomerID'].nunique():,}")
    mean_sat=decline['CustomerSatisfaction'].mean()
    c.metric("Avg. satisfaction",f"{mean_sat:.2f} / 5" if pd.notna(mean_sat) else "—")
    left,right=st.columns(2)
    left.plotly_chart(currency_bar(aggregate_revenue(decline,"CustomerRegion"),"CustomerRegion","Decline amount by region",horizontal=True,color=CORAL),use_container_width=True)
    right.plotly_chart(currency_bar(aggregate_revenue(decline,"RetailChannel"),"RetailChannel","Decline amount by channel",horizontal=True,color="#BB735E"),use_container_width=True)
    st.plotly_chart(currency_bar(aggregate_revenue(decline,"ProductCategory").head(10),"ProductCategory","Categories seen in Decline records",horizontal=True,height=430,color=CORAL),use_container_width=True)
    customer=(decline.groupby("CustomerID",as_index=False)
              .agg(PurchaseAmount=("PurchaseAmount","sum"), Records=("PurchaseAmount","size"),
                   AvgSatisfaction=("CustomerSatisfaction","mean"), LastDate=("TransactionDate","max"))
              .sort_values("PurchaseAmount",ascending=False))
    customer["CustomerID"] = customer["CustomerID"].astype(str)
    st.markdown("#### Customers appearing in Decline-labeled records")
    st.dataframe(customer,hide_index=True,use_container_width=True,
                 column_config={"PurchaseAmount": st.column_config.NumberColumn("Purchase amount",format="$%.2f"),
                                "AvgSatisfaction":st.column_config.NumberColumn("Avg. satisfaction",format="%.2f")})
    st.caption("A Decline label on a record does not prove that its customer has a single permanent segment or will churn.")


def render_customer_detail(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-kicker">04 / CUSTOMER LENS</div>', unsafe_allow_html=True)
    st.subheader("Explore your most valuable customers")
    if df.empty:
        st.info("No records match the current filters.")
        return
    customer=(df.groupby("CustomerID",as_index=False)
              .agg(PurchaseAmount=("PurchaseAmount","sum"), Records=("PurchaseAmount","size"),
                   AvgPurchase=("PurchaseAmount","mean"), AvgSatisfaction=("CustomerSatisfaction","mean"),
                   FirstDate=("TransactionDate","min"),LastDate=("TransactionDate","max"))
              .sort_values("PurchaseAmount",ascending=False))
    labels=(df.groupby("CustomerID")["Segment"]
            .apply(lambda s:", ".join(sorted(set(s)))).rename("SegmentLabelsObserved").reset_index())
    customer=customer.merge(labels,on="CustomerID",how="left")
    ranked=customer.head(12).sort_values("PurchaseAmount",ascending=True).copy()
    ranked["CustomerLabel"]="Customer " + ranked["CustomerID"].astype(str)
    # CustomerLabel is a categorical axis, never a continuous numerical ID.
    fig=px.bar(ranked,x="PurchaseAmount",y="CustomerLabel",orientation="h",
               title="Top 12 customers by purchase amount",text="PurchaseAmount",
               color_discrete_sequence=[TEAL],
               category_orders={"CustomerLabel":ranked["CustomerLabel"].tolist()})
    fig.update_traces(texttemplate="$%{x:,.0f}",textposition="outside",cliponaxis=False,
                      hovertemplate="%{y}<br>Purchase amount: $%{x:,.2f}<extra></extra>")
    fig.update_yaxes(type="category",showgrid=False,title=None)
    fig.update_xaxes(title="Purchase amount ($)",rangemode="tozero",showgrid=True,gridcolor="#EDF1F3")
    st.plotly_chart(chart_style(fig,height=480),use_container_width=True)

    st.markdown("#### Customer profile")
    ids=customer['CustomerID'].tolist()
    selected=st.selectbox("Select a customer to investigate",options=ids,
                          format_func=lambda x:f"Customer {x}")
    row=customer[customer['CustomerID']==selected].iloc[0]
    a,b,c,d=st.columns(4)
    a.metric("Total purchases",format_currency(row['PurchaseAmount']))
    b.metric("Purchase records",str(int(row['Records'])))
    c.metric("Average purchase",format_currency(row['AvgPurchase']))
    d.metric("Avg. satisfaction",f"{row['AvgSatisfaction']:.2f} / 5" if pd.notna(row['AvgSatisfaction']) else "—")
    st.markdown(f"**Observed segment labels:** {row['SegmentLabelsObserved']}")
    detail=df[df['CustomerID']==selected].sort_values('TransactionDate',ascending=False).copy()
    st.dataframe(detail[['TransactionDate','ProductCategory','RetailChannel','CustomerRegion','Segment','PurchaseAmount','CustomerSatisfaction']],
                 hide_index=True,use_container_width=True,
                 column_config={'PurchaseAmount':st.column_config.NumberColumn('Purchase amount',format='$%.2f'),
                                'CustomerSatisfaction':st.column_config.NumberColumn('Satisfaction',format='%.1f')})
    st.markdown("#### Full customer ranking")
    view=customer.copy()
    view['CustomerID']=view['CustomerID'].astype(str)
    st.dataframe(view,hide_index=True,use_container_width=True,
                 column_config={'PurchaseAmount':st.column_config.NumberColumn('Total amount',format='$%.2f'),
                                'AvgPurchase':st.column_config.NumberColumn('Avg. purchase',format='$%.2f'),
                                'AvgSatisfaction':st.column_config.NumberColumn('Avg. satisfaction',format='%.2f')})
    st.caption("Customers may have multiple segment labels in the source data; no single label is inferred.")


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
    apply_design()
    st.markdown('''<div class="hero"><div class="eyebrow">NOVA RETAIL / CUSTOMER INTELLIGENCE</div>
      <h1>Make every customer signal count.</h1>
      <p>A decision workspace for revenue, growth opportunities, and retention.</p></div>''',unsafe_allow_html=True)
    try:
        source=load_data(DATA_PATH)
        df,cleaning_issues=clean_data(source)
    except Exception as exc:
        st.error(f"The dashboard could not load the dataset: {exc}")
        st.stop()

    quality=data_quality_summary(df)
    filtered=filter_data(df)
    st.markdown(
        f'<div class="filter-summary">Exploring <strong>{len(filtered):,}</strong> '
        f'of <strong>{len(df):,}</strong> purchase records. '
        'All figures respond to the sidebar filters.</div>',
        unsafe_allow_html=True,
    )
    show_kpis(filtered)

    pages=["Snapshot","Opportunity map","Retention watch","Customer explorer","Ask Nova AI"]
    selected_page=st.radio("Workspace",pages,horizontal=True,label_visibility="collapsed")
    st.divider()
    if filtered.empty:
        st.info("No records match the current filter combination. Adjust the sidebar filters to continue.")
    else:
        if selected_page==pages[0]:render_overview(filtered)
        elif selected_page==pages[1]:render_growth(filtered)
        elif selected_page==pages[2]:render_decline(filtered)
        elif selected_page==pages[3]:render_customer_detail(filtered)
        else:render_chatbot(filtered)

    with st.expander("Source data · quality notes & methodology"):
        st.markdown(
            f"- Source rows: **{quality['rows']}**\n"
            f"- Unique CustomerIDs: **{quality['unique_customers']}**\n"
            f"- Unique TransactionIDs: **{quality['unique_transaction_ids']}**\n"
            f"- Missing segment labels: **{quality['missing_segment']}**\n"
            f"- Exact duplicate rows: **{quality['exact_duplicates']}**\n"
            f"- Reused TransactionIDs: **{quality['reused_transaction_ids']}** IDs appear on multiple rows\n"
            f"- Customers with multiple segment labels: **{quality['multi_segment_customers']}**"
        )
        if cleaning_issues:
            st.markdown("**Additional validation notes:** "+"; ".join(cleaning_issues))
        st.markdown("**Interpretation:** PurchaseAmount is a purchase-value proxy. Segment charts reflect record-level labels, "
                    "not a unique classification for each person. No churn/engagement score or causal inference is fabricated. "
                    "Product-category labels remain as supplied. Findings are sample-level, not population claims.")


if __name__ == "__main__":
    main()
