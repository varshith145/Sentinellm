"""
SentinelLM — Admin Dashboard (Streamlit)

Three-page dashboard for browsing and inspecting audit records:
  1. Overview Dashboard — charts, stats, decision distribution
  2. Request Log — filterable table with color-coded decisions
  3. Request Detail — full audit record inspection

Per PRD Section 13.
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# --- Database Connection ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ppg:ppg@db:5432/ppg")
engine = create_engine(DATABASE_URL)

# --- Page Config ---
st.set_page_config(
    page_title="SentinelLM Admin",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stMetric > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .decision-allow { color: #00d26a; font-weight: bold; }
    .decision-mask { color: #f9a825; font-weight: bold; }
    .decision-block { color: #ff4444; font-weight: bold; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar Navigation ---
st.sidebar.title("🛡️ SentinelLM")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "📋 Request Log", "🔍 Request Detail"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**SentinelLM** v1.0  \n"
    "AI Gateway Security Dashboard"
)


def get_connection():
    """Get a database connection."""
    return engine.connect()


def color_decision(val):
    """Color-code decision values."""
    colors = {
        "ALLOW": "color: #00d26a",
        "MASK": "color: #f9a825",
        "BLOCK": "color: #ff4444",
    }
    return colors.get(val, "")


# ============================================================
# PAGE 1: Overview Dashboard
# ============================================================
if page == "📊 Overview":
    st.title("📊 Overview Dashboard")
    st.markdown("Real-time overview of SentinelLM gateway activity.")

    try:
        with get_connection() as conn:
            # Total requests
            total = pd.read_sql(
                text("SELECT COUNT(*) as count FROM audit_log"), conn
            )
            total_count = int(total["count"].iloc[0]) if not total.empty else 0

            # Decision breakdown
            decisions = pd.read_sql(
                text("""
                    SELECT input_decision, COUNT(*) as count
                    FROM audit_log
                    GROUP BY input_decision
                """),
                conn,
            )

            # Requests over time (last 7 days)
            timeline = pd.read_sql(
                text("""
                    SELECT DATE(created_at) as date, COUNT(*) as count
                    FROM audit_log
                    WHERE created_at >= NOW() - INTERVAL '7 days'
                    GROUP BY DATE(created_at)
                    ORDER BY date
                """),
                conn,
            )

            # Top redaction types
            redactions = pd.read_sql(
                text("""
                    SELECT input_redactions
                    FROM audit_log
                    WHERE input_redactions != '{}'::jsonb
                """),
                conn,
            )

        # --- Metrics Row ---
        col1, col2, col3, col4 = st.columns(4)

        allow_count = int(
            decisions[decisions["input_decision"] == "ALLOW"]["count"].sum()
        ) if not decisions.empty else 0
        mask_count = int(
            decisions[decisions["input_decision"] == "MASK"]["count"].sum()
        ) if not decisions.empty else 0
        block_count = int(
            decisions[decisions["input_decision"] == "BLOCK"]["count"].sum()
        ) if not decisions.empty else 0

        col1.metric("Total Requests", f"{total_count:,}")
        col2.metric("✅ Allowed", f"{allow_count:,}")
        col3.metric("🟡 Masked", f"{mask_count:,}")
        col4.metric("🔴 Blocked", f"{block_count:,}")

        # --- Charts Row ---
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Requests by Decision")
            if not decisions.empty:
                fig = px.pie(
                    decisions,
                    values="count",
                    names="input_decision",
                    color="input_decision",
                    color_discrete_map={
                        "ALLOW": "#00d26a",
                        "MASK": "#f9a825",
                        "BLOCK": "#ff4444",
                    },
                    hole=0.4,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data yet. Send some requests through the gateway!")

        with chart_col2:
            st.subheader("Requests Over Time (7d)")
            if not timeline.empty:
                fig = px.line(
                    timeline,
                    x="date",
                    y="count",
                    markers=True,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    xaxis_title="Date",
                    yaxis_title="Requests",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data in the last 7 days.")

        # --- Top Redaction Types ---
        st.subheader("Top Redaction Types")
        if not redactions.empty:
            # Aggregate redaction counts across all records
            all_redaction_counts = {}
            for _, row in redactions.iterrows():
                for entity_type, count in row["input_redactions"].items():
                    all_redaction_counts[entity_type] = (
                        all_redaction_counts.get(entity_type, 0) + count
                    )

            if all_redaction_counts:
                redaction_df = pd.DataFrame(
                    list(all_redaction_counts.items()),
                    columns=["Entity Type", "Count"],
                ).sort_values("Count", ascending=True)

                fig = px.bar(
                    redaction_df,
                    x="Count",
                    y="Entity Type",
                    orientation="h",
                    color="Count",
                    color_continuous_scale="Reds",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No redactions recorded yet.")
        else:
            st.info("No redactions recorded yet.")

    except Exception as e:
        st.error(f"Database connection error: {e}")
        st.info("Make sure PostgreSQL is running and the database is initialized.")


# ============================================================
# PAGE 2: Request Log
# ============================================================
elif page == "📋 Request Log":
    st.title("📋 Request Log")
    st.markdown("Browse and filter all audit records.")

    # --- Filters ---
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        decision_filter = st.multiselect(
            "Decision",
            ["ALLOW", "MASK", "BLOCK"],
            default=["ALLOW", "MASK", "BLOCK"],
        )
    with filter_col2:
        user_filter = st.text_input("User ID", "")
    with filter_col3:
        model_filter = st.text_input("Model", "")
    with filter_col4:
        limit = st.number_input("Max rows", value=100, min_value=10, max_value=1000)

    try:
        # Build query
        query_parts = ["SELECT request_id, created_at, user_id, model, input_decision,"]
        query_parts.append(
            "output_decision, input_redactions, total_latency_ms"
        )
        query_parts.append("FROM audit_log")
        query_parts.append("WHERE input_decision = ANY(:decisions)")

        params = {"decisions": decision_filter, "limit": limit}

        if user_filter:
            query_parts.append("AND user_id ILIKE :user_id")
            params["user_id"] = f"%{user_filter}%"

        if model_filter:
            query_parts.append("AND model ILIKE :model")
            params["model"] = f"%{model_filter}%"

        query_parts.append("ORDER BY created_at DESC")
        query_parts.append("LIMIT :limit")

        query = text(" ".join(query_parts))

        with get_connection() as conn:
            df = pd.read_sql(query, conn, params=params)

        if not df.empty:
            # Format the dataframe
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            df["latency"] = df["total_latency_ms"].apply(
                lambda x: f"{x}ms" if x else "N/A"
            )

            # Display count
            st.markdown(f"**Showing {len(df)} records**")

            # Display table
            st.dataframe(
                df[
                    [
                        "created_at",
                        "user_id",
                        "model",
                        "input_decision",
                        "output_decision",
                        "input_redactions",
                        "latency",
                        "request_id",
                    ]
                ],
                use_container_width=True,
                column_config={
                    "created_at": "Time",
                    "user_id": "User",
                    "model": "Model",
                    "input_decision": st.column_config.TextColumn(
                        "Input Decision"
                    ),
                    "output_decision": "Output Decision",
                    "input_redactions": "Redactions",
                    "latency": "Latency",
                    "request_id": "Request ID",
                },
            )
        else:
            st.info("No records found matching the filters.")

    except Exception as e:
        st.error(f"Database error: {e}")


# ============================================================
# PAGE 3: Request Detail
# ============================================================
elif page == "🔍 Request Detail":
    st.title("🔍 Request Detail")
    st.markdown("Inspect individual audit records in detail.")

    # Get request IDs for selection
    request_id_input = st.text_input(
        "Enter Request ID (UUID)", placeholder="e.g. 550e8400-e29b-41d4-a716-446655440000"
    )

    if not request_id_input:
        # Show recent requests for selection
        try:
            with get_connection() as conn:
                recent = pd.read_sql(
                    text("""
                        SELECT request_id, created_at, user_id, input_decision
                        FROM audit_log
                        ORDER BY created_at DESC
                        LIMIT 20
                    """),
                    conn,
                )

            if not recent.empty:
                st.markdown("**Or select a recent request:**")
                selected = st.selectbox(
                    "Recent requests",
                    recent["request_id"].astype(str).tolist(),
                    format_func=lambda x: f"{x[:8]}... — {recent[recent['request_id'].astype(str) == x]['input_decision'].values[0]}",
                )
                request_id_input = selected
        except Exception:
            pass

    if request_id_input:
        try:
            with get_connection() as conn:
                detail = pd.read_sql(
                    text("SELECT * FROM audit_log WHERE request_id = :rid"),
                    conn,
                    params={"rid": request_id_input},
                )

            if not detail.empty:
                row = detail.iloc[0]

                # --- Header ---
                decision_emoji = {
                    "ALLOW": "✅",
                    "MASK": "🟡",
                    "BLOCK": "🔴",
                }.get(row["input_decision"], "❓")

                st.subheader(
                    f"{decision_emoji} Request {str(row['request_id'])[:8]}..."
                )

                # --- Info Grid ---
                info_col1, info_col2, info_col3 = st.columns(3)

                with info_col1:
                    st.markdown("**Request ID:**")
                    st.code(str(row["request_id"]))
                    st.markdown("**User:**")
                    st.write(row["user_id"])
                    st.markdown("**Model:**")
                    st.write(row["model"])

                with info_col2:
                    st.markdown("**Input Decision:**")
                    st.write(f"{decision_emoji} {row['input_decision']}")
                    st.markdown("**Output Decision:**")
                    st.write(row["output_decision"] or "N/A")
                    st.markdown("**Policy:**")
                    st.write(row["policy_id"])

                with info_col3:
                    st.markdown("**Detection Latency:**")
                    st.write(f"{row['detection_latency_ms']}ms")
                    st.markdown("**LLM Latency:**")
                    st.write(
                        f"{row['llm_latency_ms']}ms"
                        if row["llm_latency_ms"]
                        else "N/A (blocked)"
                    )
                    st.markdown("**Total Latency:**")
                    st.write(f"{row['total_latency_ms']}ms")

                st.markdown("---")

                # --- Reasons ---
                st.subheader("📝 Reasons")
                reasons = row["reasons"]
                if reasons:
                    for reason in reasons:
                        st.markdown(f"- {reason}")
                else:
                    st.write("No findings — request allowed.")

                # --- Redaction Counts ---
                redact_col1, redact_col2 = st.columns(2)

                with redact_col1:
                    st.subheader("📊 Input Redactions")
                    input_redactions = row["input_redactions"]
                    if input_redactions:
                        for entity_type, count in input_redactions.items():
                            st.write(f"- **{entity_type}**: {count}")
                    else:
                        st.write("None")

                with redact_col2:
                    st.subheader("📊 Output Redactions")
                    output_redactions = row["output_redactions"]
                    if output_redactions:
                        for entity_type, count in output_redactions.items():
                            st.write(f"- **{entity_type}**: {count}")
                    else:
                        st.write("None")

                st.markdown("---")

                # --- Redacted Content ---
                st.subheader("📄 Redacted Prompt")
                st.code(row["prompt_redacted"], language="text")

                if row["response_redacted"]:
                    st.subheader("📄 Redacted Response")
                    st.code(row["response_redacted"], language="text")
                else:
                    st.info("No response (request was blocked).")

                # --- Prompt Hash ---
                st.subheader("🔐 Verification")
                st.markdown(f"**Prompt Hash (SHA-256):** `{row['prompt_hash']}`")
                st.markdown(
                    f"**Created At:** {row['created_at']}"
                )

            else:
                st.warning("No record found for that Request ID.")

        except Exception as e:
            st.error(f"Error loading request detail: {e}")
