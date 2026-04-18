"""
SentinelLM — Admin Dashboard (Streamlit)

Three-page dashboard for browsing and inspecting audit records:
  1. Overview Dashboard — charts, stats, decision distribution
  2. Request Log — filterable table with color-coded decisions
  3. Request Detail — full audit record inspection

Per PRD Section 13.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

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
    .status-dot-ok { color: #00d26a; font-size: 12px; }
    .status-dot-err { color: #ff4444; font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# --- DB Connection Helper ---
def get_connection():
    return engine.connect()


def check_db_connection() -> bool:
    try:
        with get_connection() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# --- Sidebar Navigation ---
st.sidebar.title("🛡️ SentinelLM")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "📋 Request Log", "🔍 Request Detail"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# DB connection status
db_ok = check_db_connection()
if db_ok:
    st.sidebar.markdown("🟢 **Database Connected**")
else:
    st.sidebar.markdown("🔴 **Database Offline**")

st.sidebar.markdown("---")

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**SentinelLM** v1.0  \n"
    "AI Gateway Security Dashboard  \n"
    f"*Updated: {datetime.now().strftime('%H:%M:%S')}*"
)


def color_decision(val):
    """Color-code decision values for dataframe styling."""
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

    if not db_ok:
        st.error("⚠️ Cannot connect to the database. Make sure PostgreSQL is running.")
        st.stop()

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

            # Average latencies
            latencies = pd.read_sql(
                text("""
                    SELECT
                        ROUND(AVG(detection_latency_ms)::numeric, 1) as avg_detection_ms,
                        ROUND(AVG(llm_latency_ms)::numeric, 1) as avg_llm_ms,
                        ROUND(AVG(total_latency_ms)::numeric, 1) as avg_total_ms
                    FROM audit_log
                    WHERE total_latency_ms IS NOT NULL
                """),
                conn,
            )

            # Top redaction types
            redactions = pd.read_sql(
                text("""
                    SELECT input_redactions
                    FROM audit_log
                    WHERE input_redactions IS NOT NULL
                      AND input_redactions != '{}'::jsonb
                """),
                conn,
            )

            # Recent blocked requests
            recent_blocks = pd.read_sql(
                text("""
                    SELECT request_id, created_at, user_id, reasons
                    FROM audit_log
                    WHERE input_decision = 'BLOCK'
                    ORDER BY created_at DESC
                    LIMIT 5
                """),
                conn,
            )

        # --- Metrics Row ---
        col1, col2, col3, col4, col5 = st.columns(5)

        allow_count = int(
            decisions[decisions["input_decision"] == "ALLOW"]["count"].sum()
        ) if not decisions.empty else 0
        mask_count = int(
            decisions[decisions["input_decision"] == "MASK"]["count"].sum()
        ) if not decisions.empty else 0
        block_count = int(
            decisions[decisions["input_decision"] == "BLOCK"]["count"].sum()
        ) if not decisions.empty else 0

        avg_total = (
            float(latencies["avg_total_ms"].iloc[0])
            if not latencies.empty and latencies["avg_total_ms"].iloc[0] is not None
            else None
        )

        col1.metric("Total Requests", f"{total_count:,}")
        col2.metric("✅ Allowed", f"{allow_count:,}")
        col3.metric("🟡 Masked", f"{mask_count:,}")
        col4.metric("🔴 Blocked", f"{block_count:,}")
        col5.metric(
            "⚡ Avg Latency",
            f"{avg_total:.0f}ms" if avg_total is not None else "N/A"
        )

        st.markdown("---")

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
                    margin=dict(t=30, b=0),
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
                    color_discrete_sequence=["#00d26a"],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    xaxis_title="Date",
                    yaxis_title="Requests",
                    margin=dict(t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data in the last 7 days.")

        # --- Latency Breakdown ---
        st.subheader("⚡ Latency Breakdown (Averages)")
        if not latencies.empty and latencies["avg_total_ms"].iloc[0] is not None:
            lat_col1, lat_col2, lat_col3 = st.columns(3)
            avg_det = float(latencies["avg_detection_ms"].iloc[0] or 0)
            avg_llm = float(latencies["avg_llm_ms"].iloc[0] or 0)
            avg_tot = float(latencies["avg_total_ms"].iloc[0] or 0)
            lat_col1.metric("Detection Pipeline", f"{avg_det:.1f}ms")
            lat_col2.metric("LLM Backend", f"{avg_llm:.1f}ms")
            lat_col3.metric("Total End-to-End", f"{avg_tot:.1f}ms")
        else:
            st.info("No latency data yet.")

        st.markdown("---")

        # --- Top Redaction Types ---
        st.subheader("🔍 Top Redaction Types")
        if not redactions.empty:
            all_redaction_counts: dict = {}
            for _, row in redactions.iterrows():
                rd = row["input_redactions"]
                if isinstance(rd, dict):
                    for entity_type, count in rd.items():
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
                    text="Count",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    showlegend=False,
                    margin=dict(t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No redactions recorded yet.")
        else:
            st.info("No redactions recorded yet.")

        # --- Recent Blocked Requests ---
        if not recent_blocks.empty:
            st.markdown("---")
            st.subheader("🔴 Recent Blocked Requests")
            recent_blocks["created_at"] = pd.to_datetime(
                recent_blocks["created_at"]
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(
                recent_blocks[["created_at", "user_id", "request_id", "reasons"]],
                use_container_width=True,
                column_config={
                    "created_at": "Time",
                    "user_id": "User",
                    "request_id": "Request ID",
                    "reasons": "Reasons",
                },
            )

    except Exception as e:
        st.error(f"Database error: {e}")
        st.info("Make sure PostgreSQL is running and the database is initialized.")


# ============================================================
# PAGE 2: Request Log
# ============================================================
elif page == "📋 Request Log":
    st.title("📋 Request Log")
    st.markdown("Browse and filter all audit records.")

    if not db_ok:
        st.error("⚠️ Cannot connect to the database. Make sure PostgreSQL is running.")
        st.stop()

    # --- Filters ---
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    filter_col4, filter_col5, filter_col6 = st.columns(3)

    with filter_col1:
        decision_filter = st.multiselect(
            "Decision",
            ["ALLOW", "MASK", "BLOCK"],
            default=["ALLOW", "MASK", "BLOCK"],
        )
    with filter_col2:
        user_filter = st.text_input("User ID (partial match)", "")
    with filter_col3:
        model_filter = st.text_input("Model (partial match)", "")

    with filter_col4:
        date_from = st.date_input(
            "From date",
            value=datetime.now().date() - timedelta(days=30),
        )
    with filter_col5:
        date_to = st.date_input(
            "To date",
            value=datetime.now().date(),
        )
    with filter_col6:
        limit = st.number_input("Max rows", value=100, min_value=10, max_value=1000)

    try:
        # Build parameterized query
        conditions = [
            "input_decision = ANY(:decisions)",
            "DATE(created_at) >= :date_from",
            "DATE(created_at) <= :date_to",
        ]
        params: dict = {
            "decisions": decision_filter or ["ALLOW", "MASK", "BLOCK"],
            "date_from": date_from,
            "date_to": date_to,
            "limit": int(limit),
        }

        if user_filter.strip():
            conditions.append("user_id ILIKE :user_id")
            params["user_id"] = f"%{user_filter.strip()}%"

        if model_filter.strip():
            conditions.append("model ILIKE :model")
            params["model"] = f"%{model_filter.strip()}%"

        where_clause = " AND ".join(conditions)
        query = text(f"""
            SELECT
                request_id,
                created_at,
                user_id,
                model,
                input_decision,
                output_decision,
                input_redactions,
                total_latency_ms,
                detection_latency_ms
            FROM audit_log
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit
        """)

        with get_connection() as conn:
            df = pd.read_sql(query, conn, params=params)

        if not df.empty:
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            df["latency"] = df["total_latency_ms"].apply(
                lambda x: f"{x}ms" if x is not None else "N/A"
            )
            df["detection"] = df["detection_latency_ms"].apply(
                lambda x: f"{x}ms" if x is not None else "N/A"
            )

            # Summary row
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.markdown(f"**{len(df)} records** found")
            if "total_latency_ms" in df.columns and df["total_latency_ms"].notna().any():
                avg_lat = df["total_latency_ms"].mean()
                summary_col2.markdown(f"**Avg latency:** {avg_lat:.0f}ms")
            block_pct = (
                (df["input_decision"] == "BLOCK").sum() / len(df) * 100
            )
            summary_col3.markdown(f"**Block rate:** {block_pct:.1f}%")

            st.markdown("---")

            styled_df = df[[
                "created_at",
                "user_id",
                "model",
                "input_decision",
                "output_decision",
                "input_redactions",
                "latency",
                "detection",
                "request_id",
            ]]

            st.dataframe(
                styled_df,
                use_container_width=True,
                column_config={
                    "created_at": st.column_config.TextColumn("Time", width="medium"),
                    "user_id": st.column_config.TextColumn("User", width="small"),
                    "model": st.column_config.TextColumn("Model", width="small"),
                    "input_decision": st.column_config.TextColumn(
                        "Input Decision", width="small"
                    ),
                    "output_decision": st.column_config.TextColumn(
                        "Output Decision", width="small"
                    ),
                    "input_redactions": st.column_config.TextColumn(
                        "Redactions", width="medium"
                    ),
                    "latency": st.column_config.TextColumn("Total Latency", width="small"),
                    "detection": st.column_config.TextColumn("Detection", width="small"),
                    "request_id": st.column_config.TextColumn("Request ID", width="large"),
                },
                height=500,
            )

            # CSV export
            csv = styled_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Export as CSV",
                data=csv,
                file_name=f"sentinellm_audit_{date_from}_{date_to}.csv",
                mime="text/csv",
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
    st.markdown("Inspect individual audit records in full detail.")

    if not db_ok:
        st.error("⚠️ Cannot connect to the database. Make sure PostgreSQL is running.")
        st.stop()

    # Input + recent selector side by side
    input_col, recent_col = st.columns([2, 1])

    with input_col:
        request_id_input = st.text_input(
            "Enter Request ID (UUID)",
            placeholder="e.g. 550e8400-e29b-41d4-a716-446655440000",
        )

    with recent_col:
        try:
            with get_connection() as conn:
                recent = pd.read_sql(
                    text("""
                        SELECT
                            request_id::text,
                            created_at,
                            input_decision
                        FROM audit_log
                        ORDER BY created_at DESC
                        LIMIT 20
                    """),
                    conn,
                )

            if not recent.empty:
                options = recent["request_id"].tolist()
                labels = {
                    row["request_id"]: (
                        f"{row['request_id'][:8]}... — "
                        f"{row['input_decision']} — "
                        f"{pd.to_datetime(row['created_at']).strftime('%m/%d %H:%M')}"
                    )
                    for _, row in recent.iterrows()
                }
                selected = st.selectbox(
                    "Or pick a recent request",
                    options=[""] + options,
                    format_func=lambda x: "Select..." if x == "" else labels.get(x, x),
                )
                if selected:
                    request_id_input = selected
        except Exception:
            st.caption("Could not load recent requests.")

    if not request_id_input:
        st.info("Enter a Request ID above or pick one from the dropdown to inspect it.")
        st.stop()

    try:
        with get_connection() as conn:
            detail = pd.read_sql(
                text("SELECT * FROM audit_log WHERE request_id::text = :rid"),
                conn,
                params={"rid": str(request_id_input).strip()},
            )

        if detail.empty:
            st.warning("No record found for that Request ID.")
            st.stop()

        row = detail.iloc[0]

        # --- Header ---
        decision_emoji = {"ALLOW": "✅", "MASK": "🟡", "BLOCK": "🔴"}.get(
            row["input_decision"], "❓"
        )
        decision_color = {"ALLOW": "#00d26a", "MASK": "#f9a825", "BLOCK": "#ff4444"}.get(
            row["input_decision"], "white"
        )

        st.markdown(
            f"<h2>{decision_emoji} Request "
            f"<code>{str(row['request_id'])[:8]}...</code> &nbsp;"
            f"<span style='color:{decision_color};font-size:18px;'>"
            f"{row['input_decision']}</span></h2>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # --- Info Grid ---
        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.markdown("**🆔 Request ID**")
            st.code(str(row["request_id"]), language=None)
            st.markdown("**👤 User**")
            st.write(row.get("user_id") or "anonymous")
            st.markdown("**🤖 Model**")
            st.write(row.get("model") or "N/A")

        with info_col2:
            st.markdown("**📥 Input Decision**")
            st.markdown(
                f"<span style='color:{decision_color};font-size:16px;font-weight:bold;'>"
                f"{decision_emoji} {row['input_decision']}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("**📤 Output Decision**")
            out_dec = row.get("output_decision") or "N/A"
            out_emoji = {"ALLOW": "✅", "MASK": "🟡", "BLOCK": "🔴"}.get(out_dec, "")
            st.write(f"{out_emoji} {out_dec}")
            st.markdown("**📋 Policy**")
            st.write(row.get("policy_id") or "default")

        with info_col3:
            st.markdown("**⚡ Detection Latency**")
            det_ms = row.get("detection_latency_ms")
            st.write(f"{det_ms}ms" if det_ms is not None else "N/A")
            st.markdown("**🧠 LLM Latency**")
            llm_ms = row.get("llm_latency_ms")
            st.write(f"{llm_ms}ms" if llm_ms is not None else "N/A (blocked)")
            st.markdown("**🕐 Total Latency**")
            tot_ms = row.get("total_latency_ms")
            st.write(f"{tot_ms}ms" if tot_ms is not None else "N/A")

        st.markdown("---")

        # --- Latency bar chart ---
        if det_ms is not None and tot_ms is not None:
            lat_data = pd.DataFrame({
                "Stage": ["Detection Pipeline", "LLM Backend", "Other"],
                "ms": [
                    det_ms or 0,
                    llm_ms or 0,
                    max(0, (tot_ms or 0) - (det_ms or 0) - (llm_ms or 0)),
                ],
            })
            fig = px.bar(
                lat_data,
                x="ms",
                y="Stage",
                orientation="h",
                color="Stage",
                color_discrete_sequence=["#00d26a", "#4fc3f7", "#f9a825"],
                title="Latency Breakdown",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                showlegend=False,
                height=200,
                margin=dict(t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Findings / Reasons ---
        st.subheader("📝 Findings")
        reasons = row.get("reasons")
        if reasons:
            if isinstance(reasons, list):
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.write(reasons)
        else:
            st.success("No findings — request passed through clean.")

        # --- Redaction Counts ---
        redact_col1, redact_col2 = st.columns(2)

        with redact_col1:
            st.subheader("📊 Input Redactions")
            input_redactions = row.get("input_redactions")
            if input_redactions and isinstance(input_redactions, dict):
                for entity_type, count in input_redactions.items():
                    st.markdown(f"- **{entity_type}**: {count}")
            else:
                st.write("None")

        with redact_col2:
            st.subheader("📊 Output Redactions")
            output_redactions = row.get("output_redactions")
            if output_redactions and isinstance(output_redactions, dict):
                for entity_type, count in output_redactions.items():
                    st.markdown(f"- **{entity_type}**: {count}")
            else:
                st.write("None")

        st.markdown("---")

        # --- Redacted Content ---
        st.subheader("📄 Redacted Prompt")
        prompt = row.get("prompt_redacted")
        if prompt:
            st.code(prompt, language="text")
        else:
            st.info("No prompt stored.")

        st.subheader("📄 Redacted Response")
        response = row.get("response_redacted")
        if response:
            st.code(response, language="text")
        else:
            st.info(
                "No response stored — request was blocked before reaching the LLM."
                if row["input_decision"] == "BLOCK"
                else "No response stored."
            )

        # --- Verification ---
        st.markdown("---")
        st.subheader("🔐 Verification")
        ver_col1, ver_col2 = st.columns(2)
        with ver_col1:
            st.markdown(f"**Prompt Hash (SHA-256):**")
            st.code(row.get("prompt_hash") or "N/A", language=None)
        with ver_col2:
            st.markdown(f"**Created At:**")
            st.write(str(row.get("created_at", "N/A")))

    except Exception as e:
        st.error(f"Error loading request detail: {e}")
