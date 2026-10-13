"""
Voice-of-Store Engine — Master Web Application
Talent AP Mega Hackathon — Problem Statement R5
Tagline: "From scattered feedback to actionable store intelligence."
"""

import os
import io
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Voice-of-Store Engine | R5",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import internal modules
from src.config import (
    APP_TITLE,
    APP_SUBTITLE,
    PROBLEM_STATEMENT,
    TAGLINE,
    COLORS,
    TAXONOMY_CATEGORIES,
    DEPARTMENTS,
    SEVERITY_LEVELS,
    CATEGORY_DEPARTMENT_MAP,
    MANDATORY_POLICY_RULES
)
from src.preprocessing import normalize_feedback_text
from src.classification import HybridIssueClassifier
from src.routing import apply_routing_batch, evaluate_mandatory_policies
from src.emerging_detection import EarlyWarningRadar
from src.deduplication import CrossChannelDeduplicator
from src.reporting import generate_weekly_regional_brief, generate_pdf_report
from src.evaluation import (
    run_official_evaluator,
    compute_model_metrics,
    evaluate_escalation_precision,
    simulate_seeded_incident_lead_time
)
from src.data_loader import (
    generate_synthetic_demo_dataset,
    load_file_into_dataframe,
    auto_detect_column_mapping,
    compute_data_quality_audit,
    normalize_schema
)

# ---------------------------------------------------------
# Custom Styling: Enterprise Navy & Teal Clean Aesthetic
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global styles */
    .main {
        background-color: #F8F9FA;
        color: #1F2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0B192C;
        color: #F3F4F6;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #E2E8F0;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF;
    }
    
    /* Metrics and Cards */
    .metric-card {
        background: #FFFFFF;
        padding: 18px 22px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 14px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #0B192C;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #6B7280;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-delta {
        font-size: 12px;
        font-weight: 600;
        margin-top: 6px;
    }
    
    /* Status Badges */
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid #FCA5A5;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid #FDBA74;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid #FCD34D;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid #86EFAC;
    }
    .badge-mandatory {
        background-color: #7F1D1D;
        color: #FFFFFF;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    
    /* Header Container */
    .app-header {
        background: linear-gradient(135deg, #0B192C 0%, #1E3E62 100%);
        color: #FFFFFF;
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .app-header h1 {
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #94A3B8;
        font-size: 14px;
        margin-top: 4px;
        margin-bottom: 0;
    }
    
    /* Alert cards */
    .banner-demo {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 18px;
        color: #92400E;
        font-size: 13px;
        font-weight: 600;
    }
    .banner-live {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 18px;
        color: #065F46;
        font-size: 13px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
@st.cache_resource
def get_classifier():
    return HybridIssueClassifier()

@st.cache_resource
def get_radar():
    return EarlyWarningRadar()

@st.cache_resource
def get_deduplicator():
    return CrossChannelDeduplicator()

classifier = get_classifier()
radar = get_radar()
deduplicator = get_deduplicator()

if "is_demo_mode" not in st.session_state:
    st.session_state.is_demo_mode = True
if "raw_df" not in st.session_state:
    st.session_state.raw_df = generate_synthetic_demo_dataset()
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "SYNTHETIC DEMO DATA — NOT OFFICIAL ORGANIZER DATA"
if "dept_map" not in st.session_state:
    st.session_state.dept_map = CATEGORY_DEPARTMENT_MAP.copy()


def run_full_pipeline(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Executes full analytical pipeline:
    1. Normalization & Vernacular parsing
    2. Hybrid Classification
    3. Mandatory Severity Overrides & Department Routing
    4. Cross-Channel Incident Deduplication
    """
    df_work = df_in.copy()
    
    # 1. Text Normalization
    normalized_texts = []
    clarity_scores = []
    languages = []
    code_mixed_flags = []
    review_flags = []
    review_reasons = []

    for text in df_work["feedback_text"].astype(str):
        norm, meta = normalize_feedback_text(text)
        normalized_texts.append(norm)
        clarity_scores.append(meta["clarity_score"])
        languages.append(meta["primary_language"])
        code_mixed_flags.append(meta["is_code_mixed"])
        review_flags.append(meta["needs_human_review"])
        review_reasons.append(meta["review_reason"])

    df_work["normalized_text"] = normalized_texts
    df_work["text_clarity"] = clarity_scores
    df_work["detected_language"] = languages
    df_work["is_code_mixed"] = code_mixed_flags
    df_work["needs_human_review"] = review_flags
    df_work["human_review_reason"] = review_reasons

    # 2. Classification
    df_pred = classifier.predict_batch(df_work, text_col="normalized_text")

    # 3. Routing & Severity Policy Overrides
    df_routed = apply_routing_batch(df_pred, custom_dept_map=st.session_state.dept_map)

    # 4. Deduplication
    df_dedup, _ = deduplicator.process(
        df_routed,
        store_col="store_id",
        text_col="feedback_text",
        time_col="timestamp",
        channel_col="channel",
        id_col="feedback_id"
    )

    return df_dedup


if "processed_df" not in st.session_state or st.session_state.processed_df is None:
    st.session_state.processed_df = run_full_pipeline(st.session_state.raw_df)


# ---------------------------------------------------------
# Sidebar Navigation & Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🏢 {APP_TITLE}")
    st.markdown(f"*{APP_SUBTITLE}*")
    st.caption(f"**Problem Statement:** {PROBLEM_STATEMENT}")
    st.markdown("---")

    # Dataset Status
    if st.session_state.is_demo_mode:
        st.markdown(
            '<div class="banner-demo">⚠️ SYNTHETIC DEMO DATA<br><span style="font-size:11px;font-weight:normal;">Not official organizer data</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="banner-live">✅ ACTIVE DATASET<br><span style="font-size:11px;font-weight:normal;">{st.session_state.dataset_name}</span></div>',
            unsafe_allow_html=True
        )

    # Navigation menu
    selected_page = st.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "🔍 Feedback Explorer",
            "🧠 Issue Intelligence & Routing",
            "🚨 Early-Warning Radar",
            "📑 Weekly Regional Brief",
            "⚖️ Evaluation Centre",
            "💾 Data Management & Ingestion"
        ]
    )

    st.markdown("---")
    st.subheader("Filter Controls")

    df_active = st.session_state.processed_df

    # Region filter
    available_regions = ["All Regions"] + sorted([r for r in df_active["region"].dropna().unique().tolist() if r])
    sel_region = st.selectbox("Region Filter", available_regions)

    # Channel filter
    available_channels = ["All Channels"] + sorted([c for c in df_active["channel"].dropna().unique().tolist() if c])
    sel_channel = st.selectbox("Channel Filter", available_channels)

    # Severity filter
    available_severities = ["All Severities"] + SEVERITY_LEVELS
    sel_severity = st.selectbox("Severity Filter", available_severities)

    st.markdown("---")
    st.caption("Talent AP Mega Hackathon &bull; Version 1.0.0")


# Apply Global Filters to View DataFrame
df_view = df_active.copy()
if sel_region != "All Regions":
    df_view = df_view[df_view["region"] == sel_region]
if sel_channel != "All Channels":
    df_view = df_view[df_view["channel"] == sel_channel]
if sel_severity != "All Severities":
    df_view = df_view[df_view["severity"] == sel_severity]


# ---------------------------------------------------------
# App Header Component
# ---------------------------------------------------------
def render_page_header(title: str, description: str):
    st.markdown(f"""
    <div class="app-header">
        <h1>{title}</h1>
        <p>{description} &bull; <strong>{PROBLEM_STATEMENT}</strong></p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =========================================================
if selected_page == "📊 Executive Overview":
    render_page_header(
        "Executive Retail Intelligence Overview",
        "Multi-channel store chatter aggregated into operational decision metrics"
    )

    if st.session_state.is_demo_mode:
        st.info("ℹ️ Currently displaying **SYNTHETIC DEMO DATA — NOT OFFICIAL ORGANIZER DATA**. To load organizer files, navigate to **Data Management & Ingestion** in the sidebar.")

    # High-level KPIs
    total_feedback = len(df_view)
    unique_incidents = df_view["incident_id"].nunique() if "incident_id" in df_view.columns else total_feedback
    affected_stores = df_view["store_id"].nunique() if "store_id" in df_view.columns else 0
    critical_cases = len(df_view[df_view["severity"] == "Critical"])
    high_cases = len(df_view[df_view["severity"] == "High"])
    actionable_count = len(df_view[df_view["actionability"].isin(["Highly Actionable", "Actionable"])])
    actionability_rate = round((actionable_count / total_feedback * 100), 1) if total_feedback > 0 else 0

    # Radar Alerts count
    radar_res = radar.analyze(df_view)
    active_alerts_count = len(radar_res["alerts"])

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_feedback}</div>
            <div class="metric-label">Raw Feedback</div>
            <div class="metric-delta" style="color:#028090;">{len(df_active)} Total in Scope</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_incidents}</div>
            <div class="metric-label">Unique Incidents</div>
            <div class="metric-delta" style="color:#10B981;">{total_feedback - unique_incidents} Duplicates Clustered</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{affected_stores}</div>
            <div class="metric-label">Affected Stores</div>
            <div class="metric-delta" style="color:#6B7280;">Across {df_view['region'].nunique()} Regions</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#DC2626;">{critical_cases + high_cases}</div>
            <div class="metric-label">High / Critical Risks</div>
            <div class="metric-delta" style="color:#DC2626;">{critical_cases} Critical Safety/Health</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#00A896;">{actionability_rate}%</div>
            <div class="metric-label">Actionability Rate</div>
            <div class="metric-delta" style="color:#00A896;">{actionable_count} Concrete Incidents</div>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#EA580C;">{active_alerts_count}</div>
            <div class="metric-label">Radar Alerts</div>
            <div class="metric-delta" style="color:#EA580C;">{radar_res['summary']['emergency_alerts']} Emergency Overrides</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Operational Distributions")
    c_chart1, c_chart2 = st.columns([3, 2])

    with c_chart1:
        # Category volume bar chart
        cat_counts = df_view["predicted_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Volume"]
        fig_cat = px.bar(
            cat_counts,
            x="Volume",
            y="Category",
            orientation="h",
            title="Feedback Volume by Issue Taxonomy Category",
            color="Volume",
            color_continuous_scale=["#028090", "#00A896", "#0B192C"]
        )
        fig_cat.update_layout(yaxis=dict(autorange="reversed"), height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cat, use_container_width=True)

    with c_chart2:
        # Channel share donut chart
        chan_counts = df_view["channel"].value_counts().reset_index()
        chan_counts.columns = ["Channel", "Count"]
        fig_chan = px.pie(
            chan_counts,
            names="Channel",
            values="Count",
            hole=0.55,
            title="Feedback Ingestion Channels",
            color_discrete_sequence=["#0B192C", "#028090", "#00A896", "#F59E0B"]
        )
        fig_chan.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_chan, use_container_width=True)

    # Timeline Chart
    if "timestamp" in df_view.columns:
        st.markdown("### Daily Feedback Ingestion & Cross-Channel Incident Trend")
        df_time = df_view.copy()
        df_time["date"] = pd.to_datetime(df_time["timestamp"], errors='coerce').dt.date
        time_trend = df_time.groupby("date").agg(
            Raw_Feedback=("feedback_id", "count"),
            Unique_Incidents=("incident_id", "nunique")
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=time_trend["date"],
            y=time_trend["Raw_Feedback"],
            mode="lines+markers",
            name="Raw Feedback Volume",
            line=dict(color="#0B192C", width=3)
        ))
        fig_trend.add_trace(go.Scatter(
            x=time_trend["date"],
            y=time_trend["Unique_Incidents"],
            mode="lines+markers",
            name="Deduplicated Operational Incidents",
            line=dict(color="#00A896", width=2.5, dash="dash")
        ))
        fig_trend.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Top Escalated Action Items
    st.markdown("### Urgent Escalations Requiring Immediate Action")
    urgent_records = df_view[df_view["severity"].isin(["Critical", "High"])].head(6)
    if not urgent_records.empty:
        for _, urow in urgent_records.iterrows():
            badge_class = "badge-critical" if urow["severity"] == "Critical" else "badge-high"
            override_tag = '<span class="badge-mandatory">MANDATORY OVERRIDE</span> ' if urow.get("is_mandatory_override") else ''
            
            st.markdown(f"""
            <div style="background:#FFFFFF; border-left: 4px solid {'#DC2626' if urow['severity']=='Critical' else '#EA580C'}; padding: 12px 18px; border-radius: 6px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                    <div>
                        <span class="{badge_class}">{urow['severity']}</span> {override_tag}
                        <strong style="color:#0B192C; margin-left:8px;">{urow.get('store_id')} ({urow.get('region')})</strong>
                        <span style="color:#6B7280; font-size:12px; margin-left:8px;">&bull; {urow.get('channel')}</span>
                    </div>
                    <span style="font-size:12px; font-weight:600; color:#1E3E62;">Owner: {urow.get('department')}</span>
                </div>
                <div style="font-size:13px; color:#1F2937; margin-bottom: 6px;"><em>"{urow.get('feedback_text')}"</em></div>
                <div style="font-size:11px; color:#4B5563;">
                    <strong>Next Action:</strong> {urow.get('recommended_action', 'Review incident')} &bull; 
                    <strong>Reason:</strong> {urow.get('escalation_reason', 'Standard policy')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No High or Critical severity alerts in the current filtered view.")


# =========================================================
# PAGE 2: FEEDBACK EXPLORER
# =========================================================
elif selected_page == "🔍 Feedback Explorer":
    render_page_header(
        "Feedback Explorer & Audit Trail",
        "Inspect original verbatim complaints, predictions, linguistic metadata, and routing decisions"
    )

    c_search, c_rev_filt = st.columns([3, 1])
    with c_search:
        search_query = st.text_input("Search feedback text (keywords, store ID, order/bill #):", placeholder="e.g. curd, 98421, rude, cockroach, AC...")
    with c_rev_filt:
        human_review_only = st.checkbox("Show only records flagged for Human Review", value=False)

    df_filtered = df_view.copy()
    if search_query:
        df_filtered = df_filtered[
            df_filtered["feedback_text"].astype(str).str.contains(search_query, case=False, na=False) |
            df_filtered["store_id"].astype(str).str.contains(search_query, case=False, na=False) |
            df_filtered["feedback_id"].astype(str).str.contains(search_query, case=False, na=False)
        ]
    if human_review_only:
        df_filtered = df_filtered[df_filtered["needs_human_review"] == True]

    st.caption(f"Showing **{len(df_filtered)}** matching feedback records")

    # Data table preview
    cols_to_show = [
        "feedback_id", "timestamp", "store_id", "region", "channel",
        "predicted_category", "severity", "department", "actionability",
        "needs_human_review"
    ]
    available_cols = [c for c in cols_to_show if c in df_filtered.columns]
    st.dataframe(df_filtered[available_cols], use_container_width=True, height=280)

    # Detailed Single-Record Inspector
    st.markdown("---")
    st.markdown("### Deep-Dive Record Inspector")
    if not df_filtered.empty:
        record_ids = df_filtered["feedback_id"].tolist()
        sel_rec_id = st.selectbox("Select Record ID to Inspect Verbatim & ML Audit Trail:", record_ids)
        rec = df_filtered[df_filtered["feedback_id"] == sel_rec_id].iloc[0]

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### Verbatim Evidence & Language Processing")
            st.text_area("Original Raw Verbatim (Preserved for Audit)", rec.get("feedback_text", ""), height=80, disabled=True)
            st.text_area("Normalized Text (Cleaned for Engine)", rec.get("normalized_text", ""), height=80, disabled=True)
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Language Detected", rec.get("detected_language", "English"))
            c_m2.metric("Code-Mixed / Vernacular", "Yes" if rec.get("is_code_mixed") else "No")
            c_m3.metric("Clarity Score", f"{rec.get('text_clarity', 1.0) * 100:.0f}%")

            if rec.get("needs_human_review"):
                st.warning(f"⚠️ Flagged for Human Review: {rec.get('human_review_reason')}")

        with col_right:
            st.markdown("#### Classification, Routing & Policy Diagnosis")
            
            p_cat = rec.get("predicted_category")
            p_conf = rec.get("confidence", 0.0)
            p_src = rec.get("prediction_source", "Rule Baseline")
            p_sev = rec.get("severity")
            p_dept = rec.get("department")
            p_act = rec.get("actionability")

            st.write(f"**Predicted Category:** `{p_cat}` (Confidence: {p_conf * 100:.0f}%)")
            st.write(f"**Model Source:** {p_src}")
            st.write(f"**Assigned Department:** `{p_dept}`")
            st.write(f"**Severity Level:** `{p_sev}` {'🚨 (Mandatory Policy Override)' if rec.get('is_mandatory_override') else ''}")
            st.write(f"**Actionability Assessment:** `{p_act}`")
            st.write(f"**Escalation Reason:** {rec.get('escalation_reason')}")
            st.write(f"**Recommended Action:** {rec.get('recommended_action')}")

            if "incident_id" in rec:
                st.info(f"🔗 **Incident Cluster ID:** `{rec['incident_id']}` (Cluster Size: {rec.get('cluster_size', 1)})")
    else:
        st.warning("No records match your search criteria.")


# =========================================================
# PAGE 3: ISSUE INTELLIGENCE & ROUTING
# =========================================================
elif selected_page == "🧠 Issue Intelligence & Routing":
    render_page_header(
        "Issue Intelligence & Department Routing Policy",
        "Inspect category distribution, department ownership, and mandatory zero-tolerance safety policies"
    )

    c_r1, c_r2 = st.columns([1, 1])

    with c_r1:
        st.markdown("### Department Responsibility Breakdown")
        dept_counts = df_view["department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Tickets"]
        fig_dept = px.pie(
            dept_counts,
            names="Department",
            values="Tickets",
            title="Feedback Routing by Owning Department",
            color_discrete_sequence=["#0B192C", "#00A896", "#028090", "#F59E0B", "#DC2626"]
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    with c_r2:
        st.markdown("### Severity Distribution by Department")
        sev_dept = df_view.groupby(["department", "severity"]).size().reset_index(name="Count")
        fig_sev_dept = px.bar(
            sev_dept,
            x="department",
            y="Count",
            color="severity",
            barmode="stack",
            title="Severity Mix per Department",
            color_discrete_map={
                "Critical": "#DC2626",
                "High": "#EA580C",
                "Medium": "#F59E0B",
                "Low": "#10B981"
            }
        )
        st.plotly_chart(fig_sev_dept, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🛡️ Mandatory High-Severity Policy Monitor")
    st.caption("Mandatory safety, allergen, sanitation, and staff-misconduct policies operate INDEPENDENTLY of ML models to ensure zero-tolerance operational compliance.")

    col_pol1, col_pol2 = st.columns([1, 1])
    for idx, rule in enumerate(MANDATORY_POLICY_RULES):
        target_col = col_pol1 if idx % 2 == 0 else col_pol2
        with target_col:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-left: 4px solid #DC2626; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color:#0B192C; font-size:14px;">{rule['name']}</strong>
                    <span class="badge-mandatory">{rule['mandatory_severity']}</span>
                </div>
                <div style="font-size:12px; color:#4B5563; margin-top:6px;"><strong>Owner:</strong> {rule['department']}</div>
                <div style="font-size:12px; color:#4B5563; margin-top:4px;"><strong>Policy Rationale:</strong> {rule['reason']}</div>
                <div style="font-size:12px; color:#065F46; margin-top:4px;"><strong>Mandatory Action:</strong> {rule['recommended_action']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Department Mapping Configuration
    st.markdown("---")
    st.markdown("### Inspectable Department Mapping Configuration")
    st.caption("Reviewers can adjust department ownership for any issue category in real-time.")

    c_map_left, c_map_right = st.columns([2, 1])
    with c_map_left:
        selected_cat_to_edit = st.selectbox("Select Taxonomy Category:", TAXONOMY_CATEGORIES)
        current_owner = st.session_state.dept_map.get(selected_cat_to_edit, "Store Operations")
        new_owner = st.selectbox("Assign Owning Department:", DEPARTMENTS, index=DEPARTMENTS.index(current_owner) if current_owner in DEPARTMENTS else 0)

        if st.button("Update Department Routing Rule"):
            st.session_state.dept_map[selected_cat_to_edit] = new_owner
            # Re-run routing
            st.session_state.processed_df = run_full_pipeline(st.session_state.raw_df)
            st.success(f"Updated '{selected_cat_to_edit}' routing to {new_owner}. Analytical views refreshed!")

    with c_map_right:
        st.markdown("**Current Mapping Table:**")
        map_df = pd.DataFrame(list(st.session_state.dept_map.items()), columns=["Category", "Owning Department"])
        st.dataframe(map_df, use_container_width=True, height=220)


# =========================================================
# PAGE 4: EARLY-WARNING RADAR
# =========================================================
elif selected_page == "🚨 Early-Warning Radar":
    render_page_header(
        "Early-Warning Radar: Emerging Store-Specific Issue Detector",
        "Detect low-volume localized surges and emerging anomalies at individual stores before regional escalation"
    )

    st.markdown("""
    > [!IMPORTANT]
    > **Core Problem Solved:** A single store experiencing 4 sudden refrigeration complaints may be buried under 1,000 regional complaints.
    > The **Early-Warning Radar** tracks within-store velocity, applies sparse-data safeguards, handles zero-baseline outbreaks,
    > and triggers mandatory health/safety emergency overrides.
    """)

    # Radar Configuration Controls
    with st.expander("⚙️ Radar Detection Sensitivity & Safeguards Configuration", expanded=False):
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            cfg_min_obs = st.slider("Minimum Current Window Observations", 1, 5, 2, help="Safeguard against single random feedback spikes.")
        with c_p2:
            cfg_min_growth = st.slider("Minimum Relative Growth %", 20.0, 200.0, 30.0, help="Required relative velocity increase.")
        with c_p3:
            cfg_z = st.slider("Poisson / Z-Score Spike Threshold", 0.5, 3.0, 1.0, help="Statistical unexpectedness index.")

        radar.min_current_count = cfg_min_obs
        radar.min_growth_pct = cfg_min_growth
        radar.spike_z_threshold = cfg_z

    # Run Analysis
    radar_results = radar.analyze(df_view)
    alerts = radar_results["alerts"]
    summary = radar_results["summary"]

    c_sum1, c_sum2, c_sum3, c_sum4 = st.columns(4)
    c_sum1.metric("Active Emerging Alerts", summary["total_alerts"])
    c_sum2.metric("Emergency Policy Alerts", summary["emergency_alerts"], help="Triggered immediately by zero-tolerance safety policies")
    c_sum3.metric("Statistical Velocity Spikes", summary["statistical_spikes"], help="Triggered by within-store growth velocity exceeding threshold")
    c_sum4.metric("Window End Date", summary.get("max_date", "Current"))

    st.caption(f"**Denominator Status:** `{radar_results['denominator_status']}`")

    if alerts:
        # Alert Table Display
        alert_rows = []
        for a in alerts:
            alert_rows.append({
                "Alert ID": a["alert_id"],
                "Store": a["store_id"],
                "Region": a["region"],
                "Category": a["category"],
                "Alert Type": a["alert_type"],
                "Curr Count": a["current_count"],
                "Prior Count": a["prior_count"],
                "Change": f"+{a['abs_change']}" if a['abs_change'] > 0 else str(a['abs_change']),
                "Growth": a["growth_display"],
                "Spike Index": a["spike_score"],
                "Owner": a["suggested_owner"],
                "Status": a["review_status"]
            })

        st.markdown("### Ranked Emerging Incident Alerts")
        st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)

        # Alert Investigation Workflow
        st.markdown("---")
        st.markdown("### Alert Investigation & Evidence Drill-Down")
        
        alert_ids = [a["alert_id"] for a in alerts]
        sel_alert_id = st.selectbox("Select Alert to Investigate:", alert_ids)
        sel_alert = next((a for a in alerts if a["alert_id"] == sel_alert_id), None)

        if sel_alert:
            c_det1, c_det2 = st.columns([3, 2])
            
            with c_det1:
                st.markdown(f"#### Store: `{sel_alert['store_id']}` &bull; Issue: `{sel_alert['category']}`")
                st.write(f"**Alert Classification:** `{sel_alert['alert_type']}`")
                st.write(f"**Assigned Department Owner:** `{sel_alert['suggested_owner']}`")
                st.write(f"**Recommended Immediate Action:** {sel_alert['recommended_action']}")
                st.write(f"**Channels Reporting:** {', '.join(sel_alert['channels_affected'])}")

                st.markdown("**Verbatim Evidence Quotes from Customers / Staff:**")
                for q in sel_alert["evidence_quotes"]:
                    st.markdown(f"""
                    <div style="background:#F1F5F9; border-left:3px solid #00A896; padding:8px 12px; margin-bottom:6px; font-size:13px; font-style:italic;">
                        "{q}"
                    </div>
                    """, unsafe_allow_html=True)

            with c_det2:
                st.markdown("#### Reviewer Triage Action")
                st.write(f"**Current Status:** `{sel_alert['review_status']}`")
                st.write(f"**Reviewer Note:** {sel_alert['reviewer_note']}")

                new_status = st.selectbox(
                    "Update Triage Status:",
                    ["Active / Unresolved", "Under Investigation", "Acknowledged (Work Order Raised)", "Dismissed (Known Maintenance / False Alarm)"]
                )
                investigation_note = st.text_input("Reviewer Notes / Work Order #:", placeholder="e.g. Dispatched technician to check compressor")

                if st.button("Save Alert Triage Status"):
                    radar.update_alert_status(sel_alert_id, new_status, investigation_note)
                    st.success(f"Updated status of {sel_alert_id} to '{new_status}'.")
                    st.rerun()
    else:
        st.success("No localized emerging alerts detected in the current window and filter selection.")


# =========================================================
# PAGE 5: WEEKLY REGIONAL BRIEF
# =========================================================
elif selected_page == "📑 Weekly Regional Brief":
    render_page_header(
        "Weekly Regional Intelligence Brief",
        "Executive one-page operational summary with week-on-week trends, evidence quotes, and PDF export"
    )

    c_b_reg, c_b_per = st.columns([2, 2])
    with c_b_reg:
        brief_regions = [r for r in df_active["region"].dropna().unique().tolist() if r]
        sel_brief_region = st.selectbox("Select Target Region for Brief:", brief_regions if brief_regions else ["All Regions"])
    with c_b_per:
        st.text_input("Reporting Period:", value="Latest Week (Trailing 7 Days)", disabled=True)

    # Generate Brief Data
    brief_data = generate_weekly_regional_brief(
        df_active,
        region=sel_brief_region,
        selected_period_label="Latest Week"
    )

    if brief_data["status"] == "success":
        m = brief_data["metrics"]
        
        # Brief KPI Summary Bar
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Records", m["total_records"])
        k2.metric("Unique Incidents", m["unique_incidents"])
        k3.metric("Stores Impacted", m["affected_stores"])
        k4.metric("High/Critical Cases", m["critical_high_cases"])
        k5.metric("Actionability Rate", f"{m['actionable_pct']}%")

        st.markdown("---")
        st.markdown(f"### Top Operational Issues & WoW Trends — {brief_data['region']}")

        # Top issues table
        issue_rows = []
        for itm in brief_data["top_issues"]:
            quote_display = itm["evidence_quotes"][0] if itm["evidence_quotes"] else "None"
            issue_rows.append({
                "Issue Category": itm["category"],
                "Owner": itm["department"],
                "Curr Volume": itm["volume"],
                "Prior Volume": itm["prior_volume"],
                "WoW Trend": f"{itm['trend_dir']} {itm['trend_display']}",
                "Stores Impacted": itm["affected_stores"],
                "Sample Evidence Quote": f'"{quote_display[:80]}..."' if len(quote_display) > 80 else f'"{quote_display}"'
            })

        st.dataframe(pd.DataFrame(issue_rows), use_container_width=True)

        # Audit Disclosures & Caveats
        st.markdown("### Compliance, Data Hygiene & Audit Disclosures")
        for cav in brief_data["caveats"]:
            st.markdown(f"- ⚠️ {cav}")

        # Export Actions
        st.markdown("---")
        st.markdown("### Export Intelligence Brief")
        c_exp1, c_exp2 = st.columns(2)

        with c_exp1:
            # Generate ReportLab PDF
            pdf_bytes = generate_pdf_report(brief_data)
            st.download_button(
                label="📄 Download Executive Brief as PDF",
                data=pdf_bytes,
                file_name=f"VoiceOfStore_Brief_{brief_data['region']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with c_exp2:
            # Export CSV
            csv_buf = pd.DataFrame(issue_rows).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download Regional Issues as CSV",
                data=csv_buf,
                file_name=f"VoiceOfStore_Issues_{brief_data['region']}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning(brief_data.get("message", "No data available for brief generation."))


# =========================================================
# PAGE 6: EVALUATION CENTRE
# =========================================================
elif selected_page == "⚖️ Evaluation Centre":
    render_page_header(
        "Evaluation & Benchmarking Centre",
        "Rigorous verification, classification metrics, confusion matrices, and organizer evaluate.py integration"
    )

    st.markdown("### 1. Organizer Official Evaluator ('evaluate.py')")
    st.caption("Inspects workspace for official evaluation contract and executes genuine evaluation without bypassing rules.")

    eval_res = run_official_evaluator("evaluate.py")
    if not eval_res["is_available"]:
        st.warning(f"ℹ️ {eval_res['message']}")
    else:
        st.success(f"Status: {eval_res['status']} (Exit Code: {eval_res['exit_code']})")
        if eval_res.get("stdout"):
            st.code(eval_res["stdout"], language="text")
        if eval_res.get("stderr"):
            st.error(eval_res["stderr"])

    st.markdown("---")
    st.markdown("### 2. Multi-Class Classification Performance")

    # Check if ground truth category is available in loaded data
    if "ground_truth_category" in df_active.columns:
        y_true = df_active["ground_truth_category"].dropna().tolist()
        y_pred = df_active.loc[df_active["ground_truth_category"].notna(), "predicted_category"].tolist()

        if len(y_true) >= 5:
            metrics = compute_model_metrics(y_true, y_pred)
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Overall Accuracy", f"{metrics['accuracy']*100:.1f}%")
            c_m2.metric("Macro F1-Score", f"{metrics['macro_f1']*100:.1f}%")
            c_m3.metric("Weighted F1-Score", f"{metrics['weighted_f1']*100:.1f}%")

            c_cm, c_tab = st.columns([1, 1])

            with c_cm:
                st.markdown("#### Confusion Matrix")
                labels = metrics["labels"]
                cm = metrics["confusion_matrix"]
                fig_cm = px.imshow(
                    cm,
                    x=labels,
                    y=labels,
                    labels=dict(x="Predicted Category", y="Actual Category", color="Count"),
                    color_continuous_scale="Blues",
                    text_auto=True
                )
                fig_cm.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_cm, use_container_width=True)

            with c_tab:
                st.markdown("#### Per-Category Precision, Recall & F1")
                st.dataframe(metrics["class_metrics"], use_container_width=True, height=420)
        else:
            st.info("Insufficient ground truth labelled records (< 5) to compute full multi-class classification matrix.")
    else:
        st.info("Loaded dataset does not contain a 'ground_truth_category' column. Upload a labelled evaluation dataset in Data Management to view live evaluation scores.")

    # Escalation Precision
    st.markdown("---")
    st.markdown("### 3. Actionable Escalation Precision")
    st.caption("Evaluates what percentage of escalated (High/Critical) complaints contain concrete, actionable operational specifics.")
    
    esc_metrics = evaluate_escalation_precision(df_active)
    if esc_metrics.get("status") != "empty":
        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.metric("Total Escalated Cases", esc_metrics["total_escalated"])
        col_e2.metric("Actionable Escalated Cases", esc_metrics["actionable_in_escalated"])
        col_e3.metric("Escalation Precision", f"{esc_metrics['actionable_precision_pct']}%")

    # Seeded Incident Detection Lead Time
    st.markdown("---")
    st.markdown("### 4. Seeded Incident Lead-Time Benchmark")
    st.caption("Demonstrating operational lead-time gain comparing weekly batch review vs Early-Warning Radar.")

    sim_res = simulate_seeded_incident_lead_time()
    st.dataframe(sim_res["scenarios"], use_container_width=True)
    st.success(f"⚡ Average Operational Lead Time Advantage: **{sim_res['mean_lead_time_improvement']}**")


# =========================================================
# PAGE 7: DATA MANAGEMENT & INGESTION
# =========================================================
elif selected_page == "💾 Data Management & Ingestion":
    render_page_header(
        "Data Management, Upload & Schema Mapping",
        "Ingest organizer CSV/Excel/JSON files, inspect data quality, and export enriched intelligence datasets"
    )

    st.markdown("### Ingest External Retail Feedback Dataset")
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV, XLSX, or JSON):",
        type=["csv", "xlsx", "xls", "json"],
        help="Upload official organizer files or retail store feedback exports."
    )

    if uploaded_file is not None:
        raw_uploaded_df, load_msg = load_file_into_dataframe(uploaded_file)
        if raw_uploaded_df is not None:
            st.success(load_msg)

            st.markdown("#### Schema & Column Mapping")
            st.caption("The engine automatically detected column matches. Adjust any mapping if required:")

            auto_map = auto_detect_column_mapping(list(raw_uploaded_df.columns))
            cols_list = ["(None / Unmapped)"] + list(raw_uploaded_df.columns)

            c_map1, c_map2 = st.columns(2)
            user_mapping = {}

            with c_map1:
                f_text = auto_map.get("feedback_text")
                idx_text = cols_list.index(f_text) if f_text in cols_list else 0
                user_mapping["feedback_text"] = st.selectbox("Feedback Text Column * (Mandatory)", cols_list, index=idx_text)

                f_store = auto_map.get("store_id")
                idx_store = cols_list.index(f_store) if f_store in cols_list else 0
                user_mapping["store_id"] = st.selectbox("Store ID Column", cols_list, index=idx_store)

                f_time = auto_map.get("timestamp")
                idx_time = cols_list.index(f_time) if f_time in cols_list else 0
                user_mapping["timestamp"] = st.selectbox("Timestamp / Date Column", cols_list, index=idx_time)

            with c_map2:
                f_region = auto_map.get("region")
                idx_region = cols_list.index(f_region) if f_region in cols_list else 0
                user_mapping["region"] = st.selectbox("Region Column", cols_list, index=idx_region)

                f_chan = auto_map.get("channel")
                idx_chan = cols_list.index(f_chan) if f_chan in cols_list else 0
                user_mapping["channel"] = st.selectbox("Feedback Channel Column", cols_list, index=idx_chan)

                f_cat = auto_map.get("category")
                idx_cat = cols_list.index(f_cat) if f_cat in cols_list else 0
                user_mapping["category"] = st.selectbox("Ground Truth Category (Optional)", cols_list, index=idx_cat)

            # Clean mapped values
            cleaned_mapping = {
                k: (v if v != "(None / Unmapped)" else None)
                for k, v in user_mapping.items()
            }

            # Quality Audit Summary
            audit = compute_data_quality_audit(raw_uploaded_df, cleaned_mapping)
            st.markdown("#### Data Quality & Hygiene Audit")
            qa1, qa2, qa3, qa4 = st.columns(4)
            qa1.metric("Total Records", audit["total_rows"])
            qa2.metric("Missing Text Rows", audit["missing_text_count"])
            qa3.metric("Duplicate Rows", audit["exact_duplicates_count"])
            qa4.metric("Invalid Dates", audit["invalid_dates_count"])

            if st.button("🚀 Process & Ingest Uploaded Dataset", type="primary"):
                if not cleaned_mapping.get("feedback_text"):
                    st.error("Please select a valid Feedback Text column before proceeding.")
                else:
                    norm_df = normalize_schema(raw_uploaded_df, cleaned_mapping)
                    with st.spinner("Processing multilingual normalization, hybrid classification, and policy routing..."):
                        processed = run_full_pipeline(norm_df)
                        st.session_state.raw_df = norm_df
                        st.session_state.processed_df = processed
                        st.session_state.is_demo_mode = False
                        st.session_state.dataset_name = uploaded_file.name
                    st.success(f"Successfully processed {len(processed)} records from {uploaded_file.name}!")
                    st.rerun()

    st.markdown("---")
    st.markdown("### Demonstration Mode Reset")
    st.caption("Switch back to the transparent synthetic demonstration dataset at any time.")
    if st.button("Reset to Synthetic Demonstration Data"):
        st.session_state.is_demo_mode = True
        st.session_state.raw_df = generate_synthetic_demo_dataset()
        st.session_state.processed_df = run_full_pipeline(st.session_state.raw_df)
        st.session_state.dataset_name = "SYNTHETIC DEMO DATA — NOT OFFICIAL ORGANIZER DATA"
        st.success("Reset to synthetic demonstration data.")
        st.rerun()

    st.markdown("---")
    st.markdown("### Export Processed Intelligence Dataset")
    st.caption("Download the fully enriched dataset including predictions, severity flags, department routing, and cluster IDs.")

    csv_data = st.session_state.processed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Enriched Intelligence Dataset (CSV)",
        data=csv_data,
        file_name="VoiceOfStore_Enriched_Intelligence.csv",
        mime="text/csv"
    )
