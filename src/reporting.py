"""
Weekly Regional Intelligence Brief & Multi-Format Reporting (PDF / CSV / Structured View)
Talent AP Mega Hackathon — Problem Statement R5: Voice-of-Store Engine
"""

import io
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from src.config import COLORS


def generate_weekly_regional_brief(
    df: pd.DataFrame,
    region: str,
    selected_period_label: str,
    date_col: str = "timestamp"
) -> Dict[str, Any]:
    """
    Computes genuine quantitative intelligence for the selected region and period.
    Never invents quotes, metrics, or affected store lists.
    """
    if df.empty:
        return {"status": "empty", "message": "No data available."}

    df_work = df.copy()
    if date_col in df_work.columns:
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors='coerce')

    # Filter by region if specified
    if region and region != "All Regions" and "region" in df_work.columns:
        df_reg = df_work[df_work["region"] == region]
    else:
        df_reg = df_work

    if df_reg.empty:
        return {"status": "empty", "message": f"No records found for region: {region}"}

    # Time window calculation
    max_date = df_reg[date_col].max() if date_col in df_reg.columns and not df_reg[date_col].dropna().empty else pd.Timestamp.now()
    curr_start = max_date - pd.Timedelta(days=7)
    prior_start = curr_start - pd.Timedelta(days=7)

    curr_df = df_reg[df_reg[date_col] > curr_start] if date_col in df_reg.columns else df_reg
    prior_df = df_reg[(df_reg[date_col] <= curr_start) & (df_reg[date_col] > prior_start)] if date_col in df_reg.columns else pd.DataFrame()

    total_records = len(curr_df)
    affected_stores = curr_df["store_id"].nunique() if "store_id" in curr_df.columns else 0
    unique_incidents = curr_df["incident_id"].nunique() if "incident_id" in curr_df.columns else total_records
    critical_cases = len(curr_df[curr_df["severity"].isin(["Critical", "High"])]) if "severity" in curr_df.columns else 0
    actionable_count = len(curr_df[curr_df["actionability"].isin(["Highly Actionable", "Actionable"])]) if "actionability" in curr_df.columns else 0
    actionable_pct = round((actionable_count / total_records * 100), 1) if total_records > 0 else 0.0

    # Top 5 Operational Issues
    top_issues = []
    category_counts = curr_df["predicted_category"].value_counts().head(5) if "predicted_category" in curr_df.columns else pd.Series()

    for cat, count in category_counts.items():
        prior_count = len(prior_df[prior_df["predicted_category"] == cat]) if not prior_df.empty and "predicted_category" in prior_df.columns else 0
        diff = count - prior_count

        if prior_count == 0:
            trend_str = "New / Unprecedented"
            trend_dir = "⬆️"
        elif diff > 0:
            trend_str = f"+{(diff / prior_count)*100:.0f}%"
            trend_dir = "⬆️"
        elif diff < 0:
            trend_str = f"{(diff / prior_count)*100:.0f}%"
            trend_dir = "⬇️"
        else:
            trend_str = "Stable (0%)"
            trend_dir = "➡️"

        cat_slice = curr_df[curr_df["predicted_category"] == cat]
        dept = cat_slice["department"].iloc[0] if "department" in cat_slice.columns else "Store Operations"
        stores_impacted = cat_slice["store_id"].nunique() if "store_id" in cat_slice.columns else 1
        
        # Genuine evidence quotes from the data
        quotes = cat_slice["feedback_text"].dropna().head(2).tolist()

        top_issues.append({
            "category": cat,
            "volume": count,
            "prior_volume": prior_count,
            "trend_dir": trend_dir,
            "trend_display": trend_str,
            "department": dept,
            "affected_stores": stores_impacted,
            "evidence_quotes": quotes
        })

    # Caveats & Data Quality Notes
    review_flagged = len(curr_df[curr_df["needs_human_review"] == True]) if "needs_human_review" in curr_df.columns else 0
    code_mixed_count = len(curr_df[curr_df["is_code_mixed"] == True]) if "is_code_mixed" in curr_df.columns else 0
    caveats = []
    if review_flagged > 0:
        caveats.append(f"{review_flagged} feedback records in this period require human review due to colloquial slang, brevity, or low model confidence.")
    if code_mixed_count > 0:
        caveats.append(f"{code_mixed_count} records were code-mixed (Hinglish/Tamil/Marathi) and processed via vernacular normalization heuristics.")
    if "transactions" not in df.columns:
        caveats.append("Exposure normalization is currently inactive; trend rates represent within-store historical comparisons.")

    return {
        "status": "success",
        "region": region or "All Regions",
        "period_label": f"{curr_start.strftime('%d %b %Y')} – {max_date.strftime('%d %b %Y')}",
        "metrics": {
            "total_records": total_records,
            "unique_incidents": unique_incidents,
            "affected_stores": affected_stores,
            "critical_high_cases": critical_cases,
            "actionable_pct": actionable_pct
        },
        "top_issues": top_issues,
        "caveats": caveats
    }


def generate_pdf_report(brief_data: Dict[str, Any]) -> bytes:
    """
    Renders an executive 1-page PDF intelligence brief using ReportLab.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor(COLORS['navy_dark'])
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor(COLORS['text_muted'])
    )
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor(COLORS['navy_medium']),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor(COLORS['text_dark'])
    )
    quote_style = ParagraphStyle(
        'QuoteStyle',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor(COLORS['navy_medium'])
    )

    story = []

    # Title & Header
    story.append(Paragraph("VOICE-OF-STORE ENGINE", title_style))
    story.append(Paragraph(
        f"Weekly Regional Intelligence Brief &bull; Region: {brief_data.get('region')} &bull; Period: {brief_data.get('period_label')}",
        subtitle_style
    ))
    story.append(Spacer(1, 10))

    # Executive KPI Summary Table
    m = brief_data.get("metrics", {})
    kpi_data = [
        ["Total Feedback", "Unique Incidents", "Stores Affected", "High/Critical Risks", "Actionability Rate"],
        [
            str(m.get("total_records", 0)),
            str(m.get("unique_incidents", 0)),
            str(m.get("affected_stores", 0)),
            str(m.get("critical_high_cases", 0)),
            f"{m.get('actionable_pct', 0)}%"
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[108, 108, 108, 108, 108])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(COLORS['navy_medium'])),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 13),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor(COLORS['teal_primary'])),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(COLORS['navy_medium'])),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # Top Operational Issues Table
    story.append(Paragraph("TOP OPERATIONAL ISSUES (WEEK-ON-WEEK ANALYSIS)", section_style))

    table_rows = [["Rank & Category", "Owner Dept", "Curr Vol", "WoW Trend", "Stores", "Evidence Quote Sample"]]
    for idx, item in enumerate(brief_data.get("top_issues", []), 1):
        quote_sample = item["evidence_quotes"][0] if item["evidence_quotes"] else "No verbatim quote"
        if len(quote_sample) > 75:
            quote_sample = quote_sample[:75] + "..."
        table_rows.append([
            f"#{idx} {item['category']}",
            item["department"],
            str(item["volume"]),
            f"{item['trend_dir']} {item['trend_display']}",
            str(item["affected_stores"]),
            Paragraph(f'"{quote_sample}"', quote_style)
        ])

    issues_table = Table(table_rows, colWidths=[140, 85, 55, 75, 45, 140])
    issues_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('ALIGN', (2, 0), (4, -1), 'CENTER'),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94A3B8')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(issues_table)
    story.append(Spacer(1, 14))

    # Caveats & Audit Disclosures
    story.append(Paragraph("DATA AUDIT, CAVEATS & COMPLIANCE NOTES", section_style))
    for cav in brief_data.get("caveats", []):
        story.append(Paragraph(f"&bull; {cav}", body_style))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Confidential Internal Operational Intelligence &bull; Generated by Voice-of-Store Engine (Talent AP Mega Hackathon R5)",
        subtitle_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
