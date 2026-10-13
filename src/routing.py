"""
Severity Assessment, Mandatory Policy Overrides, Actionability, and Department Routing
Talent AP Mega Hackathon — Problem Statement R5: Voice-of-Store Engine
"""

import re
from typing import Dict, Any, List
import pandas as pd

from src.config import (
    CATEGORY_DEPARTMENT_MAP,
    MANDATORY_POLICY_RULES,
    ACTIONABLE_PATTERNS
)

# Standard urgency keywords that elevate normal severity to Medium or High
URGENCY_KEYWORDS = [
    "urgent", "immediately", "unacceptable", "terrible", "worst", "disaster",
    "completely broken", "waste of money", "filthy", "disgusted", "lawsuit", "police"
]


def evaluate_mandatory_policies(text: str, category: str) -> Dict[str, Any]:
    """
    Evaluates mandatory health, safety, allergen, and misconduct policies.
    MANDATORY POLICY RULES OPERATE INDEPENDENTLY OF ML MODELS AND OVERRIDE LOWER SEVERITIES.
    """
    text_lower = text.lower()

    for rule in MANDATORY_POLICY_RULES:
        for kw in rule["keywords"]:
            if kw in text_lower:
                return {
                    "is_override": True,
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "severity": rule["mandatory_severity"],
                    "department": rule["department"],
                    "escalation_reason": f"MANDATORY POLICY OVERRIDE: {rule['name']} (matched '{kw}')",
                    "recommended_action": rule["recommended_action"],
                    "allegation_status": "Unverified Allegation — Escalated for Urgent Human Verification"
                }

    return {
        "is_override": False,
        "rule_id": None,
        "rule_name": None,
        "severity": None,
        "department": None,
        "escalation_reason": "Standard Category Policy",
        "recommended_action": "Standard operational follow-up through routine shift management.",
        "allegation_status": "Standard Customer / Staff Feedback"
    }


def assess_actionability(text: str) -> Dict[str, Any]:
    """
    Determines whether feedback is actionable by checking for concrete details:
    bill numbers, aisle numbers, staff names, specific batch or counter details.
    """
    text_lower = text.lower()
    matched_patterns = []

    for pat in ACTIONABLE_PATTERNS:
        match = re.search(pat, text_lower)
        if match:
            matched_patterns.append(match.group(0))

    word_count = len(text.split())

    if len(matched_patterns) >= 1 or (word_count >= 12 and any(w in text_lower for w in ["counter", "aisle", "rack", "cashier", "manager", "fridge", "scanner", "bill"])):
        level = "Highly Actionable"
        score = 0.90
        explanation = f"Contains concrete operational specifics ({', '.join(matched_patterns) if matched_patterns else 'descriptive store context'})."
    elif word_count >= 5:
        level = "Actionable"
        score = 0.65
        explanation = "Identifies a distinct operational issue with actionable context."
    elif word_count >= 2:
        level = "Informational"
        score = 0.40
        explanation = "General observation or brief sentiment without specific incident details."
    else:
        level = "Vague / Non-Actionable"
        score = 0.15
        explanation = "Minimal text insufficient for root-cause diagnosis."

    return {
        "actionability_level": level,
        "actionability_score": score,
        "explanation": explanation,
        "actionable_anchors": matched_patterns
    }


def route_and_assess_record(
    text: str,
    predicted_category: str,
    custom_dept_map: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Complete routing pipeline for a single record:
    1. Mandatory policy evaluation (Overrides ML severity)
    2. Fallback severity evaluation
    3. Department assignment
    4. Actionability diagnosis
    """
    dept_map = custom_dept_map or CATEGORY_DEPARTMENT_MAP
    default_dept = dept_map.get(predicted_category, "Store Operations")

    # 1. Mandatory policy evaluation
    policy_res = evaluate_mandatory_policies(text, predicted_category)

    # 2. Actionability evaluation
    act_res = assess_actionability(text)

    if policy_res["is_override"]:
        final_severity = policy_res["severity"]
        final_dept = policy_res["department"]
        escalation_reason = policy_res["escalation_reason"]
        rec_action = policy_res["recommended_action"]
        allegation_status = policy_res["allegation_status"]
    else:
        # Determine standard severity
        text_lower = text.lower()
        has_urgency = any(w in text_lower for w in URGENCY_KEYWORDS)

        if predicted_category in ["Food Safety & Hygiene", "Customer Service & Staff Conduct"]:
            final_severity = "High" if has_urgency else "Medium"
        elif predicted_category in ["Billing & Checkout Operations", "Store Infrastructure & Facilities"]:
            final_severity = "High" if has_urgency else "Medium"
        elif predicted_category in ["Product Availability & OOS", "Pricing & Promotions Discrepancy", "Delivery & Packaging"]:
            final_severity = "Medium" if has_urgency else "Low"
        else:
            final_severity = "Low"

        final_dept = default_dept
        escalation_reason = "Standard Department Routing"
        rec_action = f"Route to {final_dept} for scheduled operational review."
        allegation_status = "Standard Operational Feedback"

    return {
        "severity": final_severity,
        "department": final_dept,
        "is_mandatory_override": policy_res["is_override"],
        "escalation_reason": escalation_reason,
        "recommended_action": rec_action,
        "allegation_status": allegation_status,
        "actionability": act_res["actionability_level"],
        "actionability_score": act_res["actionability_score"],
        "actionability_detail": act_res["explanation"]
    }


def apply_routing_batch(df: pd.DataFrame, custom_dept_map: Dict[str, str] = None) -> pd.DataFrame:
    """
    Applies routing and severity assessments to an entire DataFrame.
    """
    severities = []
    departments = []
    overrides = []
    reasons = []
    actions = []
    allegation_statuses = []
    actionabilities = []
    actionability_scores = []

    for _, row in df.iterrows():
        text = str(row.get("feedback_text", row.get("normalized_text", "")))
        cat = str(row.get("predicted_category", row.get("category", "General / Uncategorized")))
        res = route_and_assess_record(text, cat, custom_dept_map)

        severities.append(res["severity"])
        departments.append(res["department"])
        overrides.append(res["is_mandatory_override"])
        reasons.append(res["escalation_reason"])
        actions.append(res["recommended_action"])
        allegation_statuses.append(res["allegation_status"])
        actionabilities.append(res["actionability"])
        actionability_scores.append(res["actionability_score"])

    df_out = df.copy()
    df_out["severity"] = severities
    df_out["department"] = departments
    df_out["is_mandatory_override"] = overrides
    df_out["escalation_reason"] = reasons
    df_out["recommended_action"] = actions
    df_out["allegation_status"] = allegation_statuses
    df_out["actionability"] = actionabilities
    df_out["actionability_score"] = actionability_scores
    return df_out
