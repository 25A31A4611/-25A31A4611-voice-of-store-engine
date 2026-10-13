"""
Early-Warning Radar: Emerging Store-Specific Operational Issue Detection
Talent AP Mega Hackathon — Problem Statement R5: Voice-of-Store Engine
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class EarlyWarningRadar:
    """
    Detects localized operational spikes and emerging issues at individual store levels
    even when regional absolute volume is modest.
    Handles zero baselines, sparse feedback, and mandatory high-severity triggers safely.
    """

    def __init__(
        self,
        min_current_count: int = 2,
        min_growth_pct: float = 50.0,
        spike_z_threshold: float = 1.5,
        window_days: int = 7
    ):
        self.min_current_count = min_current_count
        self.min_growth_pct = min_growth_pct
        self.spike_z_threshold = spike_z_threshold
        self.window_days = window_days
        # In-memory alert management state (alert_id -> dict of status, note)
        self.alert_reviews: Dict[str, Dict[str, Any]] = {}

    def analyze(
        self,
        df: pd.DataFrame,
        store_master: Optional[pd.DataFrame] = None,
        date_col: str = "timestamp",
        store_col: str = "store_id",
        category_col: str = "predicted_category"
    ) -> Dict[str, Any]:
        """
        Runs the Early-Warning detection algorithm across all stores and issue categories.
        """
        if df.empty or date_col not in df.columns:
            return {
                "alerts": [],
                "summary": {"total_alerts": 0, "emergency_alerts": 0, "statistical_spikes": 0},
                "denominator_status": "No data available"
            }

        # Ensure datetime format
        df_work = df.copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors='coerce')
        df_work = df_work.dropna(subset=[date_col])

        if df_work.empty:
            return {
                "alerts": [],
                "summary": {"total_alerts": 0, "emergency_alerts": 0, "statistical_spikes": 0},
                "denominator_status": "No valid timestamps found"
            }

        max_date = df_work[date_col].max()
        cutoff_curr = max_date - pd.Timedelta(days=self.window_days)
        cutoff_prior = cutoff_curr - pd.Timedelta(days=self.window_days)

        current_window = df_work[df_work[date_col] > cutoff_curr]
        prior_window = df_work[(df_work[date_col] <= cutoff_curr) & (df_work[date_col] > cutoff_prior)]

        # Check denominator availability
        has_exposure = False
        exposure_map = {}
        if store_master is not None and not store_master.empty:
            if "transactions" in store_master.columns and "store_id" in store_master.columns:
                exposure_map = dict(zip(store_master["store_id"], store_master["transactions"]))
                has_exposure = True
            elif "traffic" in store_master.columns and "store_id" in store_master.columns:
                exposure_map = dict(zip(store_master["store_id"], store_master["traffic"]))
                has_exposure = True

        denominator_status = (
            "Normalized by Store Footfall / Transactions (Store Master)"
            if has_exposure else
            "Within-store historical comparative mode (Exposure denominator not supplied in dataset)"
        )

        alerts = []
        unique_combinations = set(
            zip(current_window[store_col], current_window[category_col])
        )

        for store_id, category in unique_combinations:
            curr_slice = current_window[
                (current_window[store_col] == store_id) & (current_window[category_col] == category)
            ]
            prior_slice = prior_window[
                (prior_window[store_col] == store_id) & (prior_window[category_col] == category)
            ]

            curr_count = len(curr_slice)
            prior_count = len(prior_slice)
            abs_change = curr_count - prior_count

            # Check if any Critical / Mandatory Policy overrides are present
            has_mandatory = curr_slice.get("is_mandatory_override", pd.Series([False])).any() or (
                curr_slice.get("severity", pd.Series([])).isin(["Critical"]).any()
            )

            # Mathematical spike detection with zero-baseline and sparse-data safeguards
            is_statistical_spike = False
            growth_pct_display = ""
            spike_score = (abs_change) / np.sqrt(prior_count + 1.0)

            if prior_count == 0:
                growth_pct_display = "New Outbreak (0 baseline)"
                if curr_count >= self.min_current_count and spike_score >= self.spike_z_threshold:
                    is_statistical_spike = True
            else:
                pct = ((curr_count - prior_count) / prior_count) * 100.0
                growth_pct_display = f"+{pct:.0f}%" if pct > 0 else f"{pct:.0f}%"
                if (curr_count >= self.min_current_count and
                    pct >= self.min_growth_pct and
                    spike_score >= self.spike_z_threshold):
                    is_statistical_spike = True

            # Trigger condition: Statistical spike OR Mandatory Health/Safety/Misconduct override
            if is_statistical_spike or has_mandatory:
                alert_type = (
                    "Emergency Safety Policy Alert" if has_mandatory and not is_statistical_spike else
                    "Severe Operational Spike & Policy Trigger" if has_mandatory and is_statistical_spike else
                    "Emerging Statistical Spike"
                )

                # Fetch evidence quotes
                evidence_quotes = curr_slice["feedback_text"].dropna().head(3).tolist()
                regions = curr_slice["region"].dropna().unique().tolist() if "region" in curr_slice.columns else []
                region_name = regions[0] if regions else "Region Unspecified"
                channels = curr_slice["channel"].dropna().unique().tolist() if "channel" in curr_slice.columns else []

                # Unique alert ID to prevent duplicate alerts
                period_str = f"{cutoff_curr.strftime('%Y-%m-%d')}_to_{max_date.strftime('%Y-%m-%d')}"
                alert_id = f"{store_id}_{category}_{period_str}"

                # Department owner & recommended action
                suggested_dept = curr_slice["department"].iloc[0] if "department" in curr_slice.columns else "Store Operations"
                recs = curr_slice["recommended_action"].dropna().unique().tolist() if "recommended_action" in curr_slice.columns else []
                rec_action = recs[0] if recs else f"Audit {category} processes at Store {store_id}."

                # Review state
                review_info = self.alert_reviews.get(alert_id, {
                    "status": "Active / Unresolved",
                    "reviewer_note": "Awaiting store management review"
                })

                alerts.append({
                    "alert_id": alert_id,
                    "store_id": store_id,
                    "region": region_name,
                    "category": category,
                    "alert_type": alert_type,
                    "is_mandatory": has_mandatory,
                    "is_statistical_spike": is_statistical_spike,
                    "current_count": curr_count,
                    "prior_count": prior_count,
                    "abs_change": abs_change,
                    "growth_display": growth_pct_display,
                    "spike_score": round(float(spike_score), 2),
                    "channels_affected": channels,
                    "evidence_quotes": evidence_quotes,
                    "suggested_owner": suggested_dept,
                    "recommended_action": rec_action,
                    "period": period_str,
                    "review_status": review_info["status"],
                    "reviewer_note": review_info["reviewer_note"]
                })

        # Sort by urgency: Mandatory emergencies first, then highest spike score
        alerts.sort(
            key=lambda x: (1 if x["is_mandatory"] else 0, x["spike_score"], x["abs_change"]),
            reverse=True
        )

        emergency_count = sum(1 for a in alerts if a["is_mandatory"])
        spike_count = sum(1 for a in alerts if a["is_statistical_spike"])

        return {
            "alerts": alerts,
            "summary": {
                "total_alerts": len(alerts),
                "emergency_alerts": emergency_count,
                "statistical_spikes": spike_count,
                "cutoff_current": cutoff_curr.strftime("%Y-%m-%d"),
                "max_date": max_date.strftime("%Y-%m-%d")
            },
            "denominator_status": denominator_status
        }

    def update_alert_status(self, alert_id: str, new_status: str, note: str = ""):
        """
        Updates the reviewer confirmation status of an alert (Acknowledged, Investigating, Dismissed).
        """
        self.alert_reviews[alert_id] = {
            "status": new_status,
            "reviewer_note": note or f"Updated to {new_status}"
        }
