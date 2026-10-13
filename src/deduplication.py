"""
Cross-Channel Incident Deduplication & Clustering Engine
Identifies Exact and Near-Duplicate Complaints across disparate channels (Reviews, WhatsApp, Call-Center)
Talent AP Mega Hackathon — Problem Statement R5: Voice-of-Store Engine
"""

from typing import List, Dict, Any, Tuple
import hashlib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CrossChannelDeduplicator:
    """
    Identifies duplicate and correlated customer/staff incident reports across channels.
    Preserves all original records and sources while consolidating incident counts.
    """

    def __init__(self, time_window_hours: int = 48, similarity_threshold: float = 0.35):
        self.time_window_hours = time_window_hours
        self.similarity_threshold = similarity_threshold
        # Cluster review status storage: cluster_id -> dict
        self.cluster_reviews: Dict[str, Dict[str, Any]] = {}

    def process(
        self,
        df: pd.DataFrame,
        store_col: str = "store_id",
        text_col: str = "normalized_text",
        time_col: str = "timestamp",
        channel_col: str = "channel",
        id_col: str = "feedback_id"
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Runs deduplication and returns an enriched DataFrame with cluster IDs
        plus a list of multi-record incident clusters.
        """
        if df.empty:
            return df, []

        df_work = df.copy()
        if id_col not in df_work.columns:
            df_work[id_col] = [f"FB_{i:05d}" for i in range(len(df_work))]

        df_work["incident_id"] = ""
        df_work["is_cross_channel_duplicate"] = False
        df_work["cluster_size"] = 1

        # 1. Exact Duplicate Hash
        exact_hashes = {}
        for idx, row in df_work.iterrows():
            text_str = str(row.get(text_col, "")).strip().lower()
            store_str = str(row.get(store_col, "")).strip()
            hash_key = hashlib.md5(f"{store_str}_{text_str}".encode('utf-8')).hexdigest()[:8]
            if hash_key not in exact_hashes:
                exact_hashes[hash_key] = []
            exact_hashes[hash_key].append(idx)

        # Assign exact duplicate clusters
        cluster_counter = 1
        for hash_key, idxs in exact_hashes.items():
            if len(idxs) > 1:
                cid = f"INC-EXACT-{cluster_counter:04d}"
                cluster_counter += 1
                for i in idxs:
                    df_work.loc[i, "incident_id"] = cid
                    df_work.loc[i, "is_cross_channel_duplicate"] = True
                    df_work.loc[i, "cluster_size"] = len(idxs)

        # 2. Near-Duplicate Match within same Store and Time Proximity
        if time_col in df_work.columns:
            df_work["_dt"] = pd.to_datetime(df_work[time_col], errors='coerce')
        else:
            df_work["_dt"] = pd.Timestamp.now()

        # Helper to extract specific anchors (bill numbers, counter numbers, amount)
        import re
        def extract_anchors(s: str) -> set:
            return set(re.findall(r'\b(?:\d{4,}|counter\s*\d+|bill\s*#?\s*\d+)\b', s.lower()))

        # Group by store for localized near-duplicate search
        stores = df_work[store_col].dropna().unique()
        for store in stores:
            store_indices = df_work[df_work[store_col] == store].index.tolist()
            if len(store_indices) < 2:
                continue

            # Check records in store
            candidate_indices = [
                i for i in store_indices
                if not str(df_work.loc[i, "incident_id"]).startswith("INC-EXACT")
            ]

            if len(candidate_indices) < 2:
                continue

            texts = [str(df_work.loc[i, text_col]) for i in candidate_indices]
            anchors_list = [extract_anchors(t) for t in texts]

            # Build localized TF-IDF matrix
            try:
                tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
                matrix = tfidf.fit_transform(texts)
                sim_matrix = cosine_similarity(matrix)

                for a_idx, i in enumerate(candidate_indices):
                    for b_idx in range(a_idx + 1, len(candidate_indices)):
                        j = candidate_indices[b_idx]
                        score = float(sim_matrix[a_idx, b_idx])

                        # Anchor boost (e.g. sharing bill #98421 or counter 3)
                        shared_anchors = anchors_list[a_idx] & anchors_list[b_idx]
                        if shared_anchors:
                            score += 0.35

                        # Check time difference
                        dt_i = df_work.loc[i, "_dt"]
                        dt_j = df_work.loc[j, "_dt"]
                        time_diff_hrs = abs((dt_i - dt_j).total_seconds()) / 3600.0 if (pd.notnull(dt_i) and pd.notnull(dt_j)) else 0.0

                        if score >= self.similarity_threshold and time_diff_hrs <= self.time_window_hours:
                            # We have a confirmed cross-channel near duplicate match!
                            inc_i = df_work.loc[i, "incident_id"]
                            inc_j = df_work.loc[j, "incident_id"]

                            if inc_i and not inc_j:
                                df_work.loc[j, "incident_id"] = inc_i
                                df_work.loc[j, "is_cross_channel_duplicate"] = True
                            elif inc_j and not inc_i:
                                df_work.loc[i, "incident_id"] = inc_j
                                df_work.loc[i, "is_cross_channel_duplicate"] = True
                            elif not inc_i and not inc_j:
                                cid = f"INC-NEAR-{cluster_counter:04d}"
                                cluster_counter += 1
                                df_work.loc[i, "incident_id"] = cid
                                df_work.loc[j, "incident_id"] = cid
                                df_work.loc[i, "is_cross_channel_duplicate"] = True
                                df_work.loc[j, "is_cross_channel_duplicate"] = True
            except Exception:
                continue

        # Fill remaining unique records with their own incident ID
        for idx in df_work.index:
            if not df_work.loc[idx, "incident_id"]:
                fb_id = df_work.loc[idx, id_col]
                df_work.loc[idx, "incident_id"] = f"INC-SNGL-{fb_id}"

        # Clean temp column
        if "_dt" in df_work.columns:
            df_work = df_work.drop(columns=["_dt"])

        # Update cluster size column
        cluster_sizes = df_work["incident_id"].value_counts().to_dict()
        df_work["cluster_size"] = df_work["incident_id"].map(cluster_sizes)

        # Build reviewable incident cluster cards
        clusters_summary = []
        multi_clusters = df_work[df_work["cluster_size"] > 1]["incident_id"].unique()

        for cid in multi_clusters:
            c_records = df_work[df_work["incident_id"] == cid]
            channels = c_records[channel_col].dropna().unique().tolist() if channel_col in c_records.columns else []
            store = c_records[store_col].iloc[0] if store_col in c_records.columns else "Unknown"
            cat = c_records["predicted_category"].iloc[0] if "predicted_category" in c_records.columns else "Uncategorized"
            texts = c_records["feedback_text"].tolist() if "feedback_text" in c_records.columns else []

            # Check review status
            review_state = self.cluster_reviews.get(cid, {
                "status": "Candidate Match (Pending Review)",
                "note": "Automatically detected cross-channel correlation"
            })

            clusters_summary.append({
                "cluster_id": cid,
                "store_id": store,
                "category": cat,
                "record_count": len(c_records),
                "channels_involved": channels,
                "sample_feedback": texts[:3],
                "review_status": review_state["status"],
                "reviewer_note": review_state["note"],
                "record_ids": c_records[id_col].tolist()
            })

        return df_work, clusters_summary

    def update_cluster_status(self, cluster_id: str, new_status: str, note: str = ""):
        """
        Updates the reviewer confirmation status of an incident cluster.
        """
        self.cluster_reviews[cluster_id] = {
            "status": new_status,
            "note": note or f"Reviewer changed status to {new_status}"
        }
