"""
Issue Taxonomy Classification Engine
Supports TF-IDF + Scikit-Learn Classifier with Rule-Based Baseline Fallback
Talent AP Mega Hackathon — Problem Statement R5: Voice-of-Store Engine
"""

import os
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

from src.config import TAXONOMY_CATEGORIES, MANDATORY_POLICY_RULES

# High-precision domain keywords for Rule-Based Baseline
RULE_KEYWORD_PATTERNS: Dict[str, List[str]] = {
    "Food Safety & Hygiene": [
        "expired", "expiry", "fungus", "mold", "mould", "rotten", "stale", "spoiled",
        "bad smell", "stink", "foul odor", "unhygienic", "dirty floor", "restroom dirty",
        "cockroach", "rat", "rodent", "pest", "flies", "worms", "hair in", "food poison",
        "vomit", "sick after", "curdled milk", "sour milk", "stomach pain", "allergy"
    ],
    "Customer Service & Staff Conduct": [
        "rude", "misbehavior", "arrogant", "unhelpful", "shouted", "abusive", "cashier attitude",
        "staff ignored", "no manners", "disrespectful", "bribe", "theft", "harass",
        "bad service", "slapped", "counter staff", "supervisor argued", "security check rude"
    ],
    "Billing & Checkout Operations": [
        "overcharge", "overcharged", "extra money", "wrong bill", "mrp mismatch", "scanner wrong",
        "double charge", "upi deducted", "pos not working", "card machine", "change not returned",
        "receipt missing", "discount not applied", "bill counter slow", "billing error", "refund pending"
    ],
    "Product Availability & OOS": [
        "out of stock", "oos", "empty shelves", "not available", "item missing", "no stock",
        "replenish", "dairy out of stock", "bread over", "brand not available", "shortage",
        "searched everywhere but no stock", "restock"
    ],
    "Store Infrastructure & Facilities": [
        "ac not working", "ac off", "suffocating", "lift not working", "escalator broken",
        "trolley broken", "wheel jammed", "parking full", "lights off", "dim light",
        "water leakage", "refrigerator warm", "freezer broken", "cooling bad", "chiller down"
    ],
    "Pricing & Promotions Discrepancy": [
        "offer not given", "discount scam", "bogus offer", "promo code", "tag price different",
        "shelf price higher", "buy 1 get 1 not applied", "misleading discount", "coupon invalid"
    ],
    "App & Digital Ordering": [
        "app crashed", "otp not received", "login issue", "loyalty points missing",
        "membership not detected", "app slow", "online cart error", "digital voucher failed"
    ],
    "Delivery & Packaging": [
        "package torn", "box crushed", "spilled oil", "delivery late", "wrong item delivered",
        "missing packet", "bad packaging", "rider rude", "tampered seal"
    ]
}


class HybridIssueClassifier:
    """
    Hybrid text classification engine for store feedback.
    - If labelled data is provided: trains/uses a TF-IDF + Logistic Regression model.
    - If unlabelled: uses an inspectable, transparent rule-based baseline.
    - Always extracts evidence keywords and provides confidence metrics.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.categories = list(TAXONOMY_CATEGORIES)
        self.model_pipeline: Optional[Pipeline] = None
        self.is_trained: bool = False
        self.training_samples_count: int = 0
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "..", "data", "classifier_model.joblib")
        
        # Try loading saved model if available
        if os.path.exists(self.model_path):
            try:
                loaded = joblib.load(self.model_path)
                self.model_pipeline = loaded.get("pipeline")
                self.categories = loaded.get("categories", self.categories)
                self.is_trained = True
                self.training_samples_count = loaded.get("sample_count", 0)
            except Exception:
                self.model_pipeline = None
                self.is_trained = False

    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Train a supervised classifier on provided labelled feedback.
        Uses TF-IDF unigrams + bigrams with balanced logistic regression.
        """
        if len(texts) < 10 or len(set(labels)) < 2:
            return {"status": "error", "message": "Insufficient data or classes to train model."}

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
                min_df=2
            )),
            ('clf', LogisticRegression(
                C=1.2,
                max_iter=600,
                class_weight='balanced',
                solver='lbfgs'
            ))
        ])

        pipeline.fit(texts, labels)
        self.model_pipeline = pipeline
        self.is_trained = True
        self.training_samples_count = len(texts)
        self.categories = list(pipeline.classes_)

        # Save trained model to disk
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                "pipeline": self.model_pipeline,
                "categories": self.categories,
                "sample_count": self.training_samples_count
            }, self.model_path)
        except Exception:
            pass

        return {
            "status": "success",
            "samples_trained": len(texts),
            "classes": self.categories
        }

    def _rule_based_predict(self, normalized_text: str) -> Tuple[str, float, List[str]]:
        """
        Transparent rule-based scoring baseline based on keyword pattern matches.
        """
        text_lower = normalized_text.lower()
        category_scores: Dict[str, float] = {}
        category_matches: Dict[str, List[str]] = {}

        for category, patterns in RULE_KEYWORD_PATTERNS.items():
            matched_keywords = []
            score = 0.0
            for pattern in patterns:
                if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                    matched_keywords.append(pattern)
                    score += 1.0

            if matched_keywords:
                category_scores[category] = score
                category_matches[category] = matched_keywords

        if category_scores:
            best_cat = max(category_scores, key=category_scores.get)
            match_count = category_scores[best_cat]
            # Heuristic confidence based on matched keyword specificity
            confidence = min(0.95, 0.65 + (match_count * 0.10))
            return best_cat, round(confidence, 2), category_matches[best_cat]

        return "General / Uncategorized", 0.40, []

    def predict_single(self, normalized_text: str, original_text: str = "") -> Dict[str, Any]:
        """
        Predict category for a single text record.
        Combines ML model probabilities (if trained) with rule-based pattern matching.
        """
        if not normalized_text or not normalized_text.strip():
            return {
                "category": "General / Uncategorized",
                "confidence": 0.0,
                "prediction_source": "Empty Record",
                "evidence_keywords": [],
                "alternative_classes": []
            }

        rule_cat, rule_conf, rule_matches = self._rule_based_predict(normalized_text)

        # If supervised model is trained, get probabilistic prediction
        if self.is_trained and self.model_pipeline is not None:
            try:
                probs = self.model_pipeline.predict_proba([normalized_text])[0]
                classes = self.model_pipeline.classes_
                top_idx = int(np.argmax(probs))
                ml_cat = classes[top_idx]
                ml_conf = float(probs[top_idx])

                # Get sorted alternatives
                sorted_indices = np.argsort(probs)[::-1]
                alternatives = [
                    {"category": classes[idx], "prob": round(float(probs[idx]), 3)}
                    for idx in sorted_indices[1:4]
                ]

                # High-confidence rule match can refine ML tie-breakers
                if rule_cat in ["Food Safety & Hygiene", "Customer Service & Staff Conduct"] and len(rule_matches) >= 2:
                    final_cat = rule_cat
                    final_conf = max(ml_conf, rule_conf)
                    source = f"Hybrid (Rule override: {', '.join(rule_matches[:2])})"
                elif ml_conf >= 0.50:
                    final_cat = ml_cat
                    final_conf = ml_conf
                    source = "Supervised ML (TF-IDF + Logistic Regression)"
                elif rule_conf > ml_conf:
                    final_cat = rule_cat
                    final_conf = rule_conf
                    source = "Rule-Based Baseline"
                else:
                    final_cat = ml_cat
                    final_conf = ml_conf
                    source = "Supervised ML (Low Confidence)"

                # Extract supporting evidence tokens
                evidence = list(set(rule_matches))
                if not evidence:
                    # Extract salient words from text
                    tokens = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', normalized_text.lower())
                              if w not in ["this", "that", "there", "store", "very", "were", "with", "have"]]
                    evidence = tokens[:3]

                return {
                    "category": final_cat,
                    "confidence": round(final_conf, 2),
                    "prediction_source": source,
                    "evidence_keywords": evidence,
                    "alternative_classes": alternatives
                }
            except Exception:
                pass

        # Fallback to pure rule-based baseline
        return {
            "category": rule_cat,
            "confidence": rule_conf,
            "prediction_source": "Rule-Based Baseline",
            "evidence_keywords": rule_matches,
            "alternative_classes": []
        }

    def predict_batch(self, df: pd.DataFrame, text_col: str = "normalized_text") -> pd.DataFrame:
        """
        Batch prediction over a DataFrame, adding predicted category, confidence, and source.
        """
        categories = []
        confidences = []
        sources = []
        evidence_list = []

        for _, row in df.iterrows():
            text = str(row.get(text_col, ""))
            orig = str(row.get("feedback_text", text))
            res = self.predict_single(text, orig)
            categories.append(res["category"])
            confidences.append(res["confidence"])
            sources.append(res["prediction_source"])
            evidence_list.append(", ".join(res["evidence_keywords"]) if res["evidence_keywords"] else "None")

        df_out = df.copy()
        df_out["predicted_category"] = categories
        df_out["confidence"] = confidences
        df_out["prediction_source"] = sources
        df_out["evidence_keywords"] = evidence_list
        return df_out
