"""
Automated Pipeline and Feature Unit Tests
Talent AP Mega Hackathon — Problem Statement R5: Voice-of-Store Engine
"""

import unittest
import pandas as pd

from src.preprocessing import normalize_feedback_text, detect_language_characteristics
from src.classification import HybridIssueClassifier
from src.routing import evaluate_mandatory_policies, route_and_assess_record, apply_routing_batch
from src.emerging_detection import EarlyWarningRadar
from src.deduplication import CrossChannelDeduplicator
from src.reporting import generate_weekly_regional_brief, generate_pdf_report
from src.evaluation import compute_model_metrics, evaluate_escalation_precision, simulate_seeded_incident_lead_time
from src.data_loader import generate_synthetic_demo_dataset, compute_data_quality_audit, auto_detect_column_mapping


class TestVoiceOfStoreEngine(unittest.TestCase):

    def setUp(self):
        self.demo_df = generate_synthetic_demo_dataset()

    def test_text_preprocessing(self):
        # Hinglish normalization test
        text_hinglish = "Store mein AC kaam nahi kar raha hai 😡"
        norm, meta = normalize_feedback_text(text_hinglish)
        self.assertIn("not working", norm.lower())
        self.assertTrue(meta["has_emojis"])
        self.assertIn("Hinglish", meta["primary_language"])

        # Tamil transliterated test
        text_tamil = "Billing counter la romba mosam and azhukku"
        norm_ta, meta_ta = normalize_feedback_text(text_tamil)
        self.assertIn("bad", norm_ta.lower())

    def test_mandatory_severity_policy(self):
        # Allergen emergency
        res = evaluate_mandatory_policies("Severe allergic reaction to unlabelled peanuts", "Food Safety & Hygiene")
        self.assertTrue(res["is_override"])
        self.assertEqual(res["severity"], "Critical")
        self.assertEqual(res["department"], "Store Operations")

        # Pest emergency
        res_pest = evaluate_mandatory_policies("Saw a cockroach crawling on bakery counter", "Food Safety & Hygiene")
        self.assertTrue(res_pest["is_override"])
        self.assertEqual(res_pest["severity"], "High")

        # Staff harassment emergency
        res_hr = evaluate_mandatory_policies("Cashier shouted abusive words and physically threatened customer", "Customer Service & Staff Conduct")
        self.assertTrue(res_hr["is_override"])
        self.assertEqual(res_hr["severity"], "Critical")
        self.assertEqual(res_hr["department"], "HR")

    def test_classifier_and_routing(self):
        classifier = HybridIssueClassifier()
        pred = classifier.predict_single("Milk was expired and curdled with bad odor")
        self.assertEqual(pred["category"], "Food Safety & Hygiene")
        self.assertGreater(pred["confidence"], 0.5)

        route_res = route_and_assess_record("Milk was expired and curdled", pred["category"])
        self.assertEqual(route_res["department"], "Store Operations")
        self.assertIn(route_res["severity"], ["Critical", "High"])

    def test_early_warning_radar(self):
        radar = EarlyWarningRadar(min_current_count=2, min_growth_pct=30.0, spike_z_threshold=1.0)
        
        # Enrich demo dataset with normalized text and predictions
        classifier = HybridIssueClassifier()
        df_pred = classifier.predict_batch(self.demo_df, text_col="feedback_text")
        df_routed = apply_routing_batch(df_pred)

        results = radar.analyze(df_routed)
        self.assertGreater(len(results["alerts"]), 0)

        # Store 104 should be detected with dairy/food safety issue
        detected_stores = [a["store_id"] for a in results["alerts"]]
        self.assertIn("STORE_104", detected_stores)

    def test_cross_channel_deduplication(self):
        dedup = CrossChannelDeduplicator(time_window_hours=48, similarity_threshold=0.35)
        
        classifier = HybridIssueClassifier()
        df_pred = classifier.predict_batch(self.demo_df, text_col="feedback_text")
        
        df_dedup, clusters = dedup.process(df_pred, text_col="feedback_text")
        self.assertIn("incident_id", df_dedup.columns)
        self.assertGreater(len(clusters), 0)

        # Bill 98421 incident at STORE_201 should be clustered across channels
        cluster_stores = [c["store_id"] for c in clusters]
        self.assertIn("STORE_201", cluster_stores)

    def test_weekly_brief_and_pdf(self):
        classifier = HybridIssueClassifier()
        df_pred = classifier.predict_batch(self.demo_df, text_col="feedback_text")
        df_routed = apply_routing_batch(df_pred)
        dedup = CrossChannelDeduplicator()
        df_full, _ = dedup.process(df_routed, text_col="feedback_text")

        brief = generate_weekly_regional_brief(df_full, region="South", selected_period_label="Latest Week")
        self.assertEqual(brief["status"], "success")
        self.assertGreater(brief["metrics"]["total_records"], 0)

        # Test PDF generator
        pdf_bytes = generate_pdf_report(brief)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)

    def test_evaluation_metrics(self):
        y_true = ["Food Safety & Hygiene", "Billing & Checkout Operations", "Customer Service & Staff Conduct"]
        y_pred = ["Food Safety & Hygiene", "Billing & Checkout Operations", "Customer Service & Staff Conduct"]
        metrics = compute_model_metrics(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

        lead_time = simulate_seeded_incident_lead_time()
        self.assertEqual(lead_time["status"], "success")


if __name__ == "__main__":
    unittest.main()
