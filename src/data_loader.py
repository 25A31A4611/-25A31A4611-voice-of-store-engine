"""
Data Ingestion, Schema Normalization, Quality Profiling & Synthetic Demo Generator
Talent AP Mega Hackathon — Problem Statement R5: Voice-of-Store Engine
"""

import os
import sqlite3
import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.config import CHANNELS, TAXONOMY_CATEGORIES

# Candidate column name matches for automatic schema detection
CANONICAL_COLUMN_CANDIDATES = {
    "feedback_id": ["id", "feedback_id", "review_id", "ticket_id", "comment_id", "sr_no", "sno"],
    "feedback_text": ["text", "feedback_text", "feedback", "review", "review_text", "comment", "customer_comment", "transcript", "notes", "description"],
    "store_id": ["store_id", "store", "store_code", "outlet_id", "branch_id", "store_number", "unit_id"],
    "store_name": ["store_name", "branch_name", "outlet_name", "store_loc"],
    "region": ["region", "zone", "state", "territory", "area", "cluster"],
    "channel": ["channel", "source", "feedback_channel", "source_type", "platform", "channel_name"],
    "timestamp": ["timestamp", "date", "created_at", "feedback_date", "review_date", "time", "datetime"],
    "category": ["category", "issue", "issue_category", "label", "topic", "tag", "ground_truth_category"],
    "severity": ["severity", "priority", "criticality", "urgency", "risk_level"],
    "actionability": ["actionable", "actionability", "is_actionable"]
}


def auto_detect_column_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Intelligently inspects input column headers and maps them to canonical schema fields.
    """
    mapping: Dict[str, Optional[str]] = {canon: None for canon in CANONICAL_COLUMN_CANDIDATES}
    cols_lower = {col.lower().strip().replace(" ", "_").replace("-", "_"): col for col in columns}

    for canon, candidates in CANONICAL_COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand in cols_lower:
                mapping[canon] = cols_lower[cand]
                break

    return mapping


def load_file_into_dataframe(uploaded_file_or_path: Any) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads data from CSV, Excel (XLSX/XLS), JSON, or SQLite database.
    Returns (DataFrame, message).
    """
    try:
        # If it's a file path string
        if isinstance(uploaded_file_or_path, str):
            ext = os.path.splitext(uploaded_file_or_path)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(uploaded_file_or_path)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(uploaded_file_or_path)
            elif ext == ".json":
                df = pd.read_json(uploaded_file_or_path)
            elif ext in [".db", ".sqlite", ".sqlite3"]:
                conn = sqlite3.connect(uploaded_file_or_path)
                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
                if not tables.empty:
                    first_table = tables.iloc[0, 0]
                    df = pd.read_sql(f"SELECT * FROM {first_table}", conn)
                else:
                    return None, "SQLite database contains no tables."
                conn.close()
            else:
                return None, f"Unsupported file extension: {ext}"
            return df, f"Successfully loaded {len(df)} records from {os.path.basename(uploaded_file_or_path)}"

        # If it's a Streamlit UploadedFile object
        name = getattr(uploaded_file_or_path, "name", "uploaded_file")
        ext = os.path.splitext(name)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(uploaded_file_or_path)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(uploaded_file_or_path)
        elif ext == ".json":
            df = pd.read_json(uploaded_file_or_path)
        else:
            return None, f"Unsupported file format: {ext}. Please upload CSV, XLSX, or JSON."

        return df, f"Successfully parsed {len(df)} records from {name}."

    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def compute_data_quality_audit(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Performs comprehensive data hygiene audit:
    missing values, duplicate rows, invalid timestamps, empty text records.
    """
    total_rows = len(df)
    total_cols = len(df.columns)

    text_col = mapping.get("feedback_text")
    time_col = mapping.get("timestamp")
    store_col = mapping.get("store_id")

    missing_text_count = df[text_col].isna().sum() if text_col and text_col in df.columns else total_rows
    empty_text_count = (df[text_col].astype(str).str.strip() == "").sum() if text_col and text_col in df.columns else 0

    # Duplicates check
    exact_duplicates_count = df.duplicated().sum()
    text_duplicates_count = df.duplicated(subset=[text_col]).sum() if text_col and text_col in df.columns else 0

    # Timestamp audit
    invalid_dates_count = 0
    if time_col and time_col in df.columns:
        parsed_dates = pd.to_datetime(df[time_col], errors='coerce')
        invalid_dates_count = parsed_dates.isna().sum()

    # Store coverage
    unique_stores = df[store_col].nunique() if store_col and store_col in df.columns else 0

    # Column-level null breakdown
    null_summary = df.isna().sum().to_dict()

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_text_count": int(missing_text_count),
        "empty_text_count": int(empty_text_count),
        "exact_duplicates_count": int(exact_duplicates_count),
        "text_duplicates_count": int(text_duplicates_count),
        "invalid_dates_count": int(invalid_dates_count),
        "unique_stores": int(unique_stores),
        "null_summary": null_summary
    }


def normalize_schema(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Standardizes source columns to canonical field names while keeping all original columns intact.
    """
    df_norm = df.copy()

    # Ensure canonical fields exist
    canonical_fields = ["feedback_id", "feedback_text", "store_id", "region", "channel", "timestamp"]
    for canon in canonical_fields:
        source_col = mapping.get(canon)
        if source_col and source_col in df.columns:
            df_norm[canon] = df[source_col]
        else:
            if canon == "feedback_id":
                df_norm["feedback_id"] = [f"REC_{i+1:05d}" for i in range(len(df_norm))]
            elif canon == "region":
                df_norm["region"] = "Region Unspecified"
            elif canon == "channel":
                df_norm["channel"] = "Online Reviews"
            elif canon == "timestamp":
                df_norm["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            elif canon == "store_id":
                df_norm["store_id"] = "STORE_GENERIC"
            elif canon == "feedback_text":
                df_norm["feedback_text"] = ""

    # Optional fields
    for canon in ["category", "severity", "actionability"]:
        source_col = mapping.get(canon)
        if source_col and source_col in df.columns:
            df_norm[f"ground_truth_{canon}"] = df[source_col]

    return df_norm


def generate_synthetic_demo_dataset() -> pd.DataFrame:
    """
    Generates realistic, transparently labelled synthetic demonstration data.
    LABEL: SYNTHETIC DEMO DATA — NOT OFFICIAL ORGANIZER DATA.
    Includes:
    - Routine operational feedback across 5 regions and 8 stores.
    - High-severity hygiene and food safety emergencies (allergens, rats, broken glass).
    - Localized emerging spike at Store #104 (sudden refrigeration breakdown in latest week).
    - Cross-channel duplicate incident reports across Google Review, WhatsApp, and Call Center.
    - Multilingual code-mixed Hinglish, transliterated Tamil, Marathi, retail shorthand, emojis.
    """
    base_date = pd.Timestamp.now() - pd.Timedelta(days=21)

    records = [
        # --- LOCALIZED EMERGING SPIKE: Store #104 Refrigeration & Dairy Spoil (Week 3 Spike) ---
        {
            "feedback_id": "SYN_001",
            "feedback_text": "Chiller in dairy aisle is leaking warm water. Milk packets are warm and curdled 🤢. Fix immediately!",
            "store_id": "STORE_104",
            "store_name": "Store #104 (Indiranagar)",
            "region": "South",
            "channel": "In-App Feedback",
            "timestamp": (base_date + pd.Timedelta(days=19, hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Food Safety & Hygiene",
            "ground_truth_severity": "Critical",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_002",
            "feedback_text": "Bought Amul curd yesterday from store #104, it was sour and expired. Fridge was clearly off ❄️",
            "store_id": "STORE_104",
            "store_name": "Store #104 (Indiranagar)",
            "region": "South",
            "channel": "Online Reviews",
            "timestamp": (base_date + pd.Timedelta(days=19, hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Food Safety & Hygiene",
            "ground_truth_severity": "High",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_003",
            "feedback_text": "Shift note: Main refrigerator compressor tripped again around 3 PM today. Dairy temperature rising.",
            "store_id": "STORE_104",
            "store_name": "Store #104 (Indiranagar)",
            "region": "South",
            "channel": "Staff WhatsApp / Shift Notes",
            "timestamp": (base_date + pd.Timedelta(days=20, hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Store Infrastructure & Facilities",
            "ground_truth_severity": "High",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_004",
            "feedback_text": "Call center transcript: Customer Mrs. Priya called complaining paneer purchased 2 hours ago smelled bad. Fridge temp issue.",
            "store_id": "STORE_104",
            "store_name": "Store #104 (Indiranagar)",
            "region": "South",
            "channel": "Call-Centre Transcripts",
            "timestamp": (base_date + pd.Timedelta(days=20, hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Food Safety & Hygiene",
            "ground_truth_severity": "High",
            "ground_truth_actionable": "Highly Actionable"
        },

        # --- CROSS-CHANNEL DUPLICATE INCIDENT: Store #201 Cashier Double Billing ---
        {
            "feedback_id": "SYN_005",
            "feedback_text": "Charged twice on UPI for Bill #98421 at counter 3. Cashier refused refund saying server error! 😡",
            "store_id": "STORE_201",
            "store_name": "Store #201 (Connaught Place)",
            "region": "North",
            "channel": "Online Reviews",
            "timestamp": (base_date + pd.Timedelta(days=18, hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Billing & Checkout Operations",
            "ground_truth_severity": "High",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_006",
            "feedback_text": "Customer had double deduction on UPI at counter 3 for Bill #98421. Escalating to digital accounts team.",
            "store_id": "STORE_201",
            "store_name": "Store #201 (Connaught Place)",
            "region": "North",
            "channel": "Staff WhatsApp / Shift Notes",
            "timestamp": (base_date + pd.Timedelta(days=18, hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Billing & Checkout Operations",
            "ground_truth_severity": "High",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_007",
            "feedback_text": "Ticket #CC-4421: Customer called about Rs 850 double swipe on POS at CP store counter 3 for Bill #98421.",
            "store_id": "STORE_201",
            "store_name": "Store #201 (Connaught Place)",
            "region": "North",
            "channel": "Call-Centre Transcripts",
            "timestamp": (base_date + pd.Timedelta(days=18, hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Billing & Checkout Operations",
            "ground_truth_severity": "High",
            "ground_truth_actionable": "Highly Actionable"
        },

        # --- MANDATORY POLICY OVERRIDE: Severe Safety, Pest & Harassment Reports ---
        {
            "feedback_id": "SYN_008",
            "feedback_text": "Disgusting! Saw a big cockroach 🪳 crawling on the unpackaged bakery bread counter at store #305. Children eat from here!",
            "store_id": "STORE_305",
            "store_name": "Store #305 (Bandra West)",
            "region": "West",
            "channel": "Online Reviews",
            "timestamp": (base_date + pd.Timedelta(days=15, hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Food Safety & Hygiene",
            "ground_truth_severity": "Critical",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_009",
            "feedback_text": "Severe allergic reaction after eating snack mislabelled as nut-free. Hospitalized my son with peanut allergy!",
            "store_id": "STORE_102",
            "store_name": "Store #102 (Koramangala)",
            "region": "South",
            "channel": "Call-Centre Transcripts",
            "timestamp": (base_date + pd.Timedelta(days=16, hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Food Safety & Hygiene",
            "ground_truth_severity": "Critical",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_010",
            "feedback_text": "Emergency: Fire exit door in basement stockroom is chained locked. Major fire hazard violation.",
            "store_id": "STORE_401",
            "store_name": "Store #401 (Salt Lake)",
            "region": "East",
            "channel": "Staff WhatsApp / Shift Notes",
            "timestamp": (base_date + pd.Timedelta(days=14, hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Store Infrastructure & Facilities",
            "ground_truth_severity": "Critical",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_011",
            "feedback_text": "Security guard physically grabbed my arm and shouted abusive slurs in front of other shoppers. Unacceptable harassment!",
            "store_id": "STORE_202",
            "store_name": "Store #202 (Noida Sec 18)",
            "region": "North",
            "channel": "In-App Feedback",
            "timestamp": (base_date + pd.Timedelta(days=17, hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Customer Service & Staff Conduct",
            "ground_truth_severity": "Critical",
            "ground_truth_actionable": "Highly Actionable"
        },

        # --- MULTILINGUAL & CODE-MIXED (HINGLISH, TRANSLITERATED TAMIL & MARATHI) ---
        {
            "feedback_id": "SYN_012",
            "feedback_text": "Store mein AC kaam nahi kar raha hai bilkul. Bohot suffocating lag raha tha, bheed bhi zyada thi.",
            "store_id": "STORE_201",
            "store_name": "Store #201 (Connaught Place)",
            "region": "North",
            "channel": "Online Reviews",
            "timestamp": (base_date + pd.Timedelta(days=12, hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Store Infrastructure & Facilities",
            "ground_truth_severity": "Medium",
            "ground_truth_actionable": "Actionable"
        },
        {
            "feedback_id": "SYN_013",
            "feedback_text": "Billing counter la paise jaast ghetle MRP peksha. Counter staff la vicharle tar rudely bhandan zala.",
            "store_id": "STORE_301",
            "store_name": "Store #301 (Pune FC Road)",
            "region": "West",
            "channel": "In-App Feedback",
            "timestamp": (base_date + pd.Timedelta(days=13, hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Billing & Checkout Operations",
            "ground_truth_severity": "High",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_014",
            "feedback_text": "Billing counter romba mosam. Eppo paarthalum kootam. 45 mins wait panna vendi irukku. Staff mariathai illa.",
            "store_id": "STORE_103",
            "store_name": "Store #103 (T Nagar)",
            "region": "South",
            "channel": "Online Reviews",
            "timestamp": (base_date + pd.Timedelta(days=11, hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Billing & Checkout Operations",
            "ground_truth_severity": "Medium",
            "ground_truth_actionable": "Actionable"
        },
        {
            "feedback_id": "SYN_015",
            "feedback_text": "Aloo aur tamatar bilkul sada hua tha vegetable rack par. Smell bohot gandi aa rahi thi.",
            "store_id": "STORE_501",
            "store_name": "Store #501 (Bhopal MP Nagar)",
            "region": "Central",
            "channel": "In-App Feedback",
            "timestamp": (base_date + pd.Timedelta(days=10, hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Food Safety & Hygiene",
            "ground_truth_severity": "High",
            "ground_truth_actionable": "Highly Actionable"
        },

        # --- OUT OF STOCK & MERCHANDISING ROUTINE FEEDBACK ---
        {
            "feedback_id": "SYN_016",
            "feedback_text": "Nandini milk and bread always out of stock by 8 AM. Please replenish morning delivery on time.",
            "store_id": "STORE_101",
            "store_name": "Store #101 (MG Road)",
            "region": "South",
            "channel": "Online Reviews",
            "timestamp": (base_date + pd.Timedelta(days=5, hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Product Availability & OOS",
            "ground_truth_severity": "Medium",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_017",
            "feedback_text": "App promo code SUPER50 not applying at billing counter. Cashier said offline system does not sync.",
            "store_id": "STORE_102",
            "store_name": "Store #102 (Koramangala)",
            "region": "South",
            "channel": "In-App Feedback",
            "timestamp": (base_date + pd.Timedelta(days=7, hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "App & Digital Ordering",
            "ground_truth_severity": "Medium",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_018",
            "feedback_text": "Atta bag delivery package was torn and flour spilled across doorstep. Poor packing carton.",
            "store_id": "STORE_202",
            "store_name": "Store #202 (Noida Sec 18)",
            "region": "North",
            "channel": "Call-Centre Transcripts",
            "timestamp": (base_date + pd.Timedelta(days=8, hours=7)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Delivery & Packaging",
            "ground_truth_severity": "Low",
            "ground_truth_actionable": "Actionable"
        },
        {
            "feedback_id": "SYN_019",
            "feedback_text": "Shelf price says Rs 199 for Kellogg's Cornflakes but scanned at Rs 235 at POS counter 2. Misleading pricing tag.",
            "store_id": "STORE_302",
            "store_name": "Store #302 (Andheri East)",
            "region": "West",
            "channel": "Online Reviews",
            "timestamp": (base_date + pd.Timedelta(days=9, hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Pricing & Promotions Discrepancy",
            "ground_truth_severity": "Medium",
            "ground_truth_actionable": "Highly Actionable"
        },
        {
            "feedback_id": "SYN_020",
            "feedback_text": "Nice store clean aisles ok.",
            "store_id": "STORE_101",
            "store_name": "Store #101 (MG Road)",
            "region": "South",
            "channel": "Online Reviews",
            "timestamp": (base_date + pd.Timedelta(days=4, hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "General / Uncategorized",
            "ground_truth_severity": "Low",
            "ground_truth_actionable": "Vague / Non-Actionable"
        },

        # --- HISTORICAL BASELINE RECORDS (WEEKS 1 & 2 FOR TREND AND RADAR ANALYSIS) ---
        {
            "feedback_id": "SYN_021",
            "feedback_text": "Clean store, quick billing today.",
            "store_id": "STORE_104",
            "store_name": "Store #104 (Indiranagar)",
            "region": "South",
            "channel": "Online Reviews",
            "timestamp": (base_date + pd.Timedelta(days=2, hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "General / Uncategorized",
            "ground_truth_severity": "Low",
            "ground_truth_actionable": "Informational"
        },
        {
            "feedback_id": "SYN_022",
            "feedback_text": "Need more parking space outside the store.",
            "store_id": "STORE_104",
            "store_name": "Store #104 (Indiranagar)",
            "region": "South",
            "channel": "In-App Feedback",
            "timestamp": (base_date + pd.Timedelta(days=4, hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Store Infrastructure & Facilities",
            "ground_truth_severity": "Low",
            "ground_truth_actionable": "Informational"
        },
        {
            "feedback_id": "SYN_023",
            "feedback_text": "Good shopping experience, fresh fruits available.",
            "store_id": "STORE_104",
            "store_name": "Store #104 (Indiranagar)",
            "region": "South",
            "channel": "Online Reviews",
            "timestamp": (base_date + pd.Timedelta(days=6, hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "General / Uncategorized",
            "ground_truth_severity": "Low",
            "ground_truth_actionable": "Informational"
        },
        {
            "feedback_id": "SYN_024",
            "feedback_text": "OOS on sunflower oil 1L packet.",
            "store_id": "STORE_104",
            "store_name": "Store #104 (Indiranagar)",
            "region": "South",
            "channel": "Staff WhatsApp / Shift Notes",
            "timestamp": (base_date + pd.Timedelta(days=8, hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Product Availability & OOS",
            "ground_truth_severity": "Low",
            "ground_truth_actionable": "Actionable"
        },
        {
            "feedback_id": "SYN_025",
            "feedback_text": "Checkout lines were manageable today.",
            "store_id": "STORE_201",
            "store_name": "Store #201 (Connaught Place)",
            "region": "North",
            "channel": "In-App Feedback",
            "timestamp": (base_date + pd.Timedelta(days=3, hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_category": "Billing & Checkout Operations",
            "ground_truth_severity": "Low",
            "ground_truth_actionable": "Informational"
        }
    ]

    return pd.DataFrame(records)
