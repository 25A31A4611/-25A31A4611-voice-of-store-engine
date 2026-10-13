"""
Configuration, Taxonomy, Department Routing, and Policy Rules
Talent AP Mega Hackathon — Problem Statement R5: Voice-of-Store Engine
"""

from typing import Dict, List, Set

APP_TITLE = "Voice-of-Store Engine"
APP_SUBTITLE = "Turning Store Chatter Into Action"
PROBLEM_STATEMENT = "R5: Turning Store Chatter Into Action — The Voice-of-Store Engine"
TAGLINE = "From scattered feedback to actionable store intelligence."

# Canonical Retail Feedback Channels
CHANNELS = [
    "Online Reviews",
    "In-App Feedback",
    "Staff WhatsApp / Shift Notes",
    "Call-Centre Transcripts"
]

# Canonical Issue Categories & Descriptions
TAXONOMY_CATEGORIES = [
    "Food Safety & Hygiene",
    "Customer Service & Staff Conduct",
    "Billing & Checkout Operations",
    "Product Availability & OOS",
    "Store Infrastructure & Facilities",
    "Pricing & Promotions Discrepancy",
    "App & Digital Ordering",
    "Delivery & Packaging",
    "General / Uncategorized"
]

# Default Department Mappings
CATEGORY_DEPARTMENT_MAP: Dict[str, str] = {
    "Food Safety & Hygiene": "Store Operations",
    "Customer Service & Staff Conduct": "HR",
    "Billing & Checkout Operations": "Store Operations",
    "Product Availability & OOS": "Supply Chain",
    "Store Infrastructure & Facilities": "Store Operations",
    "Pricing & Promotions Discrepancy": "Merchandising",
    "App & Digital Ordering": "Digital",
    "Delivery & Packaging": "Supply Chain",
    "General / Uncategorized": "Store Operations"
}

# Available Departments for Routing
DEPARTMENTS = [
    "Store Operations",
    "Supply Chain",
    "Merchandising",
    "Digital",
    "HR"
]

# Severity Levels
SEVERITY_LEVELS = ["Critical", "High", "Medium", "Low"]

# Actionability Levels
ACTIONABILITY_LEVELS = ["Highly Actionable", "Actionable", "Informational", "Vague / Non-Actionable"]

# Mandatory High-Severity Policy Triggers (Keywords / Patterns)
# Safety, allergen, hygiene, pest, staff-misconduct complaints MUST be escalated regardless of frequency
MANDATORY_POLICY_RULES = [
    {
        "id": "FOOD_SAFETY_ALLERGEN",
        "name": "Food Safety & Allergen Hazard",
        "category": "Food Safety & Hygiene",
        "keywords": [
            "allergy", "allergic", "allergen", "food poison", "vomit", "sick after eating",
            "expired", "expiry", "stale food", "bad smell food", "rotten", "fungus", "mold", "mould",
            "spoiled milk", "curdled", "worms in", "insect in", "hair in food", "glass piece",
            "plastic in food", "foreign object"
        ],
        "mandatory_severity": "Critical",
        "department": "Store Operations",
        "reason": "Immediate health and regulatory risk: Potential allergen or food safety violation requiring store manager inspection.",
        "recommended_action": "Immediately inspect batch, quarantine affected inventory, test storage temperature, notify QA."
    },
    {
        "id": "PEST_INFESTATION",
        "name": "Pest & Infestation Alert",
        "category": "Food Safety & Hygiene",
        "keywords": [
            "cockroach", "roach", "rat", "mouse", "mice", "rodent", "lizard", "flies everywhere",
            "pest", "termite", "bedbug", "🪳", "🐀", "🐁"
        ],
        "mandatory_severity": "High",
        "department": "Store Operations",
        "reason": "Sanitation breach: Pest sighting on store floor or stockroom violates health standards.",
        "recommended_action": "Deploy pest control vendor within 24 hours; seal entry points and audit dry food storage."
    },
    {
        "id": "PHYSICAL_SAFETY_HAZARD",
        "name": "Store Physical & Fire Safety",
        "category": "Store Infrastructure & Facilities",
        "keywords": [
            "fire exit blocked", "fire alarm", "short circuit", "sparking", "exposed wire",
            "electric shock", "gas leak", "water flood", "ceiling leak heavy", "glass shattered",
            "slippery floor no sign", "fell down", "tripped", "injured"
        ],
        "mandatory_severity": "Critical",
        "department": "Store Operations",
        "reason": "Life safety and liability hazard: Customer or employee hazard on store premises.",
        "recommended_action": "Cordon off danger area immediately; summon facility emergency maintenance team."
    },
    {
        "id": "STAFF_MISCONDUCT_HARASSMENT",
        "name": "Staff Misconduct & Harassment Allegation",
        "category": "Customer Service & Staff Conduct",
        "keywords": [
            "harass", "harassment", "abusive", "threatened", "shouted at me", "slapped",
            "hit customer", "physical fight", "bribe", "stole money", "cashier theft", "racist",
            "casteist", "misbehavior severe", "drunk staff"
        ],
        "mandatory_severity": "Critical",
        "department": "HR",
        "reason": "Severe compliance and workplace conduct breach requiring confidential investigation.",
        "recommended_action": "Log confidential incident report, review CCTV footage, conduct HR formal inquiry with store manager."
    }
]

# Actionability indicators (concrete details: product name, employee name, bill number, counter number)
ACTIONABLE_PATTERNS = [
    r"bill\s*(no|number|#)?\s*[:#]?\s*\w+",
    r"counter\s*(no|number|#)?\s*[:#]?\s*\d+",
    r"order\s*(id|no|#)?\s*[:#]?\s*\w+",
    r"staff\s*(named|called)?\s+[A-Z][a-z]+",
    r"aisle\s*\d+",
    r"batch\s*(no|#)?\s*[:#]?\s*\w+",
    r"shelf\s*\d+",
    r"mrp\s*rs\.?\s*\d+",
    r"charged\s*rs\.?\s*\d+",
    r"rupees\s*\d+"
]

# UI Palette - Enterprise Navy & Cyan/Teal
COLORS = {
    "navy_dark": "#0B192C",
    "navy_medium": "#1E3E62",
    "teal_primary": "#00A896",
    "cyan_accent": "#028090",
    "light_bg": "#F8F9FA",
    "card_bg": "#FFFFFF",
    "text_dark": "#1F2937",
    "text_muted": "#6B7280",
    "border_light": "#E5E7EB",
    "critical": "#DC2626",
    "high": "#EA580C",
    "medium": "#D97706",
    "low": "#16A34A"
}
