"""Pune market calibration for the synthetic House Price dataset (Phase 1).

The dataset is FULLY SYNTHETIC. Values below are approximate calibration
anchors derived from publicly available 2026 Pune locality asking-rate
benchmarks (MagicBricks locality pages, Nivaas research pages, SquareYards
trend reports), NOT transaction records:

- Hinjewadi IT belt ........ ~Rs 6,600-10,200/sqft (Phase 3 ~Rs 9,600)
- Wakad .................... ~Rs 7,500-11,500 (avg ~Rs 9,490)
- Baner .................... resale ~Rs 8,500-12,000, new ~Rs 10,000-14,000
- Pune city average ........ ~Rs 12,000-13,400 (premium central localities
  pull the mean up; our 12 mid-segment localities sit below it)

Each locality maps to a single base_rate_per_sqft chosen inside its
publicly observed range, plus an intra-locality variability band used by
the generator. Treat these as documented calibration assumptions; verify
against current listings before any real-world use.
"""

# Data-schema version. v1 = legacy 5-feature schema
# (area/bedrooms/bathrooms/property_age/parking -> house_price).
# v2 = Pune-aware 10-feature schema (-> price).
DATA_SCHEMA_VERSION = 2

TARGET_COLUMN = "price"

NUMERIC_FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "parking",
    "property_age",
    "floor_number",
    "total_floors",
]

CATEGORICAL_FEATURES = [
    "location",
    "furnishing_status",
    "property_type",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Locality calibration: base_rate_per_sqft is the anchor inside the
# publicly observed range; variability is the +/- band applied per row.
LOCALITY_RATES = {
    "Wakad": {"base_rate_per_sqft": 8800, "variability": 0.12},
    "Hinjawadi": {"base_rate_per_sqft": 8200, "variability": 0.12},
    "Kharadi": {"base_rate_per_sqft": 9500, "variability": 0.12},
    "Baner": {"base_rate_per_sqft": 10500, "variability": 0.13},
    "Balewadi": {"base_rate_per_sqft": 9000, "variability": 0.12},
    "Kothrud": {"base_rate_per_sqft": 11500, "variability": 0.12},
    "Hadapsar": {"base_rate_per_sqft": 7500, "variability": 0.13},
    "Wagholi": {"base_rate_per_sqft": 6800, "variability": 0.14},
    "Viman Nagar": {"base_rate_per_sqft": 10000, "variability": 0.12},
    "Aundh": {"base_rate_per_sqft": 11000, "variability": 0.12},
    "Punawale": {"base_rate_per_sqft": 7800, "variability": 0.13},
    "Ravet": {"base_rate_per_sqft": 7200, "variability": 0.13},
}

# Sampling share per locality (high-volume corridors get more rows).
LOCALITY_WEIGHTS = {
    "Hinjawadi": 0.12,
    "Wakad": 0.11,
    "Hadapsar": 0.10,
    "Wagholi": 0.09,
    "Kharadi": 0.09,
    "Baner": 0.08,
    "Punawale": 0.08,
    "Ravet": 0.07,
    "Balewadi": 0.07,
    "Kothrud": 0.07,
    "Viman Nagar": 0.06,
    "Aundh": 0.06,
}

FURNISHING_VALUES = ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]
FURNISHING_WEIGHTS = [0.45, 0.35, 0.20]
FURNISHING_PREMIUM = {
    "Unfurnished": 0.00,
    "Semi-Furnished": 0.04,
    "Fully-Furnished": 0.08,
}

PROPERTY_TYPE_VALUES = ["Apartment", "Villa", "Row House"]
PROPERTY_TYPE_WEIGHTS = [0.78, 0.10, 0.12]
PROPERTY_TYPE_PREMIUM = {
    "Apartment": 0.00,
    "Row House": 0.06,
    "Villa": 0.15,
}

# BHK mix and carpet-area band per BHK (sqft).
BHK_VALUES = [1, 2, 3, 4, 5]
BHK_WEIGHTS = [0.18, 0.34, 0.30, 0.13, 0.05]
BHK_AREA_BANDS = {
    1: (400, 700),
    2: (700, 1100),
    3: (1100, 1600),
    4: (1500, 2200),
    5: (2000, 3000),
}

DEFAULT_N_SAMPLES = 8500
PRICE_NOISE_STD = 0.08
