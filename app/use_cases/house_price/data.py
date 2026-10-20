import numpy as np
import pandas as pd
from app.core.config import load_config
from app.use_cases.house_price.market_config import (
    BHK_AREA_BANDS,
    BHK_VALUES,
    BHK_WEIGHTS,
    DEFAULT_N_SAMPLES,
    FURNISHING_PREMIUM,
    FURNISHING_VALUES,
    FURNISHING_WEIGHTS,
    LOCALITY_RATES,
    LOCALITY_WEIGHTS,
    PRICE_NOISE_STD,
    PROPERTY_TYPE_PREMIUM,
    PROPERTY_TYPE_VALUES,
    PROPERTY_TYPE_WEIGHTS,
)


def _weighted_choice(rng: np.random.RandomState, values, weights, size):
    probs = np.asarray(weights, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(values, size=size, p=probs)


def generate_house_price_data(n_samples: int = DEFAULT_N_SAMPLES) -> pd.DataFrame:
    """Generates the synthetic Pune-aware property dataset (schema v2).

    Fully synthetic population calibrated against publicly available Pune
    locality asking-rate benchmarks (see market_config.py). Fixed seed
    (``random_seed`` from platform config) makes generation reproducible.

    Features: area_sqft, bedrooms, bathrooms, parking, location,
    property_age, floor_number, total_floors, furnishing_status,
    property_type. Target: price (INR, numeric).

    Args:
        n_samples: Number of rows to generate. Minimum 1.

    Returns:
        DataFrame with the 10 feature columns plus ``price``.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}")

    config = load_config()
    seed = config.get("random_seed", 42)

    # Local RandomState: reproducible without touching global numpy state.
    rng = np.random.RandomState(seed)

    locations = list(LOCALITY_RATES.keys())
    loc_weights = [LOCALITY_WEIGHTS[loc] for loc in locations]
    location = _weighted_choice(rng, locations, loc_weights, n_samples)

    bedrooms = _weighted_choice(rng, BHK_VALUES, BHK_WEIGHTS, n_samples).astype(int)

    # Carpet area from the BHK band (uniform inside the band).
    area_sqft = np.empty(n_samples, dtype=float)
    for bhk in BHK_VALUES:
        mask = bedrooms == bhk
        lo, hi = BHK_AREA_BANDS[bhk]
        area_sqft[mask] = rng.uniform(lo, hi, size=int(mask.sum()))

    # Bathrooms track bedrooms: same (60%), +1 (30%), -1 (10%, min 1).
    roll = rng.uniform(size=n_samples)
    bathrooms = np.where(
        roll < 0.60, bedrooms, np.where(roll < 0.90, bedrooms + 1, bedrooms - 1)
    ).astype(int)
    bathrooms = np.clip(bathrooms, 1, 6)

    property_type = _weighted_choice(
        rng, PROPERTY_TYPE_VALUES, PROPERTY_TYPE_WEIGHTS, n_samples
    )

    # Parking loosely follows size/type: 1 BHK mostly 0-1, larger/Villa more.
    parking = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        bhk = int(bedrooms[i])
        ptype = str(property_type[i])
        draw = rng.uniform()
        if ptype == "Villa" or bhk >= 4:
            parking[i] = 2 if draw < 0.70 else (3 if draw < 0.85 else 1)
        elif bhk >= 2:
            parking[i] = 1 if draw < 0.80 else (2 if draw < 0.95 else 0)
        else:
            parking[i] = 0 if draw < 0.35 else 1

    # Property age: mix of new and resale stock.
    age_roll = rng.uniform(size=n_samples)
    property_age = np.where(
        age_roll < 0.25,
        rng.uniform(0, 2, size=n_samples),
        np.where(
            age_roll < 0.65,
            rng.uniform(3, 10, size=n_samples),
            np.where(
                age_roll < 0.90,
                rng.uniform(11, 25, size=n_samples),
                rng.uniform(26, 40, size=n_samples),
            ),
        ),
    )

    # Floors: apartments in 4-22 storey towers, villas/row houses low-rise.
    total_floors = np.empty(n_samples, dtype=int)
    floor_number = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        ptype = str(property_type[i])
        if ptype == "Apartment":
            total = int(rng.randint(4, 23))
        else:
            total = int(rng.randint(1, 4))
        total_floors[i] = total
        floor_number[i] = int(rng.randint(0, total + 1))

    furnishing_status = _weighted_choice(
        rng, FURNISHING_VALUES, FURNISHING_WEIGHTS, n_samples
    )

    # --- Transparent synthetic price mechanism (INR) ---
    base_rate = np.array(
        [LOCALITY_RATES[str(loc)]["base_rate_per_sqft"] for loc in location],
        dtype=float,
    )
    loc_var = np.array(
        [LOCALITY_RATES[str(loc)]["variability"] for loc in location], dtype=float
    )
    # Intra-locality rate jitter: same locality, different micro-pockets.
    effective_rate = base_rate * (1.0 + rng.uniform(-loc_var, loc_var, size=n_samples))
    base_price = effective_rate * area_sqft

    furnishing_premium = np.array(
        [FURNISHING_PREMIUM[str(f)] for f in furnishing_status], dtype=float
    )
    type_premium = np.array(
        [PROPERTY_TYPE_PREMIUM[str(t)] for t in property_type], dtype=float
    )
    bathroom_adj = 0.02 * (bathrooms.astype(float) - 2.0)
    floor_adj = 0.05 * (floor_number.astype(float) / total_floors.astype(float))
    age_depreciation = np.minimum(property_age * 0.004, 0.25)
    parking_adj = 0.01 * parking.astype(float)

    multiplier = (
        1.0
        + furnishing_premium
        + type_premium
        + bathroom_adj
        + floor_adj
        - age_depreciation
        + parking_adj
    )
    multiplier = np.maximum(multiplier, 0.55)

    noise = rng.normal(loc=0.0, scale=PRICE_NOISE_STD, size=n_samples)
    price = base_price * multiplier * (1.0 + noise)
    price = np.maximum(price, 1500000.0)

    df = pd.DataFrame(
        {
            "area_sqft": area_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "parking": parking,
            "location": location,
            "property_age": property_age,
            "floor_number": floor_number,
            "total_floors": total_floors,
            "furnishing_status": furnishing_status,
            "property_type": property_type,
            "price": price,
        }
    )

    return df
