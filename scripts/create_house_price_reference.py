import os
import sys

# Ensure the root project directory is in the path when running as a standalone script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import resolve_project_path
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.market_config import DEFAULT_N_SAMPLES
from app.use_cases.house_price.validation import validate_house_price_data

def create_reference_dataset(output_path: str = "data/reference/house_price_reference.csv") -> None:
    """Generates the Pune-aware reference dataset, validates it, and saves it.

    Fully synthetic population calibrated against publicly available Pune
    locality asking-rate benchmarks (see market_config.py). Fixed seed
    makes regeneration reproducible.

    Args:
        output_path: Destination CSV path, resolved against project root.
    """
    # Anchor relative destinations at the project root (Step 56).
    output_path = str(resolve_project_path(output_path))
    # 1. Generate the synthetic Pune-aware population.
    df = generate_house_price_data(n_samples=DEFAULT_N_SAMPLES)

    # 2. Validate the data (raises DataValidationError; prevents saving).
    validate_house_price_data(df)

    # 3. Create the directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 4. Save the generated dataset
    df.to_csv(output_path, index=False)
    print(f"Reference dataset created at: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)} ({', '.join(df.columns)})")
    print(f"Locations: {df['location'].nunique()}")
    print(f"Null values: {int(df.isnull().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print(
        f"Min price: Rs {df['price'].min():,.0f} | "
        f"Median price: Rs {df['price'].median():,.0f} | "
        f"Max price: Rs {df['price'].max():,.0f}"
    )
    print()
    print(f"{'Location':<14}{'Rows':>7}  {'Median area':>12}  {'Median price':>14}  {'Approx Rs/sqft':>15}")
    for loc, group in df.groupby("location"):
        med_area = group["area_sqft"].median()
        med_price = group["price"].median()
        print(
            f"{loc:<14}{len(group):>7,}  {med_area:>12,.0f}  {med_price:>14,.0f}  {med_price / med_area:>15,.0f}"
        )

if __name__ == "__main__":
    create_reference_dataset()
