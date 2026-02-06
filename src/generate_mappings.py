import pandas as pd
import json
import os

def generate_mappings():
    # Load raw data
    raw_data_path = 'data/raw/ncr_ride_bookings.csv'
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found.")
        return

    print("Loading raw data...")
    df = pd.read_csv(raw_data_path)

    mappings = {}

    # List of categorical columns to map
    categorical_cols = [
        'Vehicle Type',
        'Pickup Location',
        'Drop Location',
        'Payment Method'
    ]

    # Pre-processing for mapping generation (match EDA logic)
    df['Payment Method'] = df['Payment Method'].fillna('UNKNOWN')

    print("Generating mappings...")
    for col in categorical_cols:
        # Get unique values, sort them (LabelEncoder behavior), and create a map
        # Filter out NaNs if any (shouldn't be for Payment Method now)
        unique_vals = sorted([str(x) for x in df[col].unique() if pd.notna(x)])
        mappings[col] = {val: i for i, val in enumerate(unique_vals)}
        print(f"Mapped {col}: {len(unique_vals)} unique values.")

    print("Calculating medians...")
    driver_median = float(df['Driver Ratings'].median())
    customer_median = float(df['Customer Rating'].median())
    print(f"Driver Median: {driver_median}, Customer Median: {customer_median}")

    # Add medians to output
    output_data = {
        'mappings': mappings,
        'medians': {
            'driver_rating': driver_median,
            'customer_rating': customer_median
        }
    }

    # Save to JSON
    output_path = 'models/mappings.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Mappings saved to {output_path}")

if __name__ == "__main__":
    generate_mappings()
