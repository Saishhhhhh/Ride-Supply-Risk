import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import math

class RideRiskPredictor:
    def __init__(self, model_dir='models'):
        self.model_path = os.path.join(model_dir, 'multitask_ride_model.keras')
        self.scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        self.mappings_path = os.path.join(model_dir, 'mappings.json')

        self._load_artifacts()

    def _load_artifacts(self):
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        
        print(f"Loading scaler from {self.scaler_path}...")
        self.scaler = joblib.load(self.scaler_path)

        print(f"Loading mappings from {self.mappings_path}...")
        with open(self.mappings_path, 'r') as f:
            data = json.load(f)
            self.mappings = data['mappings']
            self.medians = data['medians']

    def _get_time_features(self, dt_str):
        # Allow input as datetime object or string
        if isinstance(dt_str, str):
            try:
                dt = datetime.fromisoformat(dt_str)
            except ValueError:
                # Fallback to current time if parsing fails or handle formats
                dt = datetime.now()
        elif isinstance(dt_str, datetime):
            dt = dt_str
        else:
            dt = datetime.now()

        hour = dt.hour
        day_of_week = dt.weekday() # 0=Monday, 6=Sunday

        # Cyclical encoding
        # Hour: 0-23. 2pi * hour / 24
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        # Day: 0-6. 2pi * day / 7
        day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        day_cos = np.cos(2 * np.pi * day_of_week / 7.0)

        return hour_sin, hour_cos, day_sin, day_cos

    def preprocess(self, input_data):
        """
        input_data: dict containing:
            - vehicle_type
            - pickup_location
            - drop_location
            - booking_value
            - ride_distance
            - payment_method
            - driver_rating (optional)
            - customer_rating (optional)
            - time_of_booking (optional)
        """
        
        # 1. Map Categorical Variables
        # Use .get() with a default or error? 
        # Safest is to handle unknown categories by using a fallback or error. 
        # For now, if unknown, maybe use 0 or raising error. 
        # Let's try to map, if fail, raise ValueError.
        
        def map_val(col, val):
            mapping = self.mappings.get(col, {})
            if val not in mapping:
                # Try handling if it's string/int mismatch
                if str(val) in mapping:
                    return mapping[str(val)]
                # Fallback to first category? or raise?
                # For robust API, maybe raise error telling user invalid category.
                raise ValueError(f"Unknown category '{val}' for field '{col}'. Allowed: {list(mapping.keys())[:5]}...")
            return mapping[val]

        vehicle_type_enc = map_val('Vehicle Type', input_data.get('vehicle_type'))
        pickup_loc_enc = map_val('Pickup Location', input_data.get('pickup_location'))
        drop_loc_enc = map_val('Drop Location', input_data.get('drop_location'))
        
        # Payment Method: Handle unknown logic similar to training (fill UNKNOWN)
        pymt = input_data.get('payment_method')
        if not pymt:
            pymt = 'UNKNOWN'
        # Also strictly map 'UNKNOWN' if it's in mapping (which it should be)
        if pymt not in self.mappings['Payment Method']:
             pymt = 'UNKNOWN'
        payment_enc = self.mappings['Payment Method'][pymt]

        # 2. Handle Ratings
        driver_rating = input_data.get('driver_rating')
        if driver_rating is None:
            dr_missing = 1
            dr_filled = self.medians['driver_rating']
        else:
            dr_missing = 0
            dr_filled = float(driver_rating)

        customer_rating = input_data.get('customer_rating')
        if customer_rating is None:
            cr_missing = 1
            cr_filled = self.medians['customer_rating']
        else:
            cr_missing = 0
            cr_filled = float(customer_rating)

        # 3. Time Features
        time_input = input_data.get('time_of_booking', datetime.now())
        hour_sin, hour_cos, day_sin, day_cos = self._get_time_features(time_input)

        # 4. Construct DataFrame
        # MUST match columns of X during training BEFORE scaling
        data = {
            "Vehicle Type": [vehicle_type_enc],
            "Pickup Location": [pickup_loc_enc],
            "Drop Location": [drop_loc_enc],
            "Booking Value": [float(input_data.get('booking_value', 0))],
            "Ride Distance": [float(input_data.get('ride_distance', 0))],
            "Payment Method": [payment_enc],
            "driver_rating_missing": [dr_missing],
            "driver_rating_filled": [dr_filled],
            "customer_rating_missing": [cr_missing],
            "customer_rating_filled": [cr_filled],
            "hour_sin": [hour_sin],
            "hour_cos": [hour_cos],
            "day_sin": [day_sin],
            "day_cos": [day_cos]
        }
        
        df = pd.DataFrame(data)

        # 5. Scale using ColumnTransformer
        # This will reorder columns: [Scaled...] + [Passthrough...]
        X_scaled = self.scaler.transform(df)
        
        return X_scaled

    def predict(self, input_data):
        X = self.preprocess(input_data)
        
        # Returns list of arrays: [cancel_prob, stress_prob]
        predictions = self.model.predict(X, verbose=0)
        
        cancel_prob = float(predictions[0][0][0])
        stress_prob = float(predictions[1][0][0])
        
        return {
            "cancellation_probability": cancel_prob,
            "supply_stress_probability": stress_prob
        }

if __name__ == "__main__":
    # Test run
    predictor = RideRiskPredictor()
    sample = {
        "vehicle_type": "Auto",
        "pickup_location": "Rohini West",
        "drop_location": "Dwarka Mor",
        "booking_value": 450,
        "ride_distance": 12.5,
        "payment_method": "UPI",
        "driver_rating": 4.5,
        "customer_rating": None # Should use median
    }
    print("Sample Input:", sample)
    result = predictor.predict(sample)
    print("Prediction:", result)
