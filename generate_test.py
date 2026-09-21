import pandas as pd
import numpy as np
import os

def generate_smart_city_csv(filename="uploads/surveillance_data.csv", rows=500):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.random.seed(42)
    
    # 1. Generate Baseline "Normal" Data
    # We use a tight standard deviation so the model learns a clear "Normal" pattern
    data = {
        'timestamp_hour': np.random.randint(0, 24, rows),
        'traffic_density': np.random.normal(45, 8, rows),    
        'noise_level': np.random.normal(55, 4, rows),       
        'pedestrian_count': np.random.normal(150, 25, rows), 
        'emergency_vehicle_siren': np.zeros(rows)           
    }
    
    df = pd.DataFrame(data)

    # 2. Explicitly Define Unusual Categories
    # We create high-contrast spikes so the Isolation Forest flags them accurately
    
    # Category: Public Protest / Festival (High everything)
    idx_protest = range(10, 21)
    df.loc[idx_protest, 'traffic_density'] = np.random.uniform(95, 115, len(idx_protest))
    df.loc[idx_protest, 'noise_level'] = np.random.uniform(105, 125, len(idx_protest))
    df.loc[idx_protest, 'pedestrian_count'] = np.random.uniform(900, 1300, len(idx_protest))

    # Category: Structural Incident / Blast (Extreme Noise, Zero Traffic)
    idx_incident = range(250, 256)
    df.loc[idx_incident, 'traffic_density'] = np.random.uniform(0, 5, len(idx_incident))
    df.loc[idx_incident, 'noise_level'] = np.random.uniform(140, 160, len(idx_incident))

    # Category: Emergency Response (Siren Active)
    idx_emergency = range(400, 406)
    df.loc[idx_emergency, 'emergency_vehicle_siren'] = 1
    df.loc[idx_emergency, 'traffic_density'] = np.random.uniform(0, 10, len(idx_emergency))

    # 3. Final Rectification & Cleanup
    # Apply clip only to numeric data to prevent string comparison errors
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)

    # Apply the Ground Truth Label
    df['category'] = 'Normal'
    df.loc[list(idx_protest) + list(idx_incident) + list(idx_emergency), 'category'] = 'Unusual'

    # 4. Save to CSV
    df.to_csv(filename, index=False)
    
    # Print a quick summary to the console
    print(f"✅ Data generated successfully!")
    print(f"📍 Path: {filename}")
    print(f"📊 Normal records: {len(df[df['category']=='Normal'])}")
    print(f"🚨 Unusual records: {len(df[df['category']=='Unusual'])}")

if __name__ == "__main__":
    generate_smart_city_csv()