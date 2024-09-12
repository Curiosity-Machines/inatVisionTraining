import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

# Constants
RADIUS = 250  # km
EARTH_RADIUS = 6371  # km

# Global bounding box
MIN_LAT, MAX_LAT = -90, 90
MIN_LON, MAX_LON = -180, 180

# Hard-coded total number of rows (excluding header)
TOTAL_ROWS = 264829643

def create_global_grid():
    lat_distance = 2 * RADIUS
    lon_distance = 2 * RADIUS  # This will be adjusted for each latitude

    num_rows = int(np.ceil((MAX_LAT - MIN_LAT) / (lat_distance / EARTH_RADIUS * 180 / np.pi)))
    num_cols = int(np.ceil((MAX_LON - MIN_LON) / (lon_distance / EARTH_RADIUS * 180 / np.pi)))

    return num_rows, num_cols

def assign_global_region(lat, lon, num_rows, num_cols):
    lat_size = (MAX_LAT - MIN_LAT) / num_rows
    lon_size = (MAX_LON - MIN_LON) / num_cols

    row = np.clip(((lat - MIN_LAT) / lat_size).astype(int), 0, num_rows - 1)
    col = np.clip(((lon - MIN_LON) / lon_size).astype(int), 0, num_cols - 1)
    return row * num_cols + col

def process_csv(input_file):
    # Create global grid
    num_rows, num_cols = create_global_grid()
    print(f"Grid size: {num_rows} rows, {num_cols} columns", file=sys.stderr)

    # Define column dtypes
    dtypes = {
        'observation_uuid': str,
        'photo_id': 'Int64',  # Use Int64 to allow for NaN values
        'observer_id': 'Int64',
        'latitude': float,
        'longitude': float,
        'positional_accuracy': 'float64',  # Use float64 to allow for NaN values
        'taxon_id': 'Int64',
        'quality_grade': str,
        'observed_on': str
    }

    # Read CSV in chunks, keeping all columns
    chunk_size = 100000  # Adjust based on your available memory
    chunks = pd.read_csv(input_file, chunksize=chunk_size, sep='\t', dtype=dtypes, 
                         na_values=['\\N', 'NULL', '', ' '], keep_default_na=True)

    first_chunk = True
    processed_rows = 0
    skipped_rows = 0
    
    with tqdm(total=TOTAL_ROWS, desc="Processing", unit="rows") as pbar:
        for chunk in chunks:
            # Filter out rows with invalid lat or long
            valid_mask = chunk['latitude'].notna() & chunk['longitude'].notna() & \
                         chunk['latitude'].between(MIN_LAT, MAX_LAT) & \
                         chunk['longitude'].between(MIN_LON, MAX_LON)
            
            valid_chunk = chunk[valid_mask].copy()  # Create a copy to avoid SettingWithCopyWarning
            skipped_rows += len(chunk) - len(valid_chunk)

            if not valid_chunk.empty:
                # Assign region index
                valid_chunk['region_index'] = assign_global_region(
                    valid_chunk['latitude'].values, 
                    valid_chunk['longitude'].values,
                    num_rows, num_cols
                )
                
                # Write to stdout, using tab as delimiter
                valid_chunk.to_csv(sys.stdout, index=False, header=first_chunk, sep='\t')
                first_chunk = False
            
            # Update progress bar
            processed_rows += len(chunk)
            pbar.update(len(chunk))
            
            # Print progress to stderr
            progress_percent = (processed_rows / TOTAL_ROWS) * 100
            print(f"Processed {processed_rows:,} of {TOTAL_ROWS:,} rows ({progress_percent:.2f}%)", 
                  file=sys.stderr)

    print(f"Total rows skipped due to invalid coordinates: {skipped_rows}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_csv_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    process_csv(input_file)
