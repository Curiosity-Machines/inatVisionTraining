import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from collections import defaultdict

def process_files(tsv_file):
    # Define column dtypes
    dtypes = {
        'photo_id': 'Int64',
        'taxon_id': 'Int64',
        'quality_grade': str
    }

    # Initialize defaultdict to store photo IDs
    taxon_photos = { 'research': [], 'needs_id': [] }

    # Get total number of rows (excluding header)
    total_rows = 207277540

    # Process the TSV file in chunks
    chunk_size = 100000  # Adjust based on your available memory
    #chunks = pd.read_csv(tsv_file, chunksize=chunk_size, sep='\t', dtype=dtypes, 
    #                     na_values=['\\N', 'NULL', '', ' '], keep_default_na=True)
    chunks = pd.read_csv(tsv_file, chunksize=chunk_size, sep='\t', dtype=dtypes, 
                         na_values=['\\N', 'NULL', '', ' '], keep_default_na=True)

    processed_rows = 0
    skipped_rows = 0
    last_taxon_id = None
    print("taxon_id\tresearch_photos\tnon_research_photos")

    with tqdm(total=total_rows, desc="Processing", unit="rows") as pbar:
        for chunk in chunks:
            # Filter out rows with NaN values in critical columns


            # Assume sorted by taxon id
            for _, row in chunk.iterrows():
                taxon_id = str(row['taxon_id'])

                if last_taxon_id != taxon_id and last_taxon_id is not None:
                    research_photos = ','.join(taxon_photos['research'])
                    needs_id_photos = ','.join(taxon_photos['needs_id'])

                    print(f"{last_taxon_id}\t{research_photos}\t{needs_id_photos}")

                    taxon_photos = { 'research': [], 'needs_id': [] }

                last_taxon_id = taxon_id

                photo_id = str(row['photo_id'])
                if row['quality_grade'] == 'research':
                    taxon_photos['research'].append(photo_id)
                else:
                    taxon_photos['needs_id'].append(photo_id)

            # Update progress bar
            processed_rows += len(chunk)
            pbar.update(len(chunk))

            # Print progress to stderr
            progress_percent = (processed_rows / total_rows) * 100
            print(f"Processed {processed_rows:,} of {total_rows:,} rows ({progress_percent:.2f}%)", 
                  file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python photo_list_gen.py <input_tsv_file>")
        sys.exit(1)
    
    tsv_file = sys.argv[1]
    process_files(tsv_file)
