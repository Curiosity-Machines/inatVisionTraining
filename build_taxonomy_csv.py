import json
import csv
import argparse
import sys
from collections import defaultdict

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process taxon JSON and CSV files to generate a combined CSV.')
    parser.add_argument('--json_file', required=True, help='Path to the JSON file mapping taxon_id to leaf_class_id.')
    parser.add_argument('--taxa_csv', required=True, help='Path to the taxa.csv file.')
    parser.add_argument('--output_csv', required=True, help='Path to the output CSV file.')
    return parser.parse_args()

def read_json(json_file):
    """
    Reads the JSON file and returns a dictionary mapping taxon_id to leaf_class_id.
    """
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            if not isinstance(json_data, dict):
                print(f"Error: JSON file {json_file} does not contain a valid dictionary.")
                sys.exit(1)
            return json_data
    except Exception as e:
        print(f"Error reading JSON file {json_file}: {e}")
        sys.exit(1)

def read_taxa_csv(taxa_csv):
    """
    Reads the taxa.csv file and returns a dictionary mapping taxon_id to its information.
    """
    taxa_dict = {}
    try:
        with open(taxa_csv, 'r', newline='', encoding='utf-8') as f:
            # Detect the delimiter (comma or tab)
            sample = f.read(1024)
            f.seek(0)
            if '\t' in sample:
                delimiter = '\t'
            else:
                delimiter = ','
            
            reader = csv.DictReader(f, delimiter=delimiter)
            required_fields = {'taxon_id', 'ancestry', 'rank_level', 'name'}
            if not required_fields.issubset(reader.fieldnames):
                print(f"Error: taxa.csv is missing one of the required fields: {required_fields}")
                sys.exit(1)
            
            for row in reader:
                taxon_id = row['taxon_id'].strip()
                if not taxon_id:
                    continue  # Skip rows without taxon_id
                taxa_dict[taxon_id] = {
                    'ancestry': row['ancestry'].strip(),
                    'rank_level': row['rank_level'].strip(),
                    'name': row['name'].strip(),
                    # Additional fields can be added here if needed
                }
        return taxa_dict
    except Exception as e:
        print(f"Error reading taxa CSV file {taxa_csv}: {e}")
        sys.exit(1)

def build_taxon_paths(json_taxa, taxa_dict):
    """
    Builds a set of all taxon_ids to be included based on the JSON taxa and their ancestries.
    Returns a set of taxon_ids and a list of paths.
    """
    all_taxon_ids = set()
    paths = []
    
    for taxon_id in json_taxa.keys():
        if taxon_id not in taxa_dict:
            print(f"Warning: taxon_id {taxon_id} from JSON not found in taxa.csv. Skipping.")
            continue
        ancestry = taxa_dict[taxon_id]['ancestry']
        ancestry_list = ancestry.split('/') if ancestry else []
        path = ancestry_list + [taxon_id]
        paths.append(path)
        all_taxon_ids.update(path)
    
    return all_taxon_ids, paths

def build_taxon_id_to_path(paths):
    """
    Builds a mapping from taxon_id to its full path for sorting purposes.
    """
    taxon_to_path = {}
    for path in paths:
        for i, tid in enumerate(path):
            current_path = path[:i+1]
            if tid not in taxon_to_path or len(current_path) > len(taxon_to_path[tid]):
                taxon_to_path[tid] = current_path
    return taxon_to_path

def generate_taxon_rows(all_taxon_ids, taxa_dict, json_taxa, taxon_to_path):
    """
    Generates a list of dictionaries representing each row for the output CSV.
    """
    taxon_rows = []
    for taxon_id in all_taxon_ids:
        taxon_info = taxa_dict.get(taxon_id)
        if not taxon_info:
            print(f"Warning: taxon_id {taxon_id} not found in taxa.csv. Skipping.")
            continue
        ancestry = taxon_info['ancestry']
        ancestry_list = ancestry.split('/') if ancestry else []
        parent_taxon_id = ancestry_list[-1] if ancestry_list else ''
        rank_level = taxon_info['rank_level']
        name = taxon_info['name']
        leaf_class_id = json_taxa.get(taxon_id, '')
        taxon_rows.append({
            'parent_taxon_id': parent_taxon_id,
            'taxon_id': taxon_id,
            'rank_level': rank_level,
            'leaf_class_id': leaf_class_id,
            'name': name,
            'depth': len(taxon_to_path.get(taxon_id, []))
        })
    return taxon_rows

def sort_taxon_rows(taxon_rows):
    """
    Sorts the taxon rows first by depth (ascending) and then by taxon_id (ascending).
    This ensures that parent taxa appear before their children.
    """
    return sorted(taxon_rows, key=lambda x: (x['depth'], int(x['taxon_id']) if x['taxon_id'].isdigit() else x['taxon_id']))

def write_output_csv(taxon_rows_sorted, output_csv):
    """
    Writes the sorted taxon rows to the output CSV file.
    """
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['parent_taxon_id', 'taxon_id', 'rank_level', 'leaf_class_id', 'name']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in taxon_rows_sorted:
                writer.writerow({
                    'parent_taxon_id': row['parent_taxon_id'],
                    'taxon_id': row['taxon_id'],
                    'rank_level': row['rank_level'],
                    'leaf_class_id': row['leaf_class_id'],
                    'name': row['name']
                })
        print(f"Output CSV successfully written to {output_csv}")
    except Exception as e:
        print(f"Error writing to output CSV file {output_csv}: {e}")
        sys.exit(1)

def main():
    args = parse_arguments()
    
    # Step 1: Read JSON file
    json_taxa = read_json(args.json_file)
    
    # Step 2: Read taxa.csv file
    taxa_dict = read_taxa_csv(args.taxa_csv)
    
    # Step 3: Build taxon paths
    all_taxon_ids, paths = build_taxon_paths(json_taxa, taxa_dict)
    
    if not all_taxon_ids:
        print("No valid taxon_ids found to process. Exiting.")
        sys.exit(0)
    
    # Step 4: Build taxon_id to path mapping for sorting
    taxon_to_path = build_taxon_id_to_path(paths)
    
    # Step 5: Generate taxon rows
    taxon_rows = generate_taxon_rows(all_taxon_ids, taxa_dict, json_taxa, taxon_to_path)
    
    if not taxon_rows:
        print("No taxon rows to write. Exiting.")
        sys.exit(0)
    
    # Step 6: Sort taxon rows
    taxon_rows_sorted = sort_taxon_rows(taxon_rows)
    
    # Step 7: Write to output CSV
    write_output_csv(taxon_rows_sorted, args.output_csv)

if __name__ == "__main__":
    main()

