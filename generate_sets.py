import csv
import random
import argparse
from typing import Set

csv.field_size_limit(10000000)

def count_lines_and_select_validation(filename: str, V: int) -> Set[int]:
    taxon_ids = set()
    with open(filename, 'r') as f:
        next(f)  # Skip header
        for line in f:
            taxon_id = int(line.split('\t')[0])
            taxon_ids.add(taxon_id)
    
    return set(random.sample(list(taxon_ids), V))

def select_photos(research_photos: str, non_research_photos: str, count: int) -> list:
    research_list = research_photos.split(',') if research_photos else []
    non_research_list = non_research_photos.split(',') if non_research_photos else []
    
    selected = random.sample(research_list, min(count, len(research_list)))
    if len(selected) < count:
        selected += random.sample(non_research_list, min(count - len(selected), len(non_research_list)))
    
    return selected

def process_data(input_file: str, training_file: str, validation_file: str, T: int, Q: int, validation_taxons: Set[int]):
    with open(input_file, 'r') as infile, \
         open(training_file, 'w', newline='') as train_out, \
         open(validation_file, 'w', newline='') as val_out:
        
        reader = csv.reader(infile, delimiter='\t')
        train_writer = csv.writer(train_out)
        val_writer = csv.writer(val_out)

        # Write headers
        train_writer.writerow(['taxon_id', 'photo_id'])
        val_writer.writerow(['taxon_id', 'photo_id'])

        next(reader)  # Skip header
        for row in reader:
            taxon_id = int(row[0])
            research_photos = row[1]
            non_research_photos = row[2]

            # Always select T photos for training
            selected_training = select_photos(research_photos, non_research_photos, T)
            for photo in selected_training:
                train_writer.writerow([taxon_id, photo])

            if taxon_id in validation_taxons:
                # Select Q photos for validation, excluding those already selected for training
                remaining_research = ','.join([p for p in research_photos.split(',') if p not in selected_training])
                remaining_non_research = ','.join([p for p in non_research_photos.split(',') if p not in selected_training])
                selected_validation = select_photos(remaining_research, remaining_non_research, Q)
                for photo in selected_validation:
                    val_writer.writerow([taxon_id, photo])

def main():
    parser = argparse.ArgumentParser(description="Process photo data and create training/validation sets.")
    parser.add_argument("input_file", help="Input TSV file")
    parser.add_argument("T", type=int, help="Number of images per taxon for training")
    parser.add_argument("V", type=int, help="Number of validation taxons")
    parser.add_argument("Q", type=int, help="Number of validation items per taxon")
    args = parser.parse_args()

    print("Counting lines and selecting validation taxons...")
    validation_taxons = count_lines_and_select_validation(args.input_file, args.V)

    print("Processing data and writing output files...")
    process_data(args.input_file, "training_set.csv", "validation_set.csv", args.T, args.Q, validation_taxons)

    print("Done!")

if __name__ == "__main__":
    main()
