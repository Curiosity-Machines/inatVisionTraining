import csv
import random
import argparse
from typing import Set, List

csv.field_size_limit(10000000)

def count_lines_and_select_taxons(filename: str, num_taxons: int) -> Set[int]:
    taxon_ids = set()
    with open(filename, 'r') as f:
        next(f)  # Skip header
        for line in f:
            taxon_id = int(line.split('\t')[0])
            taxon_ids.add(taxon_id)
    
    return set(random.sample(list(taxon_ids), num_taxons))

def select_photos(research_photos: str, non_research_photos: str, count: int) -> List[str]:
    research_list = research_photos.split(',') if research_photos else []
    non_research_list = non_research_photos.split(',') if non_research_photos else []
    
    selected = random.sample(research_list, min(count, len(research_list)))
    if len(selected) < count:
        selected += random.sample(non_research_list, min(count - len(selected), len(non_research_list)))
    
    return selected

def process_data(input_file: str, training_file: str, validation_file: str, test_file: str, 
                 train_photos_per_taxon: int, validation_taxons: Set[int], test_taxons: Set[int], 
                 validation_photos_per_taxon: int, test_photos_per_taxon: int):
    with open(input_file, 'r') as infile, \
         open(training_file, 'w', newline='') as train_out, \
         open(validation_file, 'w', newline='') as val_out, \
         open(test_file, 'w', newline='') as test_out:
        
        reader = csv.reader(infile, delimiter='\t')
        train_writer = csv.writer(train_out)
        val_writer = csv.writer(val_out)
        test_writer = csv.writer(test_out)

        # Write headers
        headers = ['taxon_id', 'photo_id']
        train_writer.writerow(headers)
        val_writer.writerow(headers)
        test_writer.writerow(headers)

        next(reader)  # Skip header
        for row in reader:
            taxon_id = int(row[0])
            research_photos = row[1]
            non_research_photos = row[2]

            # Always select train_photos_per_taxon photos for training
            selected_training = select_photos(research_photos, non_research_photos, train_photos_per_taxon)
            for photo in selected_training:
                train_writer.writerow([taxon_id, photo])

            remaining_photos = set(research_photos.split(',') + non_research_photos.split(',')) - set(selected_training)

            if taxon_id in validation_taxons:
                selected_validation = random.sample(list(remaining_photos), min(validation_photos_per_taxon, len(remaining_photos)))
                for photo in selected_validation:
                    val_writer.writerow([taxon_id, photo])
                remaining_photos -= set(selected_validation)

            if taxon_id in test_taxons:
                selected_test = random.sample(list(remaining_photos), min(test_photos_per_taxon, len(remaining_photos)))
                for photo in selected_test:
                    test_writer.writerow([taxon_id, photo])

def main():
    parser = argparse.ArgumentParser(description="Process photo data and create training, validation, and test sets.")
    parser.add_argument("input_file", help="Input TSV file")
    parser.add_argument("--train_photos", type=int, required=True, help="Number of images per taxon for training")
    parser.add_argument("--val_taxons", type=int, required=True, help="Number of validation taxons")
    parser.add_argument("--test_taxons", type=int, required=True, help="Number of test taxons")
    parser.add_argument("--val_photos", type=int, required=True, help="Number of validation items per taxon")
    parser.add_argument("--test_photos", type=int, required=True, help="Number of test items per taxon")
    args = parser.parse_args()

    print("Selecting validation and test taxons...")
    validation_taxons = count_lines_and_select_taxons(args.input_file, args.val_taxons)
    test_taxons = count_lines_and_select_taxons(args.input_file, args.test_taxons)

    print("Processing data and writing output files...")
    process_data(args.input_file, "training_set.csv", "validation_set.csv", "test_set.csv",
                 args.train_photos, validation_taxons, test_taxons, args.val_photos, args.test_photos)
    print("Done!")

if __name__ == "__main__":
    main()
