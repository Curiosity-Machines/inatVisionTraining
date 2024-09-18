import argparse
import pandas as pd
import os
import math
from multiprocessing import Process

def run_inference_on_chunk(chunk_file, gpu_id, script_path):
    """
    Run the inference script on a chunk of the CSV file, using the specified GPU.
    
    Args:
        chunk_file (str): Path to the CSV chunk file.
        gpu_id (int): The GPU ID to use for this process (0 or 1).
        script_path (str): Path to the inference script to run.
    """
    # Construct the command to run the inference script with the specified GPU and chunk file
    command = f"python {script_path} --input_csv {chunk_file} --gpu_id {gpu_id} --taxonomy taxonomy_c1.csv --model model.onnx"
    os.system(command)  # Execute the command

def split_csv(input_csv, num_chunks, output_dir):
    """
    Splits a CSV file into a specified number of chunks.
    
    Args:
        input_csv (str): Path to the input CSV file.
        num_chunks (int): Number of chunks to split the CSV into.
        output_dir (str): Directory to save the chunked CSV files.
    
    Returns:
        List[str]: List of paths to the chunked CSV files.
    """
    # Read the input CSV
    df = pd.read_csv(input_csv)
    
    # Calculate chunk size
    chunk_size = math.ceil(len(df) / num_chunks)
    
    chunk_files = []
    for i in range(num_chunks):
        chunk_df = df[i * chunk_size:(i + 1) * chunk_size]
        chunk_file = os.path.join(output_dir, f"chunk_{i}.csv")
        chunk_df.to_csv(chunk_file, index=False)
        chunk_files.append(chunk_file)
    
    return chunk_files

def main():
    parser = argparse.ArgumentParser(description="Split CSV and run inference in parallel using multiple GPUs.")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of processes to run (should be divisible by 2 for GPU balancing).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the split CSV chunks.')
    parser.add_argument('--script_path', type=str, required=True, help='Path to the inference script.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Ensure num_processes is even to balance GPU usage
    if args.num_processes % 2 != 0:
        raise ValueError("Number of processes must be divisible by 2 to ensure even GPU usage.")
    
    # Split the CSV file into num_processes chunks
    chunk_files = split_csv(args.input_csv, args.num_processes, args.output_dir)
    
    # Create processes and alternate between GPU 0 and GPU 1
    processes = []
    for i, chunk_file in enumerate(chunk_files):
        gpu_id = i % 2  # Alternate between GPU 0 and GPU 1
        process = Process(target=run_inference_on_chunk, args=(chunk_file, gpu_id, args.script_path))
        processes.append(process)
    
    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()

