#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prompt files for VBench and EvalCrafter")
    parser.add_argument("--ec_path", type=str, required=True, help="Path to EvalCrafter directory")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Path to CSV file containing prompts")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp for output directory")
    return parser.parse_args()


def extract_filename(path):
    """Extract filename without extension from a video path."""
    # Extract the base filename
    basename = os.path.basename(path)
    # Remove the extension
    filename_without_ext = os.path.splitext(basename)[0]
    return filename_without_ext


def read_csv_prompts(csv_path):
    """Read prompts from CSV file and extract filenames and prompts."""
    prompts_dict = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) >= 2:  # Ensure row has at least 2 columns
                    path = row[0].strip()
                    prompt = row[1].strip()
                    
                    # Extract filename without extension
                    filename = extract_filename(path)
                    prompts_dict[filename] = prompt
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}
        
    return prompts_dict


def create_vbench_json(prompts_dict, ec_path):
    """Create prompt.json file for VBench."""
    vbench_prompts_dir = Path(ec_path) / "VBench/prompts"
    vbench_prompts_dir.mkdir(exist_ok=True, parents=True)
    
    json_path = vbench_prompts_dir / "prompt.json"
    
    try:
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(prompts_dict, json_file, indent=4, ensure_ascii=False)
        print(f"Successfully created VBench prompt.json at {json_path}")
    except Exception as e:
        print(f"Error creating VBench prompt.json: {e}")


def create_evalcrafter_prompts(prompts_dict, ec_path):
    """Create individual prompt text files for EvalCrafter."""
    ec_prompts_dir = Path(ec_path) / "EvalCrafter/prompts"
    ec_prompts_dir.mkdir(exist_ok=True, parents=True)
    
    for filename, prompt in prompts_dict.items():
        try:
            prompt_file_path = ec_prompts_dir / f"{filename}.txt"
            with open(prompt_file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(prompt)
            print(f"Successfully created {prompt_file_path}")
        except Exception as e:
            print(f"Error creating prompt file for {filename}: {e}")


def main():
    args = parse_args()
    
    # Check if the prompt CSV file exists
    if not os.path.isfile(args.prompt_dir):
        print(f"Error: CSV file {args.prompt_dir} not found.")
        return
    
    # Read prompts from CSV
    prompts_dict = read_csv_prompts(args.prompt_dir)
    
    if not prompts_dict:
        print("No prompts found in the CSV file or file format is incorrect.")
        return
    
    print(f"Found {len(prompts_dict)} prompt entries.")
    
    # Create VBench prompt.json
    create_vbench_json(prompts_dict, args.ec_path)
    
    # Create EvalCrafter prompt text files
    create_evalcrafter_prompts(prompts_dict, args.ec_path)
    
    print("Prompt generation completed.")


if __name__ == "__main__":
    main()
