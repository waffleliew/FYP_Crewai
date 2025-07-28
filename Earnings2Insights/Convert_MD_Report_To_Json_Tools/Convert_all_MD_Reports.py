import json
import os

# File paths
reports_dir = 'Earnings2Insights/Generated_Reports'
json_file_path = 'Earnings2Insights/Earnings2Insights_Result_final.json'

# Function to extract ECC from filename
def extract_ecc_from_filename(filename):
    # Remove file extension
    base_name = os.path.splitext(filename)[0]
    return base_name

# Function to process a single markdown file
def process_md_file(file_path):
    # Extract filename for ECC
    filename = os.path.basename(file_path)
    ecc = extract_ecc_from_filename(filename)
    
    # Read the markdown file content
    with open(file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    return ecc, md_content

# Function to find all markdown files in a directory and its subdirectories
def find_md_files(directory):
    md_files = []
    
    # First process files in the main directory
    for file in sorted(os.listdir(directory)):
        if file.endswith('.md'):
            md_files.append(os.path.join(directory, file))
    
    # Then process files in subdirectories
    for subdir in sorted(os.listdir(directory)):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for file in sorted(os.listdir(subdir_path)):
                if file.endswith('.md'):
                    md_files.append(os.path.join(subdir_path, file))
    
    return md_files

# Main function
def main():
    # Find all markdown files
    md_files = find_md_files(reports_dir)
    
    # Create a new JSON array to store entries in the same order as files
    json_data = []
    
    # Process each markdown file and add to the JSON data
    for md_file in md_files:
        ecc, md_content = process_md_file(md_file)
        
        # Create new entry
        new_entry = {
            'ECC': ecc,
            'Report': md_content
        }
        json_data.append(new_entry)
    
    # Write the updated JSON data back to the file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Successfully updated '{json_file_path}' with content from {len(md_files)} markdown files.")

if __name__ == "__main__":
    main()