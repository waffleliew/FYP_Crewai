import json

# File paths
md_file_path = '/Users/raphaelliew/Documents/GitHub/FYP/saved_reports/ABM_20250724_235030.md'
json_file_path = '/Users/raphaelliew/Documents/GitHub/FYP/Earnings2Insights_Result.json'

# Read the markdown file content
with open(md_file_path, 'r') as f:
    md_content = f.read()

# Read the JSON file
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

# Update the 'Report' field of the first item in the JSON data
# This assumes the JSON contains a list of objects
if isinstance(json_data, list) and len(json_data) > 0:
    json_data[0]['Report'] = md_content
else:
    # Handle other JSON structures if necessary, e.g., a single dictionary
    if isinstance(json_data, dict):
        json_data['Report'] = md_content

# Write the updated JSON data back to the file
with open(json_file_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"Successfully updated '{json_file_path}' with the content from '{md_file_path}'.")