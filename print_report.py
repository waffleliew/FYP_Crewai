import json

def print_reports(file_path):
    """
    Reads a JSON file, parses it, and prints the 'Report' field for each object.

    Args:
        file_path (str): The path to the JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        for item in data:
            if 'ECC' in item:
                print(f"--- Report for {item['ECC']} ---")
            if 'Report' in item:
                report_content = item['Report']
                print(report_content)
                print("\n")
            else:
                print("No 'Report' field found in this item.")

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    print_reports('/Users/raphaelliew/Documents/GitHub/FYP/Earnings2Insights_Result.json')