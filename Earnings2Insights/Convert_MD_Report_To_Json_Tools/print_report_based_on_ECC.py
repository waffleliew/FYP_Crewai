import json

def print_report_by_ecc(file_path, target_ecc):
    """
    Reads a JSON file, parses it, and prints the 'Report' field for a specific ECC value.

    Args:
        file_path (str): The path to the JSON file.
        target_ecc (str): The ECC value to search for.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        found = False
        for item in data:
            if 'ECC' in item and item['ECC'] == target_ecc:
                found = True
                print(f"--- Report for {item['ECC']} ---")
                if 'Report' in item:
                    report_content = item['Report']
                    print(report_content)
                else:
                    print("No 'Report' field found for this ECC.")
                break
        
        if not found:
            print(f"No report found with ECC value: {target_ecc}")

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    target_ecc = input("Enter the ECC value to display: ")
    print_report_by_ecc('Earnings2Insights/Earnings2Insights_Result_final.json', target_ecc)