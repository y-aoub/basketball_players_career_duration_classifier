import requests
import json
from pathlib import Path

# URL for the API endpoint to make predictions
URL = "http://127.0.0.1:8000/predict/"


# Path to the input JSON file containing the data for prediction
IN_DATA_PATH = Path(__file__).parent / "data" / "test_request.json"

def read_input_data(file_path):
    """
    Read input data from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the input data.

    Returns:
        dict: The input data as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    # Read input data from the JSON file
    in_data = read_input_data(file_path=IN_DATA_PATH)

    # Send a POST request to the API endpoint with the input data
    response = requests.post(url=URL, json=in_data)

    # Print the JSON response from the API
    print(response.json())
    

