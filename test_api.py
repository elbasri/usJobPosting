import requests
import os

# Define API URLs
TRAIN_URL = 'http://localhost:5000/train'
PREDICT_URL = 'http://localhost:5000/predict'
RESULT_URL = 'http://localhost:5000/results/'

# Paths to your CSV files
TRAIN_CSV = 'data/data_jobs.csv'
PREDICT_CSV = 'data/data_jobs.csv'
DOWNLOAD_DIR = 'downloads/'  # Directory to store downloaded results

# Ensure the downloads directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Test the /train endpoint
def test_train():
    with open(TRAIN_CSV, 'rb') as f:
        response = requests.post(TRAIN_URL, files={'file': f})
        print("Train Response:", response.json())

# Test the /predict endpoint
def test_predict():
    with open(PREDICT_CSV, 'rb') as f:
        response = requests.post(PREDICT_URL, files={'file': f})
        print("Predict Response:", response.json())
        
        # Get the result file name from the response
        result_file = response.json().get('result_file')
        
        # Test the /results endpoint to download the prediction results
        if result_file:
            result_filename = result_file.split('/')[-1]
            result_url = RESULT_URL + result_filename
            
            print(f"Downloading result from {result_url}")
            result_response = requests.get(result_url, stream=True)
            
            if result_response.status_code == 200:
                local_filename = os.path.join(DOWNLOAD_DIR, result_filename)
                with open(local_filename, 'wb') as f:
                    for chunk in result_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Result saved to {local_filename}")
            else:
                print(f"Error downloading result file: {result_response.status_code}")
        else:
            print("No result file found.")

if __name__ == '__main__':
    print("Starting training process...")
    #test_train()
    
    print("\nStarting prediction process...")
    test_predict()
