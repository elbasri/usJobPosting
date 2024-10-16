import pandas as pd

# Define the file path for the JSON file
file_path = "/kaggle/input/us-job-postings-from-2023-05-05/techmap-jobs_us_2023-05-05.json"

# Load the JSON file in chunks to avoid memory issues
df = pd.read_json(file_path, lines=True, chunksize=2000)

# Get the first 2000 rows
df_first_chunk = next(df)

# Save the first 2000 rows to a CSV file
csv_file_path = "/kaggle/working/first_2000_jobs.csv"
df_first_chunk.to_csv(csv_file_path, index=False)

print(f"First 2000 rows saved to {csv_file_path}")