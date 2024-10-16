import pandas as pd

# csv
csvFile = "data/first_2000_jobs.csv"
df = pd.read_csv(csvFile)
pd.set_option('display.max_columns', None)

print(df.head(100))


# Load the JSON file in chunks to avoid memory issues
df = pd.read_json(csvFile, lines=True, chunksize=100)

# Get the first 2000 rows
df_first_chunk = next(df)

# Save the first 2000 rows to a CSV file
csv_file_path = "data/first_100_jobs.csv"
df_first_chunk.to_csv(csv_file_path, index=False)

print(f"First 2000 rows saved to {csv_file_path}")
