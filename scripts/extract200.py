import pandas as pd

csvFile = "data/first_2000_jobs.csv"

df = pd.read_csv(csvFile)

df_first_chunk = df.head(100)

csv_file_path = "data/first_100_jobs.csv"
df_first_chunk.to_csv(csv_file_path, index=False)

print(f"First 100 rows saved to {csv_file_path}")