import pandas as pd

# csv
csvFile = "data/first_2000_jobs.csv"
df = pd.read_csv(csvFile)

print(df.head())