import pandas as pd

# csv
csvFile = "data/first_2000_jobs.csv"
df = pd.read_csv(csvFile)
pd.set_option('display.max_columns', None)

print(df.head(100))
