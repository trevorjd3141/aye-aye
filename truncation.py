import pandas as pd

def truncate(processed_path, truncated_path, count=50000):
    df = pd.read_csv(processed_path)
    df.dropna(inplace=True)
    df = df.sample(n=count)
    df.to_csv(truncated_path, index=False)