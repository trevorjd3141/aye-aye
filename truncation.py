import pandas as pd

def truncate(processedPath, truncatedPath, documents=50000):
    df = pd.read_csv(processedPath)
    df.dropna(inplace=True)
    df = df.sample(n=documents)
    df.to_csv(truncatedPath, index=False)