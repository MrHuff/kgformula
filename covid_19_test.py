import pandas as pd
pd.options.display.max_rows = 100
pd.options.display.max_columns = 1000
if __name__ == '__main__':
    df = pd.read_csv("combined_dataset_latest.csv")
    print(df.shape)