import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = pd.read_csv("wvs.csv",index_col=0)
    df = df[df['satisfaction_life']>1]
    df = df[(df['age']>0)&(df['age']<90)]
    print(df.head())

    df_2 = df.groupby('age')['satisfaction_life','financial_happiness'].mean().reset_index()
    print(df_2.head())

    plt.scatter(df_2['age'],df_2['satisfaction_life'])
    plt.scatter(df_2['age'],df_2['financial_happiness'])
    plt.show()