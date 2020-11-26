import pandas as pd
if __name__ == '__main__':
    df = pd.read_csv("post_process_d=50.csv",index_col=0)
    #post_process_h1_all_true_weights.csv
    # df = df.sort_values(['beta_xy'])
    print(df)
    # print(df.to_latex())