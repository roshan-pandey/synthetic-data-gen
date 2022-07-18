import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import category_encoders as ce


long_pdf = pd.read_spss('data/uktus15_diary_ep_long.sav')
individual_pdf = pd.read_csv('data/uktus15_individual_new.csv')

long_cols = ['serial', 'pnum', 'DMFlag', 'dmonth', 'ddayw', 'WhereStart', 'WhereEnd', 'RushedD', 'KindOfDay', 'dia_wt_a', 'Trip', 'tid', 'Device', 'WhereWhen', 'whatdoing', 'eptime']
long_df = long_pdf.loc[:,long_cols]

long_df = long_df.dropna()

ind_cols = ['serial', 'pnum', 'DMSex', 'WorkSta', 'DVAge', 'Income', 'Sector', 'NumChild', 'NumAdult']
ind_df = individual_pdf.loc[:, ind_cols]

merged_df = ind_df.merge(long_df, on=['serial', 'pnum'], how='inner')
merged_df.drop(['serial', 'pnum'], axis=1, inplace=True)

merged_df = merged_df.dropna()

def count_encoding(df):
    data = pd.DataFrame()
    for i in df.columns:
        if df[i].dtype == 'category' or df[i].dtype == 'object':
            count_enc = ce.CountEncoder()
            # Transform the features, rename the columns with the _count suffix, and join to dataframe
            count_encoded = count_enc.fit_transform(df[i])
            if len(data.columns) < 1:
                data = df.join(count_encoded.add_suffix("_count"))
            else:
                data = data.join(count_encoded.add_suffix("_count"))
            data.drop(i, axis=1, inplace=True)
    return data

merged_df_ce = count_encoding(merged_df)

merged_df.to_csv('data/long_individual_merged.csv')