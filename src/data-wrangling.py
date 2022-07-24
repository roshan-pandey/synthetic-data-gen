import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import category_encoders as ce
import pyreadstat as prs

############################################################################### Count Encoding ######################################################################################

long_pdf = pd.read_spss('data/uktus15_diary_ep_long.sav')
individual_pdf = pd.read_csv('data/uktus15_individual_new.csv')

long_cols = ['serial', 'pnum', 'DMFlag', 'dmonth', 'ddayw', 'WhereStart', 'WhereEnd', 'RushedD', 'KindOfDay', 'dia_wt_a', 'Trip', 'tid', 'Device', 'WhereWhen', 'whatdoing', 'eptime']
long_df = long_pdf.loc[:,long_cols]

long_df = long_df.dropna()

ind_cols = ['serial', 'pnum', 'DMSex', 'WorkSta', 'DVAge', 'Income', 'Sector', 'NumChild', 'NumAdult']
ind_df = individual_pdf.loc[:, ind_cols]

merged_df = ind_df.merge(long_df, on=['serial', 'pnum'], how='inner')
# merged_df.drop(['serial', 'pnum'], axis=1, inplace=True)

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

merged_df_ce['serial'] = merged_df_ce['serial'].astype('int').astype('str')
merged_df_ce['pnum'] = merged_df_ce['pnum'].astype('int').astype('str')
merged_df_ce['uid'] = merged_df_ce['serial'] + merged_df_ce['pnum']
merged_df_ce = merged_df_ce.drop(['serial', 'pnum'], axis = 1)


merged_df_ce.to_csv('data/ce_merged.csv', index= False)

############################################################################### spss encoding ######################################################################################

def spss_encoder():
    long_df, long_meta = prs.read_sav('data/uktus15_diary_ep_long.sav', encoding="latin1")
    individual_df, individual_meta = prs.read_sav('data/uktus15_individual.sav', encoding="latin1")


    long_cols = ['serial', 'pnum', 'DMFlag', 'dmonth', 'ddayw', 'WhereStart', 'WhereEnd', 'RushedD', 'KindOfDay', 'dia_wt_a', 'Trip', 'tid', 'Device', 'WhereWhen', 'whatdoing', 'eptime']
    long_df = long_df.loc[:,long_cols]
    long_df = long_df.dropna()

    ind_cols = ['serial', 'pnum', 'DMSex', 'WorkSta', 'DVAge', 'Income', 'Sector', 'NumChild', 'NumAdult']
    ind_df = individual_df.loc[:, ind_cols]
    ind_df = ind_df.dropna()

    long_df['serial'] = long_df['serial'].astype('int').astype('str')
    long_df['pnum'] = long_df['pnum'].astype('int').astype('str')

    ind_df['serial'] = ind_df['serial'].astype('int').astype('str')
    ind_df['pnum'] = ind_df['pnum'].astype('int').astype('str')

    long_df['uid'] = long_df['serial'] + long_df['pnum']
    ind_df['uid'] = ind_df['serial'] + ind_df['pnum']

    long_df = long_df.drop(['serial', 'pnum'], axis = 1)
    ind_df = ind_df.drop(['serial', 'pnum'], axis = 1)

    merged_df = ind_df.merge(long_df, on='uid', how='inner')
    merged_df.drop(['uid'], axis=1, inplace=True)
    return merged_df

spss_merged_df = spss_encoder()
spss_merged_df.to_csv('data/spss_merged.csv', index= False)
