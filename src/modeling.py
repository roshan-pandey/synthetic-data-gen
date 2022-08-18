import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
import pickle
from sdv import SDV
from sdv import Metadata
import pyreadstat as prs
from sdv.tabular import CTGAN
############################################################################### UTILITY FUNCTIONS ######################################################################################

def data_load_and_col_select():
    df = pd.read_csv('data/spss_merged.csv')

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

    return df, long_df, ind_df

def regression_train(df, n_estimator):

    # df = df[df.select_dtypes(include=[np.number]).ge(0).all(1)]
    print(df.shape)
    y = df.loc[:,'eptime'].values
    X = df.drop('eptime', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    X_train = normalize(X_train, norm='l2')
    X_test = normalize(X_test, norm='l2')

    rf_regressor = RandomForestRegressor(n_estimators = n_estimator, random_state = 42)
    rf_regressor.fit(X_train, y_train)

    return rf_regressor

def train_gen(metadata, tables, name):
    sdv = SDV()
    sdv.fit(metadata, tables)
    sdv.save(f'models/{name}.pkl')

def data_gen(name):
    sdv = SDV.load(f'models/{name}.pkl')
    samples = sdv.sample()
    return samples

def merge_data(left, right):
    merged_df = left.merge(right, on='uid', how='inner')
    merged_df.drop(['uid'], axis=1, inplace=True)
    return merged_df


############################################################################### Generating Data Separately ######################################################################################

df, long_df, ind_df = data_load_and_col_select()

metadata_spss = Metadata()
table_spss = {'long': long_df, 'ind': ind_df}

metadata_spss.add_table(
    name = 'long',
    data = table_spss['long'],
    primary_key='uid'
)

metadata_spss.add_table(
    name = 'ind',
    data = table_spss['ind'],
    primary_key='uid'
)

file_name = 'sdv-sep'
train_gen(metadata_spss, table_spss, file_name)
generated_data_spss = data_gen(file_name)

generated_data_spss['long'].to_csv('data/long_gen.csv', index = False)
generated_data_spss['ind'].to_csv('data/ind_gen.csv', index = False)


long_gen = pd.read_csv('data/long_gen.csv')
ind_gen = pd.read_csv('data/ind_gen.csv')


merged_df_spss = merge_data(ind_gen, long_gen)
merged_df_spss.to_csv('data/gen_spss_sep.csv', index = False)

gen_sep_spss = regression_train(merged_df_spss, 50)
pickle.dump(gen_sep_spss, open("models/gen_sep_spss.pkl", 'wb'))


regressor_spss = regression_train(df, 50)
pickle.dump(regressor_spss, open("models/real_spss.pkl", 'wb'))

#################################################### CE #################################################

long_ce_df = pd.read_csv('data/long_ce.csv')
ind_ce_df = pd.read_csv('data/ind_ce.csv')

merged_ce_df = merge_data(ind_ce_df, long_ce_df)

regressor = regression_train(merged_ce_df, 50)
pickle.dump(regressor, open("models/real_ce.pkl", 'wb'))

metadata_ce_sep = Metadata()
table_ce_sep = {'long': long_ce_df, 'ind': ind_ce_df}

metadata_ce_sep.add_table(
    name = 'long',
    data = table_ce_sep['long'],
    primary_key='uid'
)

metadata_ce_sep.add_table(
    name = 'ind',
    data = table_ce_sep['ind'],
    primary_key='uid'
)

file_name = 'sdv-sep_ce'
train_gen(metadata_ce_sep, table_ce_sep, file_name)
generated_data_ce = data_gen(file_name)

generated_data_ce['long'].to_csv('data/long_gen_ce.csv', index = False)
generated_data_ce['ind'].to_csv('data/ind_gen_ce.csv', index = False)

long_gen_ce = pd.read_csv('data/long_gen_ce.csv')
ind_gen_ce = pd.read_csv('data/ind_gen_ce.csv')

merged_df_ce = merge_data(ind_gen_ce, long_gen_ce)

merged_df_ce.to_csv('data/gen_ce_sep.csv', index = False)

gen_sep_ce_model = regression_train(merged_df_ce, 50)
pickle.dump(gen_sep_ce_model, open("models/gen_sep_ce_model.pkl", 'wb'))

############################################################################### Generating Merged Data ######################################################################################

ce_merged = pd.read_csv('data\ce_merged.csv')
spss_merged = pd.read_csv('data\spss_merged.csv')

metadata_ce = Metadata()
table_ce = {'ce': ce_merged}
metadata_ce.add_table(
    name = 'ce',
    data = table_ce['ce']
)

file_name = 'sdv-ce-merged'
train_gen(metadata_ce, table_ce, file_name)
generated_data_ce_merged = data_gen(file_name)

generated_data_ce_merged['ce'].to_csv('data/gen_ce.csv', index = False)

regressor_ce_merged = regression_train(generated_data_ce_merged['ce'], 50)
pickle.dump(regressor_ce_merged, open("models/gen_ce_model.pkl", 'wb'))


metadata_spss = Metadata()
table_spss = {'spss': spss_merged}
metadata_spss.add_table(
    name = 'spss',
    data = table_spss['spss']
)

file_name = 'sdv-spss-merged'
train_gen(metadata_spss, table_spss, file_name)
generated_data_spss_merged = data_gen(file_name)


generated_data_spss_merged['spss'].to_csv('data/gen_spss.csv', index = False)
regressor_spss_merged = regression_train(generated_data_spss_merged['spss'], 50)
pickle.dump(regressor_spss_merged, open("models/gen_spss_model.pkl", 'wb'))
