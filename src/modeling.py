import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
import pickle
from sdv import SDV
from sdv import Metadata
import numpy as np
import pyreadstat as prs


############################################################################### UTILITY FUNCTIONS ######################################################################################

def data_load_and_col_select():
    df = pd.read_csv('data/long_individual_merged_left.csv')

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
    y = df.iloc[:,-1].values
    X = df.drop('eptime', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    X_train = normalize(X_train,norm='l2')
    X_test = normalize(X_test,norm='l2')

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

metadata = Metadata()
table = {'long': long_df, 'ind': ind_df}

metadata.add_table(
    name = 'long',
    data = table['long'],
    primary_key='uid'
)

metadata.add_table(
    name = 'ind',
    data = table['ind'],
    primary_key='uid'
)

file_name = 'sdv-sep'
train_gen(metadata, table, file_name)
generated_data = data_gen(file_name)

generated_data['long'].to_csv('data/long_gen.csv', index = False)
generated_data['ind'].to_csv('data/ind_gen.csv', index = False)


long_gen = pd.read_csv('data/long_gen.csv')
ind_gen = pd.read_csv('data/ind_gen.csv')


merged_df = merge_data(ind_gen, long_gen)

regressor = regression_train(df, 50)
pickle.dump(regressor, open("models/real_spss.pkl", 'wb'))


# load_lr_model =pickle.load(open("models/real_test_train.sav", 'rb'))
# y_load_predit=load_lr_model.predict(X_test)

# generated_data['df'].to_csv('data/generated_data_test.csv', index = False)

# gen_data = pd.read_csv('data/generated_data_test.csv')
# gen_data = pd.read_csv('data/long_individual_merged_short.csv')

synth_data_model = regression_train(merged_df, 50)
pickle.dump(regressor, open("models/gen_sep_spss.pkl", 'wb'))


############################################################################### Generating Merged Data ######################################################################################
