import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
import pickle
from sdv import SDV
from sdv import Metadata



df = pd.read_csv('data/long_individual_merged.csv')

def regression_train(df):

    y = df.iloc[:,4].values
    X = df.drop('eptime', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    X_train = normalize(X_train,norm='l2')
    X_test = normalize(X_test,norm='l2')

    regressor = RandomForestRegressor(n_estimators = 10, random_state = 42)
    regressor.fit(X_train, y_train)

    return regressor


regressor = regression_train(df)
pickle.dump(regressor, open("models/real_test_train.sav", 'wb'))


# load_lr_model =pickle.load(open("models/real_test_train.sav", 'rb'))
# y_load_predit=load_lr_model.predict(X_test)

def train_gen(data):
    metadata = Metadata()
    table = {'df': data}
    metadata.add_table(
        name = 'df',
        data = table['df']
    )
    sdv = SDV()
    sdv.fit(metadata, table)
    sdv.save('models/sdv.pkl')

def data_gen(data):
    sdv = SDV.load('models/sdv.pkl')
    samples = sdv.sample()
    return samples


train_gen(df)
generated_data = data_gen(df)

generated_data['df'].to_csv('data/generated_data.csv', index = False)

gen_data = pd.read_csv('data/generated_data.csv')

synth_data_model = regression_train(gen_data)
pickle.dump(regressor, open("models/gen_test_train.sav", 'wb'))
