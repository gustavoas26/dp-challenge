# %pip install sklearn
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

warnings.filterwarnings('ignore')

train_path = '/gustavo/train_data/'
model_dir = '/dbfs/gustavo/trained_model/'
train_cols = ['l_0', 'l_1', 's_0', 's_1', 'c_0', 'c_1', 'c_2', 'sc_0', 'sc_1', 'sc_2', 'g_0', 'g_1', 'g_2', 'g_3',
              'sg_0', 'sg_1', 'sg_2', 'sg_3']

df = spark.read.parquet(train_path)
patents = df.toPandas()
patents = patents.sample(frac=1)
patents = patents.drop("file_name", axis=1)


def get_values(x):
    """
    return a list from an str list
    :param x: string list
    :return: list
    """
    x = x.replace('[', "").replace(']', "").replace(" ", "")
    return [item for item in x.split(',')]


patents['levels'] = patents['levels'].apply(lambda x: get_values(x))
patents['sections'] = patents['sections'].apply(lambda x: get_values(x))
patents['clasess'] = patents['clasess'].apply(lambda x: get_values(x))
patents['subclasess'] = patents['subclasess'].apply(lambda x: get_values(x))
patents['groups'] = patents['groups'].apply(lambda x: get_values(x))
patents['subgroups'] = patents['subgroups'].apply(lambda x: get_values(x))

cat_values = list(patents.levels.values) + list(patents.sections.values) + list(patents.subclasess.values)
cat_values = list(set([item for sublist in cat_values for item in sublist]))
cat_encoder = {k: v for v, k in enumerate(cat_values)}
num_values = list(patents.clasess.values) + list(patents.groups.values) + list(patents.subgroups.values)
num_values = list(set([item for sublist in num_values for item in sublist]))
num_encoder = {k: v for v, k in enumerate(num_values)}


def encode_num(values):
    """
    Custom label encoder
    :param values: values to encode
    :return: encoded value
    """
    return [num_encoder[v] if v in num_encoder.keys() else -1 for v in values]


def encode_cat(values):
    """
    Custom categorical encoder
    :param values: values to encode
    :return: encoded value
    """
    return [cat_encoder[v] if v in cat_encoder.keys() else -1 for v in values]


def prepare_colums(df):
    """
    Prepare the dataframe columns to train the model
    :param df: train data
    :return: new dataframe
    """
    df_new = pd.DataFrame()
    df_new['l_0'] = df['levels'].apply(lambda x: x[0])
    df_new['l_1'] = df['levels'].apply(lambda x: x[1])
    df_new['s_0'] = df['sections'].apply(lambda x: x[0])
    df_new['s_1'] = df['sections'].apply(lambda x: x[1])
    df_new['c_0'] = df['clasess'].apply(lambda x: x[0])
    df_new['c_1'] = df['clasess'].apply(lambda x: x[1])
    df_new['c_2'] = df['clasess'].apply(lambda x: x[2])
    df_new['sc_0'] = df['subclasess'].apply(lambda x: x[0])
    df_new['sc_1'] = df['subclasess'].apply(lambda x: x[1])
    df_new['sc_2'] = df['subclasess'].apply(lambda x: x[2])
    df_new['g_0'] = df['groups'].apply(lambda x: x[0])
    df_new['g_1'] = df['groups'].apply(lambda x: x[1])
    df_new['g_2'] = df['groups'].apply(lambda x: x[2])
    df_new['g_3'] = df['groups'].apply(lambda x: x[3])
    df_new['sg_0'] = df['subgroups'].apply(lambda x: x[0])
    df_new['sg_1'] = df['subgroups'].apply(lambda x: x[1])
    df_new['sg_2'] = df['subgroups'].apply(lambda x: x[2])
    df_new['sg_3'] = df['subgroups'].apply(lambda x: x[3])

    return df_new

# Prepare data
patents['levels'] = patents['levels'].apply(lambda x: encode_cat(x))
patents['sections'] = patents['sections'].apply(lambda x: encode_cat(x))
patents['clasess'] = patents['clasess'].apply(lambda x: encode_num(x))
patents['subclasess'] = patents['subclasess'].apply(lambda x: encode_cat(x))
patents['groups'] = patents['groups'].apply(lambda x: encode_num(x))
patents['subgroups'] = patents['subgroups'].apply(lambda x: encode_num(x))

train_data = prepare_colums(patents)

x_data = train_data[train_cols]
y_data = patents[['target']]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0)

rm_forest = RandomForestClassifier()
rm_forest.fit(x_train, y_train)

x_test['y_pred'] = list(rm_forest.predict(x_test))
x_test['target'] = y_test.target

print("Accuracy: {}".format(len(x_test[x_test.y_pred == x_test.target]) / len(x_test)))
dbutils.fs.mkdirs(model_dir)

# Save model and dependencies
dump(rm_forest, model_dir + 'random_f.joblib')
dump(cat_encoder, model_dir + 'cat_encoder.joblib')
dump(num_encoder, model_dir + 'num_encoder.joblib')
