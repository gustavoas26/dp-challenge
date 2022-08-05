# %pip install azure-storage-blob

# %pip install elasticsearch==7.9.1

# %run ./document_processing

import pyspark.sql
from joblib import load
from datetime import datetime as dt
from azure.storage.blob import ContainerClient
from multiprocessing import Process
from pyspark.sql import functions as F

configs = {
    "container_name": "updates",
    "blob_key": "",
    "dp_output_path": "/gustavo/temp_patents/part_{}/",
    "dp_part_output_path": "/gustavo/temp_patents/part_{}_{}/",
    "max_rows": 3000,
    "counts_path": '/dbfs/gustavo/temp_counts/'
}

connection_string = configs['blob_key']
model_dir = '/dbfs/gustavo/trained_model/'
final_counts_dir = '/gustavo/word_counts/'
output_path = '/gustavo/energy_patents/'
dp_output_dir = '/gustavo/temp_patents/'
tp_counts_dir = '/gustavo/temp_counts/'

train_cols = ['l_0', 'l_1', 's_0', 's_1', 'c_0', 'c_1', 'c_2', 'sc_0', 'sc_1', 'sc_2', 'g_0', 'g_1', 'g_2', 'g_3',
              'sg_0', 'sg_1', 'sg_2', 'sg_3']

client = ContainerClient.from_connection_string(connection_string, container_name=configs['container_name'])
blob_list = client.list_blobs()
blob_names = [(i, blob.name) for i, blob in enumerate(blob_list)]

n_blobs = len(blob_names) // 5


def run_multiple_jobs(*funcs):
    """
    Executes multiple functions in parallel
    :param funcs:  functions to be executed
    """
    process = []
    for fun in funcs:
        p = Process(target=fun)
        p.start()
        process.append(p)
    for p in process:
        p.join()


def get_values(x: str) -> list:
    """
    return a list from a str list
    :param x: string list
    :return: list
    """
    x = x.replace('[', "").replace(']', "").replace(" ", "")
    return [item for item in x.split(',')]


def get_position(x: list, i: int):
    """
    Gets position i from list
    :param x: list
    :param i: index
    :return: element i of the list
    """
    return x[i]


def get_columns_values(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
    Transforms an str list to a list object
    :param df: input dataframe
    :return: Modified dataframe
    """
    get_val = F.udf(lambda x: get_values(x))
    df = df.withColumn('levels', get_val(F.col("levels")))
    df = df.withColumn('sections', get_val(F.col("sections")))
    df = df.withColumn('clasess', get_val(F.col("clasess")))
    df = df.withColumn('subclasess', get_val(F.col("subclasess")))
    df = df.withColumn('groups', get_val(F.col("groups")))
    df = df.withColumn('subgroups', get_val(F.col("subgroups")))
    return df


def encode_num(values: list, num_encoder: dict) -> list:
    """
    Custom label encoder
    :param num_encoder: label encoder
    :param values: values to encode
    :return: encoded value
    """
    return [num_encoder[v] if v in num_encoder.keys() else -1 for v in values]


def encode_cat(values: list, cat_encoder: dict) -> list:
    """
    Custom categorical encoder
    :param cat_encoder: categorical encoder
    :param values: values to encode
    :return: encoded value
    """
    return [cat_encoder[v] if v in cat_encoder.keys() else -1 for v in values]


def prepare_colums(df: pyspark.sql.DataFrame) -> tuple:
    """
    Prepare the dataframe columns to train the model
    :param df: train data
    :return: predict dataframe and patents dataframe
    """
    get_0 = F.udf(lambda x: get_position(x, 0))
    get_1 = F.udf(lambda x: get_position(x, 1))
    get_2 = F.udf(lambda x: get_position(x, 2))
    get_3 = F.udf(lambda x: get_position(x, 3))

    df = df.withColumn('l_0', get_0(F.col("levels")))
    df = df.withColumn('l_1', get_1(F.col("levels")))
    df = df.withColumn('s_0', get_0(F.col("sections")))
    df = df.withColumn('s_1', get_1(F.col("sections")))
    df = df.withColumn('c_0', get_0(F.col("clasess")))
    df = df.withColumn('c_1', get_1(F.col("clasess")))
    df = df.withColumn('c_2', get_2(F.col("clasess")))
    df = df.withColumn('sc_0', get_0(F.col("subclasess")))
    df = df.withColumn('sc_1', get_1(F.col("subclasess")))
    df = df.withColumn('sc_2', get_2(F.col("subclasess")))
    df = df.withColumn('g_0', get_0(F.col("groups")))
    df = df.withColumn('g_1', get_1(F.col("groups")))
    df = df.withColumn('g_2', get_2(F.col("groups")))
    df = df.withColumn('g_3', get_3(F.col("groups")))
    df = df.withColumn('sg_0', get_0(F.col("subgroups")))
    df = df.withColumn('sg_1', get_1(F.col("subgroups")))
    df = df.withColumn('sg_2', get_2(F.col("subgroups")))
    df = df.withColumn('sg_3', get_3(F.col("subgroups")))
    pred_cols = train_cols + ['row_id']
    df_new = df.select(*pred_cols)
    df = df.drop(*train_cols)

    return df_new, df


def append_words(final_words: dict, word_list: dict) -> dict:
    """
    Function that updates the word counts
    :param final_words: result dict
    :param word_list: list of word_counts of the patents
    """
    for word, count in word_list.items():
        try:
            final_words[word] = final_words[word] + count
        except KeyError:
            final_words[word] = count
    return final_words


# Create the different document processing objects, splitting the blob names
dp_job1 = DocumentProcessing(blob_names[0:n_blobs], 1, configs)
dp_job2 = DocumentProcessing(blob_names[1 * n_blobs:2 * n_blobs], 2, configs)
dp_job3 = DocumentProcessing(blob_names[2 * n_blobs:3 * n_blobs], 3, configs)
dp_job4 = DocumentProcessing(blob_names[3 * n_blobs:4 * n_blobs], 4, configs)
dp_job5 = DocumentProcessing(blob_names[4 * n_blobs:], 5, configs)

run_multiple_jobs(dp_job1.run(), dp_job2.run(), dp_job3.run(), dp_job4.run(), dp_job5.run())

final_word_counts = {}
temp_counts_path = ['/dbfs' + file.path.split(':')[1] for file in dbutils.fs.ls(tp_counts_dir)]

# Read temp word_counts and join them
for path in temp_counts_path:
    temp_count = load(path)
    final_word_counts = append_words(final_word_counts, temp_count)

# Get top 1000 words
final_word_counts = dict(sorted(final_word_counts.items(), key=lambda item: item[1], reverse=True))
final_word_counts.pop('', None)
n_1000 = final_word_counts.items()
n_1000 = list(n_1000)[:1000]
n_1000 = dict(n_1000)

# Save top 1000 words
dbutils.fs.mkdirs(final_counts_dir)
dump(n_1000, '/dbfs/' + final_counts_dir + 'word_count.json')

print("#############################################")
print("Document porcessing done.")
print("#############################################")

# Load model and dependencies
model = load(model_dir + 'random_f.joblib')
cat_encoder = load(model_dir + 'cat_encoder.joblib')
num_encoder = load(model_dir + 'num_encoder.joblib')

# Data to predict
patents_paths = [file.path for file in dbutils.fs.ls(dp_output_dir)]

n_encoder = F.udf(lambda x: encode_num(x, num_encoder))
c_encoder = F.udf(lambda x: encode_cat(x, cat_encoder))

ex_date = dt.today().strftime('%Y%m%d')

predictions_path = output_path + ex_date + '/'
dbutils.fs.mkdirs(predictions_path)

print("#############################################")
print("Starting prediction.")
print("#############################################")

for path in patents_paths:

    try:
        df_patents = spark.read.parquet(path)
    except Exception:
        print("Corrupted file {}".format(path))
        continue

    # Prepares data for prediction
    df_patents = get_columns_values(df_patents)
    df_patents = df_patents.withColumn('levels', c_encoder(F.col("levels")))
    df_patents = df_patents.withColumn('sections', c_encoder(F.col("sections")))
    df_patents = df_patents.withColumn('clasess', n_encoder(F.col("clasess")))
    df_patents = df_patents.withColumn('subclasess', c_encoder(F.col("subclasess")))
    df_patents = df_patents.withColumn('groups', n_encoder(F.col("groups")))
    df_patents = df_patents.withColumn('subgroups', n_encoder(F.col("subgroups")))
    df_patents = df_patents.withColumn("row_id", F.monotonically_increasing_id())
    df_patents = get_columns_values(df_patents)
    df_pred, df_patents = prepare_colums(df_patents)
    pd_df = df_pred.toPandas()
    pred_df = pd_df[train_cols]

    # Predict data
    pd_df['y_pred'] = list(model.predict(pred_df))

    # Filter patents for those related to energy
    pd_df = pd_df[pd_df.y_pred == 0]
    energy_ids = pd_df.row_id.values.tolist()
    df_energy = df_patents.filter(F.col("row_id").isin(energy_ids))
    df_energy = df_energy.select(F.col("title"), F.col("abstract"), F.col("year"))

    # Persist energy patents
    if df_energy.count() > 0:
        df_energy.write.parquet(predictions_path, mode="append", compression="snappy")
        # Save in ElasticSearch
        # df_energy.write.format("org.elasticsearch.spark.sql").option("es.resource", "http://10.2.0.4:9200/'challenge-gustavo'").save()

# Delete temp files
dbutils.fs.rm(tp_counts_dir, True)
dbutils.fs.rm(dp_output_dir, True)

print("#############################################")
print("Process done.")
print("#############################################")