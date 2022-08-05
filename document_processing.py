# %pip install azure-storage-blob

# %pip nltk

import io
import time
import nltk
import tarfile
import pyspark.sql
from nltk.corpus import stopwords
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from azure.storage.blob import ContainerClient
from collections import Counter
from joblib import dump


class DocumentProcessing:

    def __init__(self, blob_names: list, n_job: int, configs: dict):
        """
        Class that process the documents and writes temp files.
        :param blob_names: List of the blobs filenames
        :param n_job: Job number
        :param configs: Configs dict
        """
        self.configs = configs
        self.n_job = n_job
        self.blob_names = blob_names
        self.word_count = {}
        self.max_rows = self.configs['max_rows']
        self.connection_string = self.configs['blob_key']
        self.output_path = self.configs['dp_output_path']
        self.part_output_path = self.configs['dp_part_output_path']
        self.client = ContainerClient.from_connection_string(self.connection_string, container_name='updates')
        self.spark = SparkSession.builder.appName("Document-Processing {}".format(self.n_job)).getOrCreate()
        self.counts_path = configs['counts_path']

    def append_words(self, word_list: dict):
        """
        Function that updates the word counts
        :param word_list: list of word_counts of the patents
        """
        for word, count in word_list.items():
            try:
                self.word_count[word] = self.word_count[word] + count
            except KeyError:
                self.word_count[word] = count

    def extract_patents(self, blob_name: str) -> list:
        """
        Downloads the tar file from the blob, then unzip it and extracts the xml files as string
        :param blob_name:name of the blob folder that will be downloaded and uncompressed
        :return: list of tuples that contains the filename and th content as string
        """
        t0 = time.time()
        patents = []
        try:
            blob_client = self.client.get_blob_client(blob_name)
            raw_file = blob_client.download_blob().readall()
            tar = tarfile.open(fileobj=io.BytesIO(raw_file))
        except Exception as e:
            print("ERROR file: {}. {}".format(blob_name, e))
            return []
        files = tar.getnames()
        for xml_file in files:
            f = str(tar.extractfile(xml_file).read())
            patents.append((xml_file, f))
        tar.close()
        print("JOB: {}. TOTAL TIME: {}s. Total patents: {}".format(self.n_job, time.time() - t0, len(patents)))
        return patents

    def process_data(self, patents: list) -> pyspark.sql.DataFrame:
        """
        Function that transforms the string files into spark dataframes, then validates the data
        :param patents: list of patents
        :return: Dataframe with abstracts and titles.
        """

        def get_tokens(txt: str, stw: list) -> list:
            """
            Gets the tokens from the text ignoring the stopwords
            :param txt: raw text
            :param stw: stopwords list
            :return: text tokens
            """
            return [word for word in txt.split(' ') if word not in stw]

        def get_abstract(txt: str) -> str:
            """
            Gets the abstract from the patent
            :param txt: File as string
            :return: abstract
            """
            abstract = txt.split("<abstract")[1].split("<p")[1].split("</p>")[0].replace("<br/>", "").split(">")[
                1] if "<abstract" in txt else ""
            return abstract

        def get_title(txt: str) -> str:
            """
            Gets the title of the patent
            :param txt: File as string
            :return: title
            """
            title = txt.split("id=\"title_en\">")[1].split("</invention-title>")[0] if "id=\"title_en\">" in txt else ""
            return title

        def get_publication_year(txt: str) -> str:
            """
            Gets the publication year
            :param txt: File as string
            :return: publication year
            """
            year = txt.split("<document-id>")[1].split("<date>")[1][0:4] if "<document-id>" in txt else ""
            return year

        def clean_text(df: pyspark.sql.DataFrame, column: str) -> pyspark.sql.DataFrame:
            """
            Cleans the text from an specified column
            :param df: patents dataframe
            :param column: column name
            :return: cleaned dataframe
            """
            df = df.withColumn(column, F.regexp_replace(column, r"\n", " "))
            df = df.withColumn(column, F.regexp_replace(column, r"\r", " "))
            df = df.withColumn(column, F.regexp_replace(column, r"\\n", " "))
            df = df.withColumn(column, F.regexp_replace(column, "<br/>", ""))
            df = df.withColumn(column, F.regexp_replace(column, "\<!\[CDATA\[", ""))
            df = df.withColumn(column, F.regexp_replace(column, "\]\]\>", ""))
            df = df.withColumn(column, F.trim(F.col(column)))
            return df

        def detect_english(txt: str) -> bool:
            """
            Checks if the is written in english
            :param txt: text to be analise
            :return: true or false in case of the text is in english
            """
            try:
                txt.encode(encoding='utf-8').decode('ascii')
            except UnicodeDecodeError:
                return False
            return True

        def pad_sequence(features: list, max_len: int, t: str) -> list:
            """
            Add padding to the features
            :param features: list with train features
            :param max_len:  list max len
            :param t: type (str or int)
            :return: list with padding
            """
            pad = "0A" if t == "str" else -1
            features = features[1:]
            x = len(features)
            if x > max_len:
                features = features[0:max_len]
            else:
                extra = [pad] * (max_len - x)
                features = features + extra
            return features

        def get_classification(txt: str) -> list:
            """
            Gets all the classifications-ipc items from the patent
            :param txt: raw patent txt
            :return: list with all the classifications-ipc items
            """
            return txt.split("<classifications-ipcr>")[1].split("</classifications-ipcr>")[0].split(
                "</classification-ipcr>") if "</classifications-ipcr>" in txt else ""

        def get_tags(class_list: list, tag: str, max_len: int, t: str) -> list:
            """
            Gets all the values from a specified tag
            :param class_list: list with all the classifications-ipc items
            :param tag: name of the tag
            :param max_len:  list max len
            :param t: type (str or int)
            :return: list with all the values of the tag
            """
            raw_tag = list(
                set([txt.split("<" + tag + ">")[1].split("</" + tag + ">")[0] if "</" + tag + ">" in txt else "" for txt
                     in class_list]))
            return pad_sequence(raw_tag, max_len, t)

        extract_abstract = F.udf(lambda x: get_abstract(x), StringType())
        extract_title = F.udf(lambda x: get_title(x), StringType())
        extract_year = F.udf(lambda x: get_publication_year(x), StringType())
        english_text = F.udf(lambda x: detect_english(x), BooleanType())

        get_classifications = F.udf(lambda x: get_classification(x))
        get_levels = F.udf(lambda x: get_tags(x, "classification-level", 2, "str"))
        get_sections = F.udf(lambda x: get_tags(x, "section", 2, "str"))
        get_classes = F.udf(lambda x: get_tags(x, "class", 3, "int"))
        get_subclasses = F.udf(lambda x: get_tags(x, "subclass", 3, "str"))
        get_groups = F.udf(lambda x: get_tags(x, "main-group", 4, "int"))
        get_subgroups = F.udf(lambda x: get_tags(x, "subgroup", 4, "int"))

        df_patents = self.spark.createDataFrame(patents).toDF(*["file_name", "raw_txt"])
        del patents
        df_patents = df_patents.withColumn("abstract", extract_abstract(F.col("raw_txt")))
        df_patents = df_patents.withColumn("title", extract_title(F.col("raw_txt")))
        df_patents = df_patents.withColumn("year", extract_year(F.col("raw_txt")))
        df_patents = df_patents.withColumn("classification", get_classifications("raw_txt"))
        df_patents = df_patents.withColumn("levels", get_levels("classification"))
        df_patents = df_patents.withColumn("sections", get_sections("classification"))
        df_patents = df_patents.withColumn("clasess", get_classes("classification"))
        df_patents = df_patents.withColumn("subclasess", get_subclasses("classification"))
        df_patents = df_patents.withColumn("groups", get_groups("classification"))
        df_patents = df_patents.withColumn("subgroups", get_subgroups("classification"))

        df_patents = df_patents.drop("raw_txt", "classification", "file_name")

        df_patents = df_patents.where(F.col("abstract") != "").where(F.col("title") != "")
        df_patents = clean_text(df_patents, "abstract")
        df_patents = clean_text(df_patents, "title")
        df_patents = df_patents.withColumn("en_abs", english_text(F.col("abstract")))
        df_patents = df_patents.withColumn("en_title", english_text(F.col("title")))
        df_patents = df_patents.filter(F.col("en_abs") & F.col("en_title"))
        df_patents = df_patents.drop("en_abs", "en_title")

        stw = stopwords.words('english')
        stw = stw + ['THE', 'A', 'OF', 'TO', 'AND', 'IN', 'IS', 'AN', 'FOR', 'WITH', 'IF']
        tokens = F.udf(lambda x: get_tokens(x, stw))

        df_patents = df_patents.withColumn("conc", F.concat(F.col("abstract"), F.col("abstract")))
        df_patents = df_patents.withColumn("conc", F.regexp_replace(F.col("conc"), r"[\.,\[\]\(\)]", " "))
        df_patents = df_patents.withColumn("conc", F.trim(F.upper("conc")))
        df_patents = df_patents.withColumn("conc", F.trim(F.col("conc")))
        df_patents = df_patents.withColumn("conc_tokens", tokens(F.col("conc")))

        df_tokens = df_patents.select("conc_tokens")
        df_tokens = df_tokens.toPandas()

        df_patents.drop("conc", "conc_tokens")
        df_tokens['conc_tokens'] = df_tokens['conc_tokens'].apply(lambda x: x.replace(' ', ''))
        df_tokens['conc_tokens'] = df_tokens['conc_tokens'].apply(lambda x: x.replace('\"', ''))

        df_tokens['conc_tokens'] = df_tokens['conc_tokens'].apply(lambda x: x.replace(',', '\",\"'))
        df_tokens['conc_tokens'] = df_tokens['conc_tokens'].apply(lambda x: x.replace('[', '[\"'))
        df_tokens['conc_tokens'] = df_tokens['conc_tokens'].apply(lambda x: x.replace(']', '\"]'))
        df_tokens['conc_tokens'] = df_tokens['conc_tokens'].apply(lambda x: eval(x))

        tokens_values = [item for sublist in df_tokens.conc_tokens.values for item in sublist]

        aux_count = dict(Counter(tokens_values))
        self.append_words(aux_count)
        return df_patents

    def run(self):
        """
        Main function that process all the patents from the list, creates the Dataframe, and then writes the temporary
        results
        """
        dbutils.fs.mkdirs(self.counts_path)
        nltk.download('stopwords')

        i_last_folder, _ = self.blob_names[-1]
        for i, file_name in self.blob_names:
            print("JOB: {}. Processing folder: {}. {}/{}".format(self.n_job, file_name, i, i_last_folder))
            patents = self.extract_patents(file_name)

            if not patents:
                continue

            elif len(patents) > self.max_rows:
                for j in range(0, (len(patents) // self.max_rows) + 1):
                    chunk = patents[j * self.max_rows:(j + 1) * self.max_rows]
                    df_patents = self.process_data(chunk)
                    dbutils.fs.mkdirs(self.part_output_path.format(i, j))
                    df_patents.write.parquet(self.part_output_path.format(i, j), mode="overwrite", compression="snappy")
            else:
                df_patents = self.process_data(patents)
                dbutils.fs.mkdirs(self.output_path.format(i))
                df_patents.write.parquet(self.output_path.format(i), mode="overwrite", compression="snappy")

        self.word_count = dict(sorted(self.word_count.items(), key=lambda item: item[1]))
        words_path = self.counts_path.format(self.n_job)
        dump(self.word_count, words_path + 'word_count_{}.joblib'.format(self.n_job))