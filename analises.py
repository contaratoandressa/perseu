# bibliotecas
from pyspark.sql import SparkSession # pacote dp py spark para grandes bases de dados
from pyspark.sql.functions import monotonically_increasing_id # ajuste dos dados
import pandas as pd
import numpy as np

def data_dwl(file_location = "/FileStore/tables/desafio_posts_data-1.csv"):

  """
  Download dataset.
  file_location = local of the dataset.

  """

  # File location and type
  file_type = "csv"

  # CSV options
  infer_schema = "false"
  first_row_is_header = "true"
  delimiter = ","

  # The applied options are for CSV files. For other file types, these will be ignored.
  df = spark.read.format(file_type) \
    .option("inferSchema", infer_schema) \
    .option("header", first_row_is_header) \
    .option("multiline", True) \
    .option("sep", delimiter) \
    .option("quote", '"') \
    .option("escape", '"') \
    .option('ignoreLeadingWhiteSpace', True) \
    .load(file_location).toPandas()

  display(df)

  # verifying if the data column contains only data type

  contains_only_dates = all(pd.to_datetime(df['published_at'], errors='coerce').notnull())
  print(f"The 'date_column' contains only date values: {contains_only_dates}")

  return(df)

# About the database
# data from YouTube, Instagram, Facebook, Twitch and TikTok
# 1 post = 1 video
# metrics not available for a platform = 0
# creator = who/where posts

# ETL
# check post_id duplicates (it is the id that informs the posts, and cannot be duplicated)

if (len(df[df.duplicated(['post_id'], keep=False)]) == 0):
    print('Dataset is Ok')
else:
    print('Dataset is not Ok')

# analisar missing values e ver se estes tem algum padrao

print('Find missing values')
print(df.isnull().sum())

# remove 'null'
df.replace('null', 0, inplace=True)

# summary
for col in ['likes','views','comments', 'engagement']:
    df[col] = [int(x) for x in df[col]]
    if np.min(df[col]) < 0:
        print('Problem with: '+col)
    if np.max(df[col]) > (np.percentile(df[col], 75) - 1.5*(np.percentile(df[col], 75) - np.percentile(df[col], 25)))
        print('Problem with outlier: '+col)
