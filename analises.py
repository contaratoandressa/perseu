# libraries
from pyspark.sql import SparkSession # dp py spark package for large databases
from pyspark.sql.functions import monotonically_increasing_id # data adjustment
import pandas as pd # analysis 
import numpy as np # analysis
from datetime import datetime # data manipulation

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

# obs.: in without views

# dataset
df = data_dwl()

# ETL
# check post_id duplicates (it is the id that informs the posts, and cannot be duplicated)

if (len(df[df.duplicated(['post_id'], keep=False)]) == 0):
    print('Dataset is Ok')
else:
    print('Dataset is not Ok')

# analisar missing values e ver se estes tem algum padrao

print('Find missing values')
print(df.isnull().sum())

print('Redes sociais: ')
print(np.unique(df.provider))

print('Metricas nas redes sociais: ')
print('in: likes ')
print(np.unique(df[df.provider == 'in'].likes))

print('in: views ')
print(np.unique(df[df.provider == 'in'].views))

print('in: comments ')
print(np.unique(df[df.provider == 'in'].comments))

print('in: engagement ')
print(np.unique(df[df.provider == 'in'].engagement))

print('-----')

print('tt: likes ')
print(np.unique(df[df.provider == 'tt'].likes))

print('tt: views ')
print(np.unique(df[df.provider == 'tt'].views))

print('tt: comments ')
print(np.unique(df[df.provider == 'tt'].comments))

print('tt: engagement ')
print(np.unique(df[df.provider == 'tt'].engagement))

print('-----')

print('yt: likes ')
print(np.unique(df[df.provider == 'yt'].likes))

print('yt: views ')
print(np.unique(df[df.provider == 'yt'].views))

print('yt: comments ')
print(np.unique(df[df.provider == 'yt'].comments))

print('yt: engagement ')
print(np.unique(df[df.provider == 'yt'].engagement))

# analisando os nulos em likes
print('Tamanho da base de nulos em likes: ')
print(df[df.likes == 'null'].shape)

print('Redes Sociais: ')
print(np.unique(df[df.likes == 'null'].provider)) # in and yt have this metric, then this is missing values only

print('Criadores: ')
print(np.unique(df[df.likes == 'null'].creator_id)) # We have a set of creators with missing values ​​in this metric, so there may be a pattern as well

print('Data: ')
print(np.unique(df[df.likes == 'null'].published_at)) # As the oldest date is '2023-04-18T00:01:04.000Z' almost 1 month later '2023-05-02T01:24:24.000Z' we had null values ​​in the likes metric, there may have been some change in the API to return missing values ​​in existing metrics for the analyzed social networks? A deeper inspection should be done to see these cases.

# analisando os nulos em views
print('Tamanho da base de nulos em views: ')
print(df[df.views == 'null'].shape)

print('Redes Sociais: ')
print(np.unique(df[df.views == 'null'].provider)) # in has not this metric

print('Criadores: ')
print(np.unique(df[df.views == 'null'].creator_id)) 
aux = df[df.views == 'null']
print(np.unique(aux[aux['creator_id'].isin(np.unique(df[df.views == 'null'].creator_id))].provider))

print('Data: ')
print(np.unique(df[df.views == 'null'].published_at)) 
print(np.unique(aux[aux['published_at'].isin(np.unique(df[df.views == 'null'].published_at))].provider))

# analisando os nulos em comments
print('Tamanho da base de nulos em comments: ')
print(df[df.comments == 'null'].shape)

print('Redes Sociais: ')
print(np.unique(df[df.comments == 'null'].provider)) # yt has this metric

print('Criadores: ')
print(np.unique(df[df.comments == 'null'].creator_id)) 

print('Data: ')
print(np.unique(df[df.comments == 'null'].published_at)) 

# analisando os nulos em engagement
print('Tamanho da base de nulos em engagement: ')
print(df[df.engagement == 'null'].shape)

print('Redes Sociais: ')
print(np.unique(df[df.engagement == 'null'].provider)) # yt has this metric

print('Criadores: ')
print(np.unique(df[df.engagement == 'null'].creator_id)) 

print('Data: ')
print(np.unique(df[df.engagement == 'null'].published_at)) 

#### Diagnosis
###### For some influencers, the likes metric does not appear in in and yt, but these have this metric. The views metric does not have in, so the missing values ​​are a result of this. Some creators are not having the comments metric on YouTube even though this network has such a metric. And, some content creators are lacking engagement on both in and yt, but these also have this metric. By definition, missing values ​​are different from zero values. Having zero in the metric means no reaction, having missing values ​​means something not filled in, possibly due to some data extraction and/or manipulation problem. Then, the str values ​​in the metrics will be passed to int and the 'null' field will be recognized as -1. Depending on the case, zero and null may be the same thing, but in other cases (like the one mentioned) they are not.

####### Other procedures could be the insertion of values ​​based on the mean or median of a given group.

####### DONDERS, A. Rogier T. et al. A gentle introduction to imputation of missing values. Journal of clinical epidemiology, v. 59, n. 10, p. 1087-1091, 2006.
####### LIN, Wei-Chao; TSAI, Chih-Fong. Missing value imputation: a review and analysis of the literature (2006–2017). Artificial Intelligence Review, v. 53, p. 1487-1509, 2020.

# 'null' to -1
df.likes = [-1 if x == 'null' else x for x in df.likes]
df.views = [-1 if x == 'null' else x for x in df.views]
df.comments = [-1 if x == 'null' else x for x in df.comments]
df.engagement = [-1 if x == 'null' else x for x in df.engagement]

# str to int
df.likes = [int(x) for x in df.likes]
df.views = [int(x) for x in df.views]
df.comments = [int(x) for x in df.comments]
df.engagement = [int(x) for x in df.engagement]

# summary
print(df[df.likes != -1].likes.describe())
print(df[df.views != -1].views.describe())
print(df[df.comments != -1].comments.describe())
print(df[df.engagement != -1].engagement.describe())

#### Q1:
##### Which creators have had grown the most regarding their engagement? If engagement is not available, chose another metric which may indicate audience interest in them. Create this analysis in an timeframe that feels suitable to the problem (e.g.: monthly for the last year; weekly over the last month; daily over the last weeks etc)

# create month and year
# As we can have a huge fluctuation over the days, compiling analyzes by month will make more sense so that we can compare between content creators, as they may or may not take actions at different times throughout the month to engage more or fewer followers.
df['Month'] = [int(x[5:7]) for x in df.published_at]
df['Year'] = [int(x[0:4]) for x in df.published_at]
df['Data'] = [x[0:7] for x in df.published_at]
# df['Data'] = [str(x) + '/' + str(y) for x in df.Month for y in df.Year]

# pivot table with Data and Creators
df_pivot = pd.DataFrame(df[df.engagement != -1].pivot_table(index='Data', columns='creator_id', values='engagement', aggfunc='mean'))

for col in df_pivot.columns:
  for i in range(0, df.shape[0]):
    if i == 0:
      df_pivot[col][i] = 0
    else:
      df_pivot[col][i] = ((df_pivot[col][i]/df_pivot[col][i-1])-1)*100

aux = df[df.creator_id == '1067571152'][['engagement', 'Data']]
np.mean(aux[aux.Data == '2023-05'])
