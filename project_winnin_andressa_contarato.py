# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# Depending on the version of jupyter notebook you insert, tensorflow and keras may not be loaded, make sure to use the most current version.

# Keras.
%pip install numpy scipy --upgrade
%pip install keras tensorflow --upgrade
dbutils.library.restartPython()

# COMMAND ----------

# Libraries.
from pyspark.sql import SparkSession # dp py spark package for large databases
from pyspark.sql.functions import monotonically_increasing_id # data adjustment
import pandas as pd # analysis 
import numpy as np # analysis
from datetime import datetime # data manipulation
# ! pip install tensorflow # is you did not import this library (pay attention!!)
import tensorflow
import keras
from tensorflow.keras.models import Sequential # lstm neural network
from tensorflow.keras.layers import LSTM, Dense # lstm neural network
from sklearn.preprocessing import MinMaxScaler # min max scale
# from keras.wrappers.scikit_learn import KerasRegressor
# multilevel model
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from PIL import Image
!pip install wordcloud
import wordcloud # text analysis
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # creat wordcloud
from collections import Counter # count char
import re # text analysis
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.decomposition import PCA # PCA
from sklearn.preprocessing import StandardScaler # PCA

# COMMAND ----------

# MAGIC %md
# MAGIC Import Dataset

# COMMAND ----------

# Functions.
def data_dwl(file_location = "/FileStore/tables/desafio_posts_data-1.csv"):

  """
  Download dataset.
  file_location = local of the dataset.

  """

  # File location and type.
  file_type = "csv"

  # CSV options.
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

  # verifying if the data column contains only data type.

  contains_only_dates = all(pd.to_datetime(df['published_at'], errors='coerce').notnull())
  print(f"The 'date_column' contains only date values: {contains_only_dates}")

  return(df)

# About the database.
# data from YouTube, Instagram, Facebook, Twitch and TikTok.
# 1 post = 1 video.
# metrics not available for a platform = 0.
# creator = who/where posts.

# Obs.: in without views.

# Dataset.
df = data_dwl()

# COMMAND ----------

# MAGIC %md
# MAGIC ETL

# COMMAND ----------

# Check post_id duplicates (it is the id that informs the posts, and cannot be duplicated).

if (len(df[df.duplicated(['post_id'], keep=False)]) == 0):
    print('Dataset is Ok')
else:
    print('Dataset is not Ok')

# Analyze missing values ​​and see if they have any pattern.

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

# Analyzing nulls in likes.
print('Tamanho da base de nulos em likes: ')
print(df[df.likes == 'null'].shape)

print('Redes Sociais: ')
print(np.unique(df[df.likes == 'null'].provider)) # in and yt have this metric, then this is missing values only.

print('Criadores: ')
print(np.unique(df[df.likes == 'null'].creator_id)) # We have a set of creators with missing values ​​in this metric, so there may be a pattern as well.

print('Data: ')
print(np.unique(df[df.likes == 'null'].published_at)) # As the oldest date is '2023-04-18T00:01:04.000Z' almost 1 month later '2023-05-02T01:24:24.000Z' we had null values ​​in the likes metric, there may have been some change in the API to return missing values ​​in existing metrics for the analyzed social networks? A deeper inspection should be done to see these cases.

# Analyzing nulls in views.
print('Tamanho da base de nulos em views: ')
print(df[df.views == 'null'].shape)

print('Redes Sociais: ')
print(np.unique(df[df.views == 'null'].provider)) # in has not this metric.

print('Criadores: ')
print(np.unique(df[df.views == 'null'].creator_id)) 
aux = df[df.views == 'null']
print(np.unique(aux[aux['creator_id'].isin(np.unique(df[df.views == 'null'].creator_id))].provider))

print('Data: ')
print(np.unique(df[df.views == 'null'].published_at)) 
print(np.unique(aux[aux['published_at'].isin(np.unique(df[df.views == 'null'].published_at))].provider))

# Analyzing nulls in comments.
print('Tamanho da base de nulos em comments: ')
print(df[df.comments == 'null'].shape)

print('Redes Sociais: ')
print(np.unique(df[df.comments == 'null'].provider)) # yt has this metric.

print('Criadores: ')
print(np.unique(df[df.comments == 'null'].creator_id)) 

print('Data: ')
print(np.unique(df[df.comments == 'null'].published_at)) 

# Analyzing nulls in engagement.
print('Tamanho da base de nulos em engagement: ')
print(df[df.engagement == 'null'].shape)

print('Redes Sociais: ')
print(np.unique(df[df.engagement == 'null'].provider)) # yt has this metric.

print('Criadores: ')
print(np.unique(df[df.engagement == 'null'].creator_id)) 

print('Data: ')
print(np.unique(df[df.engagement == 'null'].published_at)) 

#### Diagnosis

###### For some influencers, the likes metric does not appear in in and yt, but these have this metric. The views metric does not have in, so the missing values ​​are a result of this. Some creators are not having the comments metric on YouTube even though this network has such a metric. And, some content creators are lacking engagement on both in and yt, but these also have this metric. By definition, missing values ​​are different from zero values. Having zero in the metric means no reaction, having missing values ​​means something not filled in, possibly due to some data extraction and/or manipulation problem. Then, the str values ​​in the metrics will be passed to int and the 'null' field will be recognized as -1. Depending on the case, zero and null may be the same thing, but in other cases (like the one mentioned) they are not.

####### Other procedures could be the insertion of values ​​based on the mean or median of a given group.

####### DONDERS, A. Rogier T. et al. A gentle introduction to imputation of missing values. Journal of clinical epidemiology, v. 59, n. 10, p. 1087-1091, 2006.
####### LIN, Wei-Chao; TSAI, Chih-Fong. Missing value imputation: a review and analysis of the literature (2006–2017). Artificial Intelligence Review, v. 53, p. 1487-1509, 2020.

# 'null' to -1.
df.likes = [-1 if x == 'null' else x for x in df.likes]
df.views = [-1 if x == 'null' else x for x in df.views]
df.comments = [-1 if x == 'null' else x for x in df.comments]
df.engagement = [-1 if x == 'null' else x for x in df.engagement]

# str to int.
df.likes = [int(x) for x in df.likes]
df.views = [int(x) for x in df.views]
df.comments = [int(x) for x in df.comments]
df.engagement = [int(x) for x in df.engagement]

# Summary.
print(df[df.likes != -1].likes.describe())
print(df[df.views != -1].views.describe())
print(df[df.comments != -1].comments.describe())
print(df[df.engagement != -1].engagement.describe())

# As the values ​​are on a very large scale, to provide a better understanding of the 1M data. And then we multiply again.
df.likes = df.likes/1000000
df.views = df.views/1000000
df.comments = df.comments/1000000
df.engagement = df.engagement/1000000

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q1:
# MAGIC ##### Which creators have had grown the most regarding their engagement? If engagement is not available, chose another metric which may indicate audience interest in them. Create this analysis in an timeframe that feels suitable to the problem (e.g.: monthly for the last year; weekly over the last month; daily over the last weeks etc)
# MAGIC

# COMMAND ----------

# Create month and year.
# As we can have a huge fluctuation over the days, compiling analyzes by month will make more sense so that we can compare between content creators, as they may or may not take actions at different times throughout the month to engage more or fewer followers.
df['Month'] = [int(x[5:7]) for x in df.published_at]
df['Year'] = [int(x[0:4]) for x in df.published_at]
df['Data'] = [x[0:7] for x in df.published_at]
# df['Data'] = [str(x) + '/' + str(y) for x in df.Month for y in df.Year]

# Pivot table with Data and Creators.
df_pivot = pd.DataFrame(df[df.engagement != -1].pivot_table(index='Data', columns='creator_id', values='engagement', aggfunc='mean'))

for col in df_pivot.columns:
  for i in range(0, df.shape[1]):
    if i == 0:
      df_pivot[col][i] = df_pivot[col][i] 
    else:
      df_pivot[col][i] = ((df_pivot[col][i]/df_pivot[col][i-1])-1)*100
df.iloc[0,:] = np.repeat(0, df.shape[1])

aux = pd.DataFrame(df_pivot.iloc[12,:])
aux = aux.sort_values(by='2024-04', ascending=False, na_position='last')
aux = aux.iloc[0:10,]
print(aux)

# For each social media.

def top_10_sm(df, sm):
    
    """
    Download dataset.
    df = dataset.
    sm = social media.
    """
    df = df[df.provider == sm]

    # pivot table with Data and Creators
    df_pivot = pd.DataFrame(df[df.engagement != -1].pivot_table(index='Data', columns='creator_id', values='engagement', aggfunc='mean'))

    for col in df_pivot.columns:
        for i in range(0, df.shape[1]):
            if i == 0:
                df_pivot[col][i] = df_pivot[col][i] 
            else:
                df_pivot[col][i] = ((df_pivot[col][i]/df_pivot[col][i-1])-1)*100
    df.iloc[0,:] = np.repeat(0, df.shape[1])

    aux = pd.DataFrame(df_pivot.iloc[12,:])
    aux = aux.sort_values(by='2024-04', ascending=False, na_position='last')

    print(sm)
    return(aux.iloc[0:10,])

print(top_10_sm(df,  sm = 'yt'))
print(top_10_sm(df,  sm = 'in'))
print(top_10_sm(df,  sm = 'tt'))

###### The criteria for choosing the 10 content creators that grew the most over time took into account the monthly analysis and the percentage variation as a function of the engagement variable. Therefore, a content creator that could be small or medium or large if its variation in the last month did not increase so much, we would not consider it in the ranking. And, small content creators may have entered this top 10, but not large content creators, because we are only seeing the variation. Another approach would be to cluster (not just by network as was done) but by classes of content creators to then verify growth.

###### Top 10: UCyZ1Asau18OWyIXMYzBJF3g; UCV_4MsySAx5Ydu5qhMumFmw -100.000012; UCyQXHV_dnjjPxgfLokl8QbA; UCISQP4EGNyInNa8eTWhTRag; 6746644933858771973; UCs31L_n3LAVXRwiy8zmvEjg; UC_4H6b-rWYgeVkBJvzFZIiw; UCHvJlY9KEz-3NGrYe1MMGFw; UCDVqXmmSw0PTIt9AdeaYJOg; 144656320497917952.


# COMMAND ----------

# MAGIC %md
# MAGIC #### Q2
# MAGIC
# MAGIC ###### Based on your answer in 1, for these content creators, make a prediction of how they will be in the future.

# COMMAND ----------

# Return original data.
df.engagement = [x*1000000 for x in df.engagement] 
df.views = [x*1000000 for x in df.views] 
df.likes = [x*1000000 for x in df.likes] 
df.comments = [x*1000000 for x in df.comments] 

# Concatenate words.
summary = df.dropna(subset=['title'], axis=0)['title']
all_summary = " ".join(str(s) for s in summary)
# soptwords
# stopwords = set(STOPWORDS)
# stopwords.update(["da", "meu", "em", "você", "de", "ao", "os", "e", "não", "você", "vc", "aí", "ai", "com"])

reg = re.compile('\S{4,}')
c = Counter(ma.group() for ma in reg.finditer(all_summary))
words = pd.DataFrame.from_records(list(dict(c).items()), columns=['word','count'])
words = words.sort_values(by='count', ascending=False, na_position='last')
top_words = words.iloc[0:150,]

# Create potencial words according the recurrence.
df['PotencialWords'] = np.repeat(0, df.shape[0]) 
index = df[df.title.isin(top_words.word)].index

for i in index:
    df.PotencialWords[i] = 1

def forecast(data, name):    
    """
    Forecast.
    df = dataset with taget = engagement.
    """
    # Data normalization.
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # To dataframe.
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    # Split train and test.
    train_size = int(len(scaled_df) * 0.8)
    test_size = len(scaled_df) - train_size
    train_data, test_data = scaled_df.iloc[0:train_size], scaled_df.iloc[train_size:len(scaled_df)]

    # Adjust dataset for implement LSTM.
    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    # Steps define.
    time_steps = 10
    X_train, y_train = create_dataset(train_data[['likes', 'views', 'comments', 'PotencialWords']], train_data['engagement'], time_steps)
    X_test, y_test = create_dataset(test_data[['likes', 'views', 'comments', 'PotencialWords']], test_data['engagement'], time_steps)

    # LSTMs model.
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Models train.
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    # Metrics.
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Train Loss: {train_loss}, Test Loss: {test_loss}')

    # Make predictions with the model.
    y_pred = model.predict(X_test)

    # Plot.
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  
    plt.title('Valor Real vs Valor Previsto ' +name)
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Previsto')
    plt.grid(True)
    plt.show()

    # Save the plot.
    # plt.savefig(name+'.png')

    return(y_pred)

def multi_forecast(df, aux):
    
    """
    Forecast for each creator.
    df = dataset with taget = engagement.
    """

    # Adjust data for each creator.
    top_creators = df[df.creator_id.isin(aux.iloc[0:10,].index.tolist())]
    top_creators = top_creators[['Data', 'creator_id', 'likes', 'views', 'comments', 'PotencialWords', 'engagement']]
    creators = np.unique(top_creators.creator_id)

    for c in creators:
        data = top_creators[top_creators.creator_id == c]
        data = data[['likes', 'views', 'comments', 'PotencialWords', 'engagement']]
        print(c)
        data = forecast(data, c)

multi_forecast(df, aux)

######  Most of the data managed to fit the original line minimally, looking at the predicted versus original graphs. But, some improvements should be made such as increasing data for each content creator. The possibility of other variables to be inserted that could explain the effectiveness of content creators. Switching the activation function to compare performance and layer augmentation.

###### Another alternative would be to use a multilevel model to be able to infer performance based on these metrics over time and the performance of all financiers in question.

###### IDEA
"""
import statsmodels.api as sm

new_data = data.iloc[round(data.shape[0]*0.80):,:-1]
data = data.iloc[0:round(data.shape[0]*0.80),:]

md = sm.MixedLM.from_formula("engagement ~ likes + views + comments + PotencialWords", data, groups=data["creator_id"])
mdf = mixedlm_model.fit()
print(mdf.summary())
predictions = result.predict(new_data)
"""


# COMMAND ----------

# MAGIC %md
# MAGIC #### Q3
# MAGIC
# MAGIC ###### Build a function that takes a set of posts from a specific creator and explain the reasons for the growth/decline behavior of this creator. Apply it to the top 10 creators from questions 1 and 2.
# MAGIC

# COMMAND ----------

# Adjust data for each creator.
top_creators = df[df.creator_id.isin(aux.iloc[0:10,].index.tolist())]
np.unique(top_creators.provider) # Without in and its not necessary adjust the metric.
top_creators = top_creators[['Data', 'creator_id', 'likes', 'views', 'comments', 'title', 'engagement']]
creators = np.unique(top_creators.creator_id)

def pca_creator_enga(top_creators, creators, top_words):

    for c in creators:
        data = top_creators[top_creators.creator_id == c]
        print(c)
        # Pick up most used words and find in creator`s dataset.
        data['PotencialWords'] = np.repeat(0, data.shape[0]) 
        index = data[data.title.isin(top_words.word)].index
        
        for i in index:
            data.PotencialWords[i] = 1
            
        # Adjust dataset to apply PCA.
        data = data[['likes', 'views', 'comments', 'PotencialWords']] # 'engagement' is calculated from others metrics here

        # Data normalization.
        data.likes = (np.array(data.likes) - np.mean(np.array(data.likes)))/np.std(np.array(data.likes))
        data.views = (np.array(data.views) - np.mean(np.array(data.views)))/np.std(np.array(data.views))
        data.comments = (np.array(data.comments) - np.mean(np.array(data.comments)))/np.std(np.array(data.comments))
        # data.engagement = (np.array(data.engagement) - np.mean(np.array(data.engagement)))/np.std(np.array(data.engagement))

        # PCA.
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        final_df = pd.concat([data, pca_df], axis=1)

        plt.figure(figsize=(10, 6))
        plt.scatter(pca_df['PC1'], pca_df['PC2'])
        plt.title('PCAs results')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True)
        plt.show()

        var_exp = pca.explained_variance_ratio_
        print('Var Total Explained: ')
        print(var_exp)

        print('Loadings: ')
        print(pca.components_)

pca_creator_enga(top_creators, creators, top_words)

###### Search about lstm:
###### https://www.deeplearningbook.com.br/

###### By analyzing the main PCA components, we can see that the first component already explains more than 50% of all cases analyzed (content creators and their respective data). And, the metrics that most influence performance are likes and views. Therefore, their performance can improve as these two variables grow.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Improvements
# MAGIC
# MAGIC ###### Further organize the code and functions to generalize the processes.
# MAGIC
# MAGIC ###### Optimize.
# MAGIC
# MAGIC ###### Create project documentation.
# MAGIC
# MAGIC ###### Analyze other lstm structures.
# MAGIC
# MAGIC ###### Analyze other models to check performance such as the multilevel model.

# COMMAND ----------


