# bibliotecas
from pyspark.sql import SparkSession # pacote dp py spark para grandes bases de dados
from pyspark.sql.functions import monotonically_increasing_id # ajuste dos dados
import pandas as pd
import numpy as np

# File location and type
file_location = "/FileStore/tables/desafio_posts_data-1.csv"
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
  .load(file_location).toPandas()

display(df)

# ajustando os nomes do dataframe 

# inicializando o sparksession
spark = SparkSession.builder \
    .appName("Rename Columns") \
    .getOrCreate()

# nomes das colunas
old = df.columns

# novos nomes
new = df.head()
new_names = new

# mapeando os nomes antigos e relacionando com os novos
names =  dict(zip(old, new_names))

# renomeando utillizando o mapeamento
df2 = df.toDF(*[names.get(col, col) for col in df.columns])

# encerrando a sessão
spark.stop()


# removendo a primeira linha
# criando uma coluna de id
df2 = df2.withColumn("id", monotonically_increasing_id())

# excluindo a primeira linha
df2 = df2.limit(df2.count() - 1)

# removendo a coluna de id
df2 = df2.drop("id")

# verificando se a troca foi realizada
display(df2)
