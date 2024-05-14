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
  .option("quote", '"') \
  .option("escape", '"') \
  .option('ignoreLeadingWhiteSpace', True) \
  .load(file_location).toPandas()

display(df)

# verificando se a coluna de data contem somente data, pois estava havendo uma quebra no texto
contains_only_dates = all(pd.to_datetime(df['published_at'], errors='coerce').notnull())
print(f"The 'date_column' contains only date values: {contains_only_dates}")
