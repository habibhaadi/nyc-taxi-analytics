"""
Load and clean NYC taxi trip data using PySpark.
"""

from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("NYC Taxi Analytics").getOrCreate()

def load_and_clean(df):
    df = df.withColumn(
        "duration_min",
        (F.unix_timestamp("tpep_dropoff_datetime") -
         F.unix_timestamp("tpep_pickup_datetime")) / 60.0
    )

    df = (
        df.filter(F.col("trip_distance") > 0)
          .filter(F.col("duration_min") > 0)
          .filter(F.col("total_amount").between(0, 500))
    )

    return df
