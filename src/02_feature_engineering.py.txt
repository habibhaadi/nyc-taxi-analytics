"""
Feature engineering for taxi trip efficiency metrics.
"""

from pyspark.sql import functions as F

def add_efficiency_features(df):
    df = (
        df.withColumn("earnings_per_min", F.col("total_amount") / F.col("duration_min"))
          .withColumn("earnings_per_km", F.col("total_amount") / (F.col("trip_distance") * 1.60934))
          .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
          .withColumn("pickup_dow", F.date_format("tpep_pickup_datetime", "E"))
    )

    return df
