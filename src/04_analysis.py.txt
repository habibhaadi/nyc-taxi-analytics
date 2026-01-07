"""
Exploratory and aggregate analysis for taxi efficiency.
"""

from pyspark.sql import functions as F

def earnings_by_hour(df):
    return (
        df.groupBy("pickup_hour")
          .agg(F.expr("percentile_approx(earnings_per_min, 0.5)").alias("p50_$per_min"),
               F.count("*").alias("trips"))
          .orderBy("pickup_hour")
    )
