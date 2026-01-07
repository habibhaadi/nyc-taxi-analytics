"""
Feature engineering for taxi trip efficiency metrics.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# some efficiency metrics

def add_efficiency_metrics(df2: DataFrame) -> DataFrame:
    df2 = (
        df2
        .withColumn("earnings_per_min", F.col("total_amount") / F.col("duration_min"))
        .withColumn("earnings_per_km", F.col("total_amount") / (F.col("trip_distance") * 1.60934))
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_dow", F.date_format("tpep_pickup_datetime","E"))
    )

    # just visually checking if the columns look correct
    df2.select("duration_min","earnings_per_min","earnings_per_km").limit(5).show(truncate=False)

    # another sanity check
    print("Approx sample size peek:", df2.limit(100000).count()) # bounded work

    return df2


def trim_outliers(df2: DataFrame) -> DataFrame:
    # get some summary statistics

    summary = (
        df2.select("earnings_per_min","earnings_per_km","trip_distance","duration_min","total_amount")
        .summary("count","mean","stddev","min","25%","50%","75%","max")
    )

    # NOTE (in my style): keeping it here so you can call it if you want the output, but not forcing it
    summary.show(truncate=False)

    # trim some obvious outliers
    df3 = (
        df2
        .filter((F.col("duration_min") >= 1) & (F.col("duration_min") <= 180)) # 1 min to 3 hours
        .filter((F.col("trip_distance") >= 0.1) & (F.col("trip_distance") <= 80)) # 0.1–80 miles
        .filter((F.col("total_amount") >= 1) & (F.col("total_amount") <= 300)) # $1–$300
        .filter((F.col("earnings_per_min") >= 0) & (F.col("earnings_per_min") <= 10))
        .filter((F.col("earnings_per_km") >= 0) & (F.col("earnings_per_km") <= 20))
    )

    # return the summary
    summary_trim = (
        df3.select("earnings_per_min","earnings_per_km","trip_distance","duration_min","total_amount")
        .summary("count","mean","stddev","min","25%","50%","75%","max")
    )
    summary_trim.show(truncate=False)

    # add columns pickup_hour and pick_dow
    df3 = (
        df3
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime")) 
        .withColumn("pickup_dow", F.date_format("tpep_pickup_datetime","E"))
    )

    return df3


def sample_for_plotting(df3: DataFrame, sample_frac: float = 0.01, hard_cap: int = 50000):
    # sample for plotting (keeps it quick)

    import numpy as np

    pdf = (
        df3
        .filter((F.col("earnings_per_min") >= 0) & (F.col("earnings_per_min") <= 10))
        .filter((F.col("earnings_per_km") >= 0) & (F.col("earnings_per_km") <= 20))
        .select("earnings_per_min","earnings_per_km","trip_distance")
        .sample(False, sample_frac, seed=42) # 1%
        .limit(hard_cap) # hard cap, keeps it snappy
        .toPandas()
    )

    # drop inf/nan just in case
    pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna()

    return pdf


def plot_efficiency_distributions(pdf):
    import matplotlib.pyplot as plt

    # $/min
    plt.hist(pdf["earnings_per_min"], bins=60, range=(0,10))
    plt.xlabel("Earnings per minute (USD)")
    plt.ylabel("Trips")
    plt.title("Distribution: $/min")
    plt.show()

    # $/km
    plt.hist(pdf["earnings_per_km"], bins=60, range=(0,20))
    plt.xlabel("Earnings per km (USD)")
    plt.ylabel("Trips")
    plt.title("Distribution: $/km")
    plt.show()
