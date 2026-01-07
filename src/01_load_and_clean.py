"""
Load and clean NYC taxi trip data.

This module computes trip duration and applies basic data quality filters.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# okay now here im gonna create some efficiency metrics

def build_df2(df: DataFrame) -> DataFrame:
    """
    Compute trip duration and apply basic cleaning rules.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Raw taxi trip dataframe.

    Returns
    -------
    pyspark.sql.DataFrame
        Cleaned dataframe with duration_min added.
    """

    # duration in minutes (pickup to dropoff)
    df2 = df.withColumn(
        "duration_min",
        (F.unix_timestamp("tpep_dropoff_datetime")
         - F.unix_timestamp("tpep_pickup_datetime")) / 60.0
    )

    # basic cleaning, like distance hance to be above 0, fare amounts have to be
    # within a reasonable range etc
    df2 = (
        df2
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("duration_min") > 0)  # was trip_duration_min
        .filter(F.col("total_amount").between(0, 500))
    )

    return df2


