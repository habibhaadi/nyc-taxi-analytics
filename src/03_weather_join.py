## bring in external dataset, which is gonna be the daily NYC weather (Central Park NOAA station, 2019)

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def load_weather(spark, weather_csv: str) -> DataFrame:
    weather = (
        spark.read.csv(weather_csv, header=True, inferSchema=True)
        .withColumn("date", F.to_date("DATE"))   # standardize date format
        .select("date", "TMAX", "TMIN", "PRCP", "AWND")
    )
    return weather


def join_weather(df3: DataFrame, weather: DataFrame) -> DataFrame:
    # add pickup_date to trips
    df3 = df3.withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))

    # join trips with weather by date
    df_weather = df3.join(weather, df3.pickup_date == weather.date, "left")

    # quick peek at merged taxi+weather data
    df_weather.select(
        "tpep_pickup_datetime", "trip_distance", "total_amount",
        "pickup_date", "TMAX", "TMIN", "PRCP", "AWND"
    ).show(5, truncate=False)

    return df_weather