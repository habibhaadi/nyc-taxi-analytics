"""
Join NYC taxi trips with daily weather data.
"""

from pyspark.sql import functions as F

def join_weather(trips_df, weather_df):
    trips_df = trips_df.withColumn(
        "pickup_date", F.to_date("tpep_pickup_datetime")
    )

    weather_df = weather_df.withColumn(
        "date", F.to_date("DATE")
    )

    return trips_df.join(weather_df, trips_df.pickup_date == weather_df.date, "left")
