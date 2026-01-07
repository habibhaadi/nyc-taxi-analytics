from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# earnings by hour of day

def earnings_by_hour(df3: DataFrame) -> DataFrame:
    by_hour = (
        df3.groupBy("pickup_hour")
        .agg(
            F.expr("percentile_approx(earnings_per_min, 0.5)").alias("p50_$per_min"),
            F.expr("percentile_approx(earnings_per_min, 0.1)").alias("p10"),
            F.expr("percentile_approx(earnings_per_min, 0.9)").alias("p90"),
            F.count("*").alias("trips")
        )
        .orderBy("pickup_hour")
    )
    return by_hour


# earnings by day of week

def earnings_by_dow(df3: DataFrame) -> DataFrame:
    by_dow = (
        df3.groupBy("pickup_dow")
        .agg(
            F.expr("percentile_approx(earnings_per_min, 0.5)").alias("p50_$per_min"),
            F.count("*").alias("trips")
        )
        .orderBy(F.expr("array_position(array('Mon','Tue','Wed','Thu','Fri','Sat','Sun'), pickup_dow)"))
    )
    return by_dow


def show_time_summaries(df3: DataFrame) -> Tuple[DataFrame, DataFrame]:
    # earnings by hour of day
    by_hour = earnings_by_hour(df3)
    by_hour.show(24, truncate=False)

    # earnings by day of week
    by_dow = earnings_by_dow(df3)
    by_dow.show(truncate=False)

    return by_hour, by_dow


## here im gonna map some zone ids to human names, join to trips, and then rank all these
## pickup areas by $/min

def load_zones(spark, zones_csv: str) -> DataFrame:
    # read the TLC zone lookup
    zones = (
        spark.read.csv(zones_csv, header=True)
        .select(F.col("LocationID").cast("int").alias("LocationID"),
                "Borough",
                "Zone")
    )
    return zones


def attach_zone_names(df3: DataFrame, zones: DataFrame) -> DataFrame:
    # build two tables, for pickup and for dropoff
    pu = zones.select(F.col("LocationID").alias("PULocationID"),
                      F.col("Borough").alias("PUBorough"),
                      F.col("Zone").alias("PUZone"))

    do = zones.select(F.col("LocationID").alias("DOLocationID"),
                      F.col("Borough").alias("DOBorough"),
                      F.col("Zone").alias("DOZone"))

    # attach human-readable zones to every trip
    dfz = df3.join(pu, "PULocationID", "left").join(do, "DOLocationID", "left")
    return dfz


def top_pickup_zones(dfz: DataFrame) -> DataFrame:
    # rank these pickup locations by median earnings per minute
    top_pu = (
        dfz.groupBy("PUBorough", "PUZone")
        .agg(
            F.count("*").alias("trips"),
            F.expr("percentile_approx(earnings_per_min, 0.5)").alias("p50_$per_min")
        )
        .filter("trips >= 2000") # avoid sample that are too tiny
        .orderBy(F.desc("p50_$per_min"))
    )
    return top_pu


# bucket the trips by distance and see how the median $/min changes, with respect to trip length

def distance_bins(df3: DataFrame) -> DataFrame:
    dist_bins = (
        df3.select(
            F.when(F.col("trip_distance") < 1, "0–1")
            .when(F.col("trip_distance") < 3, "1–3")
            .when(F.col("trip_distance") < 5, "3–5")
            .when(F.col("trip_distance") < 10, "5–10")
            .otherwise("10+").alias("dist_bin"),
            "earnings_per_min"
        )
        .groupBy("dist_bin")
        .agg(
            F.expr("percentile_approx(earnings_per_min,0.5)").alias("p50_$per_min"),
            F.count("*").alias("trips")
        )
        .orderBy("dist_bin")
    )
    return dist_bins


def show_zone_and_distance_analysis(spark, df3: DataFrame, zones_csv: str) -> Tuple[DataFrame, DataFrame, DataFrame]:
    zones = load_zones(spark, zones_csv)
    dfz = attach_zone_names(df3, zones)

    top_pu = top_pickup_zones(dfz)
    top_pu.show(20, truncate=False)

    dist_bins = distance_bins(df3)
    dist_bins.show(truncate=False)

    return dfz, top_pu, dist_bins


## Earnings vs Time, I did sm time-based performance summaries

def plot_time_summaries(by_hour: DataFrame, by_dow: DataFrame) -> None:
    ## graph the above earnings vs time

    bh = by_hour.toPandas()
    bd = by_dow.toPandas()

    import matplotlib.pyplot as plt

    bh.plot(x="pickup_hour", y="p50_$per_min", kind="line", marker="o")
    plt.title("Median $/min by Hour"); plt.ylabel("$ per minute"); plt.xlabel("Hour of day (0–23)"); plt.show()

    bd.plot(x="pickup_dow", y="p50_$per_min", kind="bar")
    plt.title("Median $/min by Day"); plt.ylabel("$ per minute"); plt.xlabel("Day of week"); plt.show()


## hour x borough heatmap

def hour_borough_heatmap(dfz: DataFrame) -> None:
    hour_boro = (
        dfz.groupBy("pickup_hour", "PUBorough")
        .agg(F.expr("percentile_approx(earnings_per_min,0.5)").alias("p50"))
    )

    # each borough would be its own column
    pdf_hb = (
        hour_boro.toPandas()
        .pivot(index="pickup_hour", columns="PUBorough", values="p50")
        .sort_index()
    )

    import matplotlib.pyplot as plt

    plt.imshow(pdf_hb, aspect="auto") # simple heatmap
    plt.xticks(range(len(pdf_hb.columns)), pdf_hb.columns, rotation=45)
    plt.yticks(range(len(pdf_hb.index)), pdf_hb.index)
    plt.colorbar(label="Median $/min")
    plt.title("Median $/min by Hour × Borough")
    plt.xlabel("Borough"); plt.ylabel("Hour")
    plt.show()
