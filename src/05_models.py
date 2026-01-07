from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


## alright i wanna bring it the NYC weather dataset

## im gonna start by comparing rainy vs dry days

def rain_vs_dry(df_weather: DataFrame) -> DataFrame:
    dfw = df_weather.withColumn("is_rain", (F.col("PRCP") >= 1.0).cast("boolean"))

    rain_vs_dry = (
        dfw.groupBy("is_rain")
           .agg(
               F.expr("percentile_approx(earnings_per_min,0.5)").alias("p50_$per_min"),
               F.expr("percentile_approx(trip_distance,0.5)").alias("p50_miles"),
               F.count("*").alias("trips")
           )
           .orderBy(F.desc("is_rain"))
    )

    rain_vs_dry.show(truncate=False)

    # bar graph is probably best for this, plotting median $/min for rainy vs dry days
    pdf_rain = rain_vs_dry.toPandas()
    labels = ["rain" if x else "dry" for x in pdf_rain["is_rain"]]

    import matplotlib.pyplot as plt
    plt.bar(labels, pdf_rain["p50_$per_min"])
    plt.ylabel("Median $/min")
    plt.title("Rain vs Dry — median $/min")
    plt.show()

    return rain_vs_dry


## then i can compare $/min by temperature too

def temp_bins_analysis(df_weather: DataFrame) -> DataFrame:
    dfw_t = df_weather.withColumn("tavg_c", (F.col("TMAX") + F.col("TMIN")) / 2.0)

    temp_bins = (
        dfw_t.select(
            F.when(F.col("tavg_c") < 0, "<0")
             .when(F.col("tavg_c") < 5, "0–5")
             .when(F.col("tavg_c") < 10, "5–10")
             .when(F.col("tavg_c") < 15, "10–15")
             .when(F.col("tavg_c") < 20, "15–20")
             .when(F.col("tavg_c") < 25, "20–25")
             .otherwise("25+").alias("temp_bin"),
            "earnings_per_min"
        )
        .groupBy("temp_bin")
        .agg(
            F.expr("percentile_approx(earnings_per_min,0.5)").alias("p50_$per_min"),
            F.count("*").alias("trips")
        )
    )

    # just to make the bins in order
    order = ["<0","0–5","5–10","10–15","15–20","20–25","25+"]
    temp_bins = (
        temp_bins
        .withColumn("ord", F.array_position(F.array([F.lit(x) for x in order]), F.col("temp_bin")))
        .orderBy("ord")
        .drop("ord")
    )

    temp_bins.show(truncate=False)

    # again bar plot is best
    pdf_tmp = temp_bins.toPandas()

    import matplotlib.pyplot as plt
    plt.bar(pdf_tmp["temp_bin"], pdf_tmp["p50_$per_min"])
    plt.xlabel("Daily avg temp (°C, binned)")
    plt.ylabel("Median $/min")
    plt.title("Median $/min by temperature bin")
    plt.show()

    return temp_bins


## light vs medium vs heavy rain

def rain_bins_analysis(df_weather: DataFrame) -> DataFrame:
    rb = (
        df_weather.select(
            F.when(F.col("PRCP") < 1.0, "0–1mm")
             .when(F.col("PRCP") < 5.0, "1–5mm")
             .when(F.col("PRCP") < 15.0, "5–15mm")
             .otherwise("15mm+").alias("rain_bin"),
            "earnings_per_min"
        )
        .groupBy("rain_bin")
        .agg(
            F.expr("percentile_approx(earnings_per_min,0.5)").alias("p50_$per_min"),
            F.count("*").alias("trips")
        )
        .orderBy("rain_bin")
    )

    rb.show(truncate=False)

    return rb


## okay now lets finish off with some ML, some deeper analysis before I get into report

## here im gonna predict trip efficiency ($/min) using certain trip features and the weather,

## using a linear regression model

def train_linear_regression(df_weather: DataFrame) -> Pipeline:
    # keep the useful fields and drop NAs
    ml_df = df_weather.select(
        "earnings_per_min", "trip_distance", "duration_min", "TMAX", "TMIN", "PRCP"
    ).dropna()

    feature_cols = ["trip_distance","duration_min","TMAX","TMIN","PRCP"]

    # assemble these features into vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # build a linear regression model
    lr = LinearRegression(featuresCol="features", labelCol="earnings_per_min")

    # pipeline = assembler + regression
    pipeline = Pipeline(stages=[assembler, lr])

    # fit the model
    model = pipeline.fit(ml_df)

    # get some summary stats
    lr_model = model.stages[-1]
    print("✅ Regression model trained")
    print("Coefficients:", lr_model.coefficients)
    print("Intercept:", lr_model.intercept)
    print("R^2:", lr_model.summary.r2)
    print("RMSE:", lr_model.summary.rootMeanSquaredError)

    # just have a look at some predictions
    preds = model.transform(ml_df).select("earnings_per_min","prediction","trip_distance","duration_min","PRCP")
    preds.show(10, truncate=False)

    return model


## then i can use logistic regression to classify high-efficieny trips

def train_logistic_regression(df_weather: DataFrame) -> Pipeline:
    # output 1 if $/min >= median, otherwise 0
    median_epm = df_weather.approxQuantile("earnings_per_min", [0.5], 0.01)[0]

    clf_df = (
        df_weather
          .withColumn("label", (F.col("earnings_per_min") >= F.lit(median_epm)).cast("int"))
          # simple engineered features
          .withColumn("rain_flag", (F.col("PRCP") > 0.1).cast("int"))
          .withColumn("tavg", (F.col("TMAX") + F.col("TMIN"))/2.0)
          .select("label","trip_distance","duration_min","tavg","PRCP","rain_flag","pickup_hour")
          .dropna()
    )

    # assemble the features
    feat_cols = ["trip_distance","duration_min","tavg","PRCP","rain_flag","pickup_hour"]
    assembler = VectorAssembler(inputCols=feat_cols, outputCol="features")

    # model
    logr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)

    pipe = Pipeline(stages=[assembler, logr])

    # conduct a train/testing split
    train_df, test_df = clf_df.randomSplit([0.8, 0.2], seed=42)
    clf_model = pipe.fit(train_df)

    # then evaluate innit
    pred = clf_model.transform(test_df)

    auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(pred)

    acc = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(pred)

    print(f"Logistic Regression trained. AUC = {auc:.3f} | Accuracy = {acc:.3f}")

    # have a look at the coefficients (thats important)
    lr_stage = clf_model.stages[-1]
    for name, coef in zip(feat_cols, lr_stage.coefficients):
        print(f"{name:15s}  coef = {coef:+.4f}")
    print("Intercept:", lr_stage.intercept)

    # and sus a few of the predictions
    pred.select("label","prediction","probability","trip_distance","duration_min","PRCP","pickup_hour").show(10, truncate=False)

    return clf_model


def run_weather_and_models(df_weather: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame, Pipeline, Pipeline]:
    # NOTE (in my style): this is just a wrapper so you can run everything in order if you want
    rvd = rain_vs_dry(df_weather)
    tb = temp_bins_analysis(df_weather)
    rb = rain_bins_analysis(df_weather)
    lr_model = train_linear_regression(df_weather)
    log_model = train_logistic_regression(df_weather)
    return rvd, tb, rb, lr_model, log_model
