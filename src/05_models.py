"""
Predictive models for taxi trip efficiency.
"""

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

def train_linear_model(df, feature_cols):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol="earnings_per_min")
    pipeline = Pipeline(stages=[assembler, lr])
    return pipeline.fit(df)
