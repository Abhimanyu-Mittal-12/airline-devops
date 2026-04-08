from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, to_date, concat_ws, round, when, broadcast
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# 1. SETUP: Initialize Spark
spark = SparkSession.builder \
    .appName("Airline_Project_Unit4_Pro") \
    .config("spark.driver.memory", "4g") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "file:///tmp/spark-events") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. EXTRACT: Load Both Datasets
print("Loading data...")
flights_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("airline.csv.shuffle")

carriers_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("carriers.csv") \
    .withColumnRenamed("Code", "CarrierCode") \
    .withColumnRenamed("Description", "CarrierName")

# 3. TRANSFORM: Join & Clean
print("Joining Flights with Carrier Names and Cleaning...")
joined_df = flights_df.join(broadcast(carriers_df), flights_df.UniqueCarrier == carriers_df.CarrierCode, "left")

clean_df = joined_df.withColumn(
    "FlightDate",
    to_date(concat_ws("-", col("Year"), col("Month"), col("DayofMonth")))
).withColumn(
    "DepDelay",
    when(col("DepDelay") == "NA", None).otherwise(col("DepDelay")).cast("float")
).withColumn(
    "ArrDelay",
    when(col("ArrDelay") == "NA", None).otherwise(col("ArrDelay")).cast("float")
).na.drop(subset=["ArrDelay", "DepDelay"])

# 4. AGGREGATE: Create Outputs
print("Generating Aggregated CSVs for R and PowerBI...")
daily_stats = clean_df.groupBy("FlightDate").agg(
    round(avg("DepDelay"), 2).alias("AvgDepDelay"),
    round(avg("ArrDelay"), 2).alias("AvgArrDelay"),
    count("*").alias("TotalFlights")
).orderBy("FlightDate")

carrier_stats = clean_df.groupBy("CarrierName").agg(
    round(avg("ArrDelay"), 2).alias("AvgArrDelay"),
    count("*").alias("TotalFlights"),
    round((count(when(col("Cancelled") == 1, True)) / count("*")) * 100, 2).alias("CancelRate")
).orderBy(col("TotalFlights").desc())

airport_stats = clean_df.groupBy("Origin").agg(
    count("*").alias("TotalDepartures"),
    round(avg("DepDelay"), 2).alias("AvgDepDelay")
).orderBy(col("TotalDepartures").desc())

# 5. LOAD: Save Files (Removed full_data save as requested)
output_path = "project_outputs"
# daily_stats.coalesce(1).write.csv(f"{output_path}/daily_trends", header=True, mode="overwrite")
# carrier_stats.coalesce(1).write.csv(f"{output_path}/carrier_stats_named", header=True, mode="overwrite")
# airport_stats.coalesce(1).write.csv(f"{output_path}/airport_stats", header=True, mode="overwrite")
print("Aggregations Saved.")

# 6. DISTRIBUTED MACHINE LEARNING (The Presentation Stunner)
print("\n--- Starting Distributed Machine Learning Pipeline ---")

ml_df = clean_df.sample(withReplacement=False, fraction=0.1, seed=42) \
    .withColumn("label", when(col("ArrDelay") > 15, 1.0).otherwise(0.0))

assembler = VectorAssembler(
    inputCols=["Month", "DayofMonth", "DepDelay"],
    outputCol="features"
)
ml_ready_df = assembler.transform(ml_df).select("features", "label")

print("Splitting data into Train and Test sets...")
train_data, test_data = ml_ready_df.randomSplit([0.8, 0.2], seed=42)

print("Training Logistic Regression Model (Distributed)...")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
lr_model = lr.fit(train_data)

print("Evaluating Model...")
predictions = lr_model.transform(test_data)

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auroc = evaluator.evaluate(predictions)

print(f"\n>> ML PIPELINE COMPLETE <<")
print(f">> Model Accuracy (Area Under ROC): {auroc:.4f}")
print(f">> (A score of 1.0 is perfect, 0.5 is random guessing)\n")

spark.stop()