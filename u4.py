import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, to_date, concat_ws, round, when, broadcast

# 1. SETUP: Initialize Spark
spark = SparkSession.builder \
    .appName("Airline_Project_Unit4_Fixed") \
    .config("spark.driver.memory", "4g") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "file:///tmp/spark-events") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. EXTRACT: Load Both Datasets
print("Loading data...")

# Load the main Flight Data (Handle .shuffle extension by forcing csv format)
flights_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("airline.csv.shuffle")

# Load the Carrier Lookup Table
carriers_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("carriers.csv") \
    .withColumnRenamed("Code", "CarrierCode") \
    .withColumnRenamed("Description", "CarrierName")

# 3. TRANSFORM: Join & Clean (CRITICAL FIX FOR 'NA' VALUES)
print("Joining Flights with Carrier Names and Cleaning...")

# Join with Broadcast for speed
joined_df = flights_df.join(broadcast(carriers_df), flights_df.UniqueCarrier == carriers_df.CarrierCode, "left")

# Handle "NA" strings explicitly
clean_df = joined_df.withColumn(
    "FlightDate",
    to_date(concat_ws("-", col("Year"), col("Month"), col("DayofMonth")))
).withColumn(
    "DepDelay",
    # If value is "NA", make it NULL. Otherwise, cast to float.
    when(col("DepDelay") == "NA", None).otherwise(col("DepDelay")).cast("float")
).withColumn(
    "ArrDelay",
    when(col("ArrDelay") == "NA", None).otherwise(col("ArrDelay")).cast("float")
).na.drop(subset=["ArrDelay", "DepDelay"])

# 4. AGGREGATE: Create Outputs

# --- Unit II (R): Daily Trends ---
print("Generating Daily Stats...")
daily_stats = clean_df.groupBy("FlightDate").agg(
    round(avg("DepDelay"), 2).alias("AvgDepDelay"),
    round(avg("ArrDelay"), 2).alias("AvgArrDelay"),
    count("*").alias("TotalFlights")
).orderBy("FlightDate")

# --- Unit II (R): Carrier Comparison ---
print("Generating Carrier Stats...")
carrier_stats = clean_df.groupBy("CarrierName").agg(
    round(avg("ArrDelay"), 2).alias("AvgArrDelay"),
    count("*").alias("TotalFlights"),
    round((count(when(col("Cancelled") == 1, True)) / count("*")) * 100, 2).alias("CancelRate")
).orderBy(col("TotalFlights").desc())

# --- Unit III (PowerBI): Airport Stats ---
print("Generating Airport Stats...")
airport_stats = clean_df.groupBy("Origin").agg(
    count("*").alias("TotalDepartures"),
    round(avg("DepDelay"), 2).alias("AvgDepDelay")
).orderBy(col("TotalDepartures").desc())

# 5. LOAD: Save Files
output_path = "project_outputs"
print(f"Saving merged results to {output_path}...")

# Coalesce(1) is safe now because we are saving tiny aggregated summaries
daily_stats.coalesce(1).write.csv(f"{output_path}/daily_trends", header=True, mode="overwrite")
carrier_stats.coalesce(1).write.csv(f"{output_path}/carrier_stats_named", header=True, mode="overwrite")
airport_stats.coalesce(1).write.csv(f"{output_path}/airport_stats", header=True, mode="overwrite")
clean_df.write.csv(f"{output_path}/full_data", header=True, mode="overwrite")
print("SUCCESS! No more 'NA' errors.")
spark.stop()