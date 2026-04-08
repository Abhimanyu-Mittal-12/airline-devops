from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from pyspark.ml import PipelineModel

# Initialize Spark for serving
spark = SparkSession.builder \
    .appName("Airline_Model_Serving") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Load the saved model
model_path = "airline_delay_model"
model = PipelineModel.load(model_path)

app = FastAPI(title="Airline Delay Prediction API")


# Define the expected JSON payload
class FlightData(BaseModel):
    Month: int
    DayofMonth: int
    Distance: float
    CRSDepTime: int
    CarrierCode: str
    Origin: str


@app.post("/predict")
def predict_delay(data: FlightData):
    # Convert input JSON to PySpark DataFrame
    schema = StructType([
        StructField("Month", IntegerType(), True),
        StructField("DayofMonth", IntegerType(), True),
        StructField("Distance", FloatType(), True),
        StructField("CRSDepTime", IntegerType(), True),
        StructField("CarrierCode", StringType(), True),
        StructField("Origin", StringType(), True)
    ])

    input_data = [(data.Month, data.DayofMonth, data.Distance, data.CRSDepTime, data.CarrierCode, data.Origin)]
    df = spark.createDataFrame(input_data, schema)

    # Run prediction
    prediction = model.transform(df)
    result = prediction.select("prediction").collect()[0][0]

    return {
        "prediction": "Delayed" if result == 1.0 else "On Time",
        "raw_prediction_value": result
    }