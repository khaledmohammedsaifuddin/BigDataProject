from pyspark import SparkContext
from pyspark.sql import SparkSession 
from pyspark.sql.types import StructType

spark = SparkSession.builder.appName ("SamplelineCount").getOrCreate()

# Define schema of the csv
userSchema = StructType().add("Time and date", "string").add("I", "string").add("II", "string"). add("III", "string").add("AVR", "string").add("AVL", "string").add("AVF","string").add("V","string").add("MCL1","string").add("ABP","string").add("PAP","string")

#Read CSV files from set path
dFCSV = spark.readStream.option("sep", ",").option("maxFilesPerTrigger", 1).schema(userSchema).csv("hdfs://hadoop1:9000/CS5433/2020/Group_Project")

dFCSV.createOrReplaceTempView("salary")
totalSalary = spark.sql("select ABP from salary where ABP not like '%ABP%' and ABP not like '%mmHg%'")
#totalSalary.show(100,truncate=False)
query = totalSalary.writeStream.outputMode("update").format("console").trigger(processingTime="2 seconds").start()
query.awaitTermination()

