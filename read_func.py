import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import TimestampType, StringType, FloatType 
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import col
spark = SparkSession.builder.appName("BigdataProject3").getOrCreate()

from pyspark import sql, SparkConf, SparkContext
conf = SparkConf().setAppName("BigdataProject2")
sc = SparkContext.getOrCreate()
sqlContext = sql.SQLContext(sc)

import sys
import os



schema = StructType([ StructField("Timeanddate", StringType(), True),
                      StructField("I", StringType(), True),
                      StructField("II", StringType(), True),
                      StructField("III", StringType(), True),
                      StructField("AVR", StringType(), True),
                      StructField("AVL", StringType(), True),
                      StructField("AVF", StringType(), True),
                      StructField("V", StringType(), True),
                      StructField("MCL1", StringType(), True),
                      StructField("ABP", FloatType(), False),
                      StructField("PAP", StringType(), True),
                    ])

csv_2_df = spark.read.csv("hdfs://hadoop1:9000/CS5433/2020/Group_Project/sample.csv", header = 'true', schema=schema)
#csv_2_df=ssc.textFileStream(inputPath).schema(schema).option("maxFilesPerTrigger", 1)
#streamingDF = (spark.readStream.option("sep", ";").schema(schema).csv(inputPath))
#streamingDF.select("Timeanddate")
#streamingDF.groupBy("ABP").count
#streamingDF.groupBy("ABP").count()
#csv_2_df.show()
csv_2_df = csv_2_df.withColumn('index', f.monotonically_increasing_id())

#df.sort('index').limit(100)
for i in range(1,450001,3750):
    df=csv_2_df.where(col("index").between(i, i+3749))
    df.show()
    ik=df.filter(df.ABP>=60).count()
    print("value: {}".format(str((ik/3750)*100)))
   

