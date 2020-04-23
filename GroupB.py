#!/usr/bin/env python
# coding: utf-8

# In[6]:




# In[7]:


import pyspark #importing pyspark
import pyspark.sql.functions as f #to access List of built-in functions available for DataFrame.
from pyspark.sql import SparkSession #creating spark session
from pyspark.sql.types import StructType, StructField #it will be used to create schema 
from pyspark.sql.types import TimestampType, StringType, FloatType# different data type for each column of csv file
from pyspark.streaming import StreamingContext# to create streaming context
from pyspark.sql.functions import col 
from pyspark import sql, SparkConf, SparkContext

spark = SparkSession.builder.appName("BigdataGroupProject").getOrCreate()#creating spark session
spark.sparkContext._conf.getAll()
conf = SparkConf().setAppName("BigdataGroupProject").setMaster("local[1]")#configuring spark
spark.sparkContext.stop()
sc = SparkContext(conf=conf)#creating instance for sparkcontext
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sqlContext = sql.SQLContext(sc)


# In[8]:



import numpy as np  #importing numpy 
import pandas as pd#importing pandas
import matplotlib.pyplot as plt#this library helps to plot graph
#matplotlib inline
#from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler#to scaling the data

#keras
from keras.preprocessing.sequence import TimeseriesGenerator#to convert data into timeseries 
from keras.models import Sequential# to do keras as a sequence of layers 
from keras.layers import Dense #output 
from keras.layers import LSTM# this one has used to do forecast
from keras.layers import Dropout #Dropout Regularization For Neural Networks
import warnings #to handle warnings 
warnings.filterwarnings ("ignore")


# In[9]:


# to Get all targets y from a TimeseriesGenerator instance.
    
def get_y_from_generator(gen):
    '''
    '''
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        if y is None:
            y = batch_y
        else:
            y = np.append(y, batch_y)
    y = y.reshape((-1,1))
    return y


# In[10]:


#below section is to create schema to read the csv file
schema = StructType([ StructField("Time", StringType(), True),
                      StructField("I", StringType(), True),
                      StructField("II", StringType(), True),
                      #StructField("III", StringType(), True),
                      #StructField("AVR", StringType(), True),
                      StructField("AVL", StringType(), True),
                      StructField("AVF", StringType(), True),
                      #StructField("V", StringType(), True),
                      StructField("MCL1", StringType(), True),
                      StructField("ABP", FloatType(), False),
                      StructField("PAP", StringType(), True),
                    ])

# Read from the csv file
csv_df = spark.read.csv("5_H1.csv", header = 'false', schema=schema)


# In[11]:


from pyspark.sql.functions import to_date #to maintain proper date-formate
csv_df=csv_df.filter(~csv_df.I.contains('I'))#filtering out unnecessary lines from csv
csv_df=csv_df.filter(~csv_df.I.contains('mV'))#filtering out unnecessary lines from csv
csv_df=csv_df.drop("I","II","AVL","AVF","MCL1","PAP")#drop out unnecessary columns

csv_df=csv_df.withColumn("ID",f.monotonically_increasing_id())#creating a column named ID primarily which will be used as an index
df=csv_df.toPandas() #converting data to pandas format
del df["ID"] #deleting ID columnn

df['Time']=df['Time'].apply(lambda x: x.replace('[','').replace(']','')) #preprocessing (removing unnecessary signs)
df.Time = pd.to_datetime(df.Time)# coverting to proper date-time formate
df=df.set_index('Time')#setting Time as an index


# In[12]:


df=df[:150000] #taking full dataset
i=(5*125*60)#setting the frist train data
#z=1


# In[79]:


def data_set(t1):

    t2=t1+(5*125*60)
    data=df[:t2]
    train  =data[:t1]#train data
    test=data[t1:t2]#test data
    scaler=MinMaxScaler()#creating scaler instance 
    train=scaler.fit_transform(train)#scaling train data
    test= scaler.fit_transform(test)#scaling test data
    
    #plt.plot(data)
    #plt.ylabel('ABP')
    #plt.xlabel('Time')
    #plt.show()

#below lines to convert time series data to sequences
    look_back=3
    train_data_gen = TimeseriesGenerator(train, train, length=look_back, sampling_rate=1,stride=1,batch_size=3)
    test_data_gen = TimeseriesGenerator(test, test,length=look_back, sampling_rate=1,stride=1,batch_size=1)


# In[105]:


    model = Sequential() #creating instance for sequential 
    model.add(LSTM(4, input_shape=(look_back, 1))) #define LSTM with 4 nuerons in hidden layer
    model.add(Dense(1))#output layer with one nueron
    model.compile(loss='mean_squared_error', optimizer='adam') #compiling with adam optimizer and mean squared error as loss function
    model.fit_generator(train_data_gen, epochs=100,verbose=0)#fitting the model


# In[106]:


    model.evaluate_generator(test_data_gen)#evaluating the model
    trainPredict = model.predict_generator(train_data_gen)#predicting train data
    #trainPredict.shape
    testPredict = model.predict_generator(test_data_gen)#predicting test data
    #testPredict.shape
    trainPredict = scaler.inverse_transform(trainPredict)#scaling out to return into normal format
    testPredict = scaler.inverse_transform(testPredict)#scaling out to return into normal format



# In[108]:


    trainY = get_y_from_generator(train_data_gen) #getting train data
    testY = get_y_from_generator(test_data_gen) #getting test data
    trainY = scaler.inverse_transform(trainY)#scaling out to return into normal format
    testY = scaler.inverse_transform(testY)#scaling out to return into normal format


# In[109]:


    from sklearn.metrics import mean_squared_error #to calculate mean squared error
    import math
    #trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
    #print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:,0]))# calculating square-root of mean squared error of test data
    print("\n")
    print('Test Score: %.2f RMSE' % (testScore))


# In[110]:


# shift train predictions for plotting
    trainPredictPlot = np.empty_like(data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    trainPredictPlot = trainPredictPlot + data
    

# In[111]:


# shift test predictions for plotting
    testPredictPlot = np.empty_like(data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2):len(data), :] = testPredict

    testPredictPlot = testPredictPlot + data
    

# In[120]:


# plot baseline and predictions
    fig = plt.figure(figsize=(10, 5))

    plt.plot(data+data-data,label='Real')
    plt.plot(trainPredictPlot-data,label='Train Predict')
    plt.plot(testPredictPlot-data,label="Test Predict")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("ABP")
    #plt.show()
    plt.savefig(str(t1)+'.png')



    #plt.plot(data )
    #plt.plot(trainPredictPlot)
    #plt.plot(testPredictPlot)
    #plt.show()

#preparing data for confusion matrix and accuracy calculation
    flat_list1 = [item for sublist in testY for item in sublist] #flattening the list
    flat_list2 = [item for sublist in testPredict for item in sublist]#flattening the list
    s=len(flat_list1)
    #w=20
    result1=[]
    result1_1=[]

    result2=[]
    result2_2=[]
    print("---------------------------******************----------------------")
       
    for w in range (10,30,10): #prediction window
        for i in range(0,s,w):
            list1=flat_list1[i:i+w]
            list2=flat_list2[i:i+w]

            d1=len([j1 for j1 in list1 if j1 >= 60])
            d2=len([j2 for j2 in list1 if j2 >= 60])
            e1=((d1/w)*100)
            e2=((d2/w)*100)
            if(len(list1)==w):
                if e1==100:
                    result1.append(1)
                    result1_1.append((1,"NORMAL",e1))
                if e1<=10:
                    result1.append(2)
                    result1_1.append((2,"AHE",e1))
                if(e1>10 and e1<100):
                    result1.append(3)
                    result1_1.append((3,"ALERT",e1))
                if e2==100:
                    result2.append(1)
                    result2_2.append((1,"NORMAL",e2))

                if e2<=10:
                    result2.append(2)
                    result2_2.append((2,"AHE",e2))
                if(e2>10 and e2<100):
                    result2.append(3)
                    result2_2.append((3,"ALERT",e2))
                    
        print ("Calculated Status based on Actual Test data for a fixed window 'W'")
        #print (result1_1)
        a1=0
        for m1 in result1_1:
            for n1 in m1:
                if n1=='AHE':
                    print("\033[45m {}\033[00m" .format(m1),end =" ")#color AHE with red
                    a1=1   
                    break
                if n1=='ALERT':
                    print("\033[44m {}\033[00m" .format(m1),end =" ")#color ALERT with YELLOW
                    a1=1   
                    break            
            if a1==0:
                print(m1,end =" ")
            a1=0
        print("\n")    
        print ("Number of AHE: ",result1.count(2))
        print ("Number of ALERT state: ",result1.count(3))
        print ("\n")
        print ("Calculated Status based on Predicted/Forecasted Test data for a fixed window 'W'")
        #print (result2_2)
        a2=0
        for m2 in result2_2:
            for n2 in m2:
                if n2=='AHE':
                    print("\033[45m {}\033[00m" .format(m2),end =" ")#color AHE with red
                    a2=1
                    break
                if n2=='ALERT':
                    print("\033[44m {}\033[00m" .format(m2),end =" ")#color ALERT with YELLOW
                    a2=1   
                    break     
            if a2==0:
                print(m2,end =" ")
            a2=0
        print("\n")
        print ("Number of AHE: ",result2.count(2))
        print ("Number of ALERT state: ",result2.count(3))
        
        #confusion matrix and accuracy
        from sklearn.metrics import multilabel_confusion_matrix
        from sklearn.metrics import confusion_matrix

        arr1=np.array(result1)
        arr2=np.array(result2)
        a=multilabel_confusion_matrix(arr1,arr2, labels=[1,2,3])
        b=confusion_matrix(arr1,arr2, labels=[1,2,3])
#calculating accuracy and other matrices
        from sklearn.metrics import accuracy_score
        c=accuracy_score(arr1,arr2)
        SE1=b[0][0]/(b[0][0]+b[0][1]+b[0][2])
        SE2=b[1][1]/(b[1][0]+b[1][1]+b[1][2])
        SE3=b[2][2]/(b[2][0]+b[2][1]+b[2][2])

        SP1=(b[1][1]+b[1][2]+b[2][1]+b[2][2])/(b[1][1]+b[1][2]+b[2][1]+b[2][2]+b[1][0]+b[2][0])
        SP2=(b[0][0]+b[0][2]+b[2][0]+b[2][2])/(b[0][0]+b[0][2]+b[2][0]+b[2][2]+b[0][1]+b[2][1])
        SP3=(b[0][0]+b[0][1]+b[1][0]+b[1][1])/(b[0][0]+b[0][1]+b[1][0]+b[1][1]+b[0][2]+b[1][2])

        #print ("\n")
        print ("Accuracy Measurement")
        print ("SE of NORMAL: ",SE1*100,"%")
        print ("SP of NORMAL: ",SP1*100,"%")
        print ("SE of AHE: ",SE2*100,"%")
        print ("SP of AHE: ",SP2*100,"%")
        print ("SE of ALERT: ",SE3*100,"%")
        print ("SP of ALERT: ",SP3*100,"%")

        print ("System accuracy: ",c*100,"%")
        #z=z+1
        del result1[:]
        del result1_1[:]

        del result2[:]
        del result2_2[:]
    


# In[14]:

#this loop is used to make train data adaptive
while True:
    data_set(i)
    i+=(5*125*60)
    if (i==150000):
       break


# In[ ]:




