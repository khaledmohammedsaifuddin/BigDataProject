#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import mpld3
mpld3.enable_notebook()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
warnings.filterwarnings ("ignore")
i=300


# In[ ]:


def get_y_from_generator(gen):
    '''
    Get all targets y from a TimeseriesGenerator instance.
    '''
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        if y is None:
            y = batch_y
        else:
            y = np.append(y, batch_y)
    y = y.reshape((-1,1))
    #print(y.shape)
    return y


# In[ ]:


def binary_accuracy(a, b):
    '''
    Helper function to compute the match score of two
    binary numpy arrays.
    '''
    assert len(a) == len(b)
    return (a == b).sum() / len(a)


# In[ ]:


df = pd.read_csv('sample.csv', usecols=[0,9])
df.rename(columns = {"'Time and date'":'Time',"'ABP'":'ABP'}, inplace = True)
df['Time']=df['Time'].apply(lambda x: x.replace('[','').replace(']',''))
df.Time = pd.to_datetime(df.Time)
df=df.set_index('Time')


# In[ ]:


df=df[332000:333000]
def data_set(t1):
    t2=t1+200
    data=df[:t2]
    train  =data[:t1]
    test=data[t1:t2]
    scaler=MinMaxScaler()
    train=scaler.fit_transform(train)
    test= scaler.fit_transform(test)
    plt.plot(data)
    plt.ylabel('ABP')
    plt.xlabel('Time')
    plt.show()


    look_back=3
    train_data_gen = TimeseriesGenerator(train, train, length=look_back, sampling_rate=1,stride=1,batch_size=3)
    test_data_gen = TimeseriesGenerator(test, test,length=look_back, sampling_rate=1,stride=1,batch_size=1)


# In[105]:


    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))#one output value
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit_generator(train_data_gen, epochs=100,verbose=0)


# In[106]:


    model.evaluate_generator(test_data_gen)
    trainPredict = model.predict_generator(train_data_gen)
    #trainPredict.shape
    testPredict = model.predict_generator(test_data_gen)
    #testPredict.shape
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)



# In[108]:


    trainY = get_y_from_generator(train_data_gen)
    testY = get_y_from_generator(test_data_gen)
    trainY = scaler.inverse_transform(trainY)
    testY = scaler.inverse_transform(testY)


# In[109]:


    from sklearn.metrics import mean_squared_error
    import math
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))


# In[110]:


# shift train predictions for plotting
    trainPredictPlot = np.empty_like(data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # Delta + previous close
    trainPredictPlot = trainPredictPlot + data
    # set empty values
    # trainPredictPlot[0:look_back, :] = np.nan
    # trainPredictPlot[len(trainPredict)+look_back:, :] = np.nan


# In[111]:


# shift test predictions for plotting
    testPredictPlot = np.empty_like(data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2):len(data), :] = testPredict

    # Delta + previous close
    testPredictPlot = testPredictPlot + data
    # set empty values
    # testPredictPlot[0:len(trainPredict)+(look_back*2), :] = np.nan
    # testPredictPlot[len(dataset):, :] = np.nan


# In[120]:


# plot baseline and predictions
    fig = plt.figure(figsize=(10, 5))

    plt.plot(data+data-data,label='Real')
    plt.plot(trainPredictPlot-data,label='Train Predict')
    plt.plot(testPredictPlot-data,label="Test Predict")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("ABP")
    plt.show()



    plt.plot(data )
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

    #preparing data for confusion matrix and accuracy calculation
    flat_list1 = [item for sublist in testY for item in sublist]
    flat_list2 = [item for sublist in testPredict for item in sublist]
    s=len(flat_list1)
    w=10
    result1=[]
    result1_1=[]

    result2=[]
    result2_2=[]
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


    print (result1_1)
    print (result2_2)
    #confusion matrix and accuracy
    from sklearn.metrics import multilabel_confusion_matrix
    from sklearn.metrics import confusion_matrix
    
    arr1=np.array(result1)
    arr2=np.array(result2)
    a=multilabel_confusion_matrix(arr1,arr2, labels=[1,2,3])
    b=confusion_matrix(arr1,arr2, labels=[1,2,3])
    print (a)
    print (b)
    #SE=TP/(TP+FN)
    #SP=TN/(TN+FP)
    #AC=(TP+TN)/(TP+TN+FP+FN)
    #print ("SE: ",SE*100,"%")
    #print ("SP: ",SP*100,"%")
    #ravel print ("AC: ",AC*100,"%")
#this loop is used to change the train data size (t1,t2.....tn)


# In[ ]:


while True:
    data_set(i)
    i+=100
    if (i==900):
       break

