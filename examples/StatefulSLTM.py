# -*- coding: utf-8 -*-
#StateFulLSTM
"""
Created on Thu May 26 08:40:34 2016

@author: dell
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM


# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 1
epochs = 100
# 预测后面lahead个值
lahead = 1



#生成输出,每个元素可以预测后面lahead个元素
def generate_xy(data):
    size=len(data)-lahead
    expected_output = np.zeros((size, 1))
    for i in range(len(data) - lahead):
        expected_output[i, 0] = np.mean(data[i + 1:i + lahead + 1])
    input_data=data[:size]
    return input_data,expected_output
    


print('Creating Model')
class StatefulLSTM():
    def __init__(self,batch_size=batch_size,tsteps=tsteps,input_dim=1):
        model = Sequential()
        model.add(LSTM(50,
                       batch_input_shape=(batch_size, tsteps, input_dim),
                       return_sequences=True,
                       stateful=True))
        model.add(LSTM(50,
                       batch_input_shape=(batch_size, tsteps, input_dim),
                       return_sequences=False,
                       stateful=True))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='rmsprop',metrics=['accuracy'])
        self.model=model
        self.batch_size=batch_size
    def fit(self,data,nb_epoch=5):
        x,y=generate_xy(data)
        
        for i in range(epochs):
            print('Epoch',i,'/',epochs)
            self.model.fit(x,y,batch_size=self.batch_size,verbose=1,nb_epoch=nb_epoch,shuffle=False)
            self.model.reset_states()
        scatter_plot(self.model,data,"SLTM")
            
    def predict(self,data,reset=True):
        if reset==True:
            self.model.reset_states()
        predicted_output = self.model.predict(data, batch_size=self.batch_size)
        return predicted_output
        
    def predict_sequence(self,data,reset=True,num=60):
        res=[]
        pre_out=self.predict(data,reset)
        cur=pre_out[len(pre_out)-1]
        cur=np.reshape(cur,(1,1,1))
        for i in range(60):
            cur=self.predict(cur,False)
            res.append(cur[0][0])
            cur=np.reshape(cur,(1,1,1))
        return res
    
def scatter_plot(model,data,legend="",batch_size=1):
    x,y=generate_xy(data)
    plt.scatter(range(len(x)),y,s=5)
    y_test=model.predict(x, batch_size=batch_size) 
    plt.plot(y_test,linewidth=2)
    plt.grid()
    plt.legend([legend],loc='upper left')
    plt.show()
    
    
data=[[[1]],[[2]],[[3]],[[4]],[[5]],[[6]]]
data=np.array(data)
def test():
    print('Generating Data')
    print('Input shape:', data.shape)
    
    model=StatefulLSTM()
    model.fit(data)
    
    return model
