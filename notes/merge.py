#practise
from keras.layers import Dense,Input,merge
from keras.models import Model
import numpy as np

#输入
input_x=Input(shape=(6,))
x=Dense(3,activation='tanh')(input_x)
model=Model(input=input_x,output=x)
model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
#model作为一个shared层

#两个输入共享model层
input_a=Input(shape=(6,))
input_b=Input(shape=(6,))
out_a=model(input_a)
out_b=model(input_b)

#合并
merged=merge([out_a,out_b],mode='concat')
out=Dense(1,activation='sigmoid')(merged)

#创建Model
net=Model(input=[input_a,input_b],output=out)
net.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

data_a=np.array([[0,0,0,0,0,0],
                 [1,1,1,1,1,1]])
data_b=np.array([[1,1,1,1,1,1],
                 [0,0,0,0,0,0]])
labels=np.array([[0],[1]])

net.fit([data_a,data_b],labels,nb_epoch=100)

#共享层的输出
model.predict(data_a)