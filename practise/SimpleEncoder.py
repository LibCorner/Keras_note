#encode
from keras.layers import Dense,Input
from keras.models import Model
import numpy as np

'''
简单的单层编码器
'''

in_x=Input(shape=(6,))
#编码层
x=Dense(3,activation='tanh',name='encoder')(in_x)
#解码层
out=Dense(6,activation='tanh',name='decoder')(x)

model=Model(input=in_x,output=out)
model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])

#中间层的编码
encoder=Model(input=in_x,output=x)
encoder.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])

data=np.array([[0,0,0,1,1,1],
               [1,1,1,0,0,0]])

model.fit(data,data,nb_epoch=1000)

encoder.predict(data)


