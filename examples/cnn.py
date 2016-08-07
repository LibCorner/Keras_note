#cnn

from keras.layers import Convolution1D,Convolution2D,Dense,Flatten,MaxPooling1D
from keras.models import Sequential,Model
import numpy as np


model_1=Sequential()
model_1.add(Convolution1D(64,3,border_mode='same',input_shape=(10,32)))  #output.shpe=(None,10 ,64)

model_1.add(Convolution1D(32,3,border_mode='same'))

model_1.compile(optimizer='sgd',loss='mse')

data=np.random.random((1,10,32))  #input.shape=(samples,timesteps, input_dim)
labels=np.random.randint(2,size=(1,10,32))  #labels.shape=(samples, tiemsteps, output_dim)

#model_1.fit(data,labels)

#Conv2D
model_2=Sequential()
#input_shape=(chanals,width,height), not inclue sample axis
#output.shape=(64,256,256)
model_2.add(Convolution2D(64,3,3,border_mode='same',input_shape=(3,256,256))) 

model_2.add(Convolution2D(8,3,3,border_mode='same'))
model_2.add(Flatten())
model_2.add(Dense(32))
model_2.compile(optimizer='sgd',loss='mse')

data=np.random.random((1,3,256,256))  #input.shape=(samples,chanals,width,height)
labels=np.random.randint(2,size=(1,32))  

model_2.fit(data,labels)

#maxpooling
model_1=Sequential()
model_1.add(Convolution1D(64,3,border_mode='same',input_shape=(10,32)))  #output.shpe=(None,10 ,64)

model_1.add(Convolution1D(32,3,border_mode='same'))
model_1.add(MaxPooling1D(10))

model_1.compile(optimizer='sgd',loss='mse')

data=np.random.random((1,10,32))  #input.shape=(samples,timesteps, input_dim)
labels=np.random.randint(2,size=(1,1,32))  #labels.shape=(samples, tiemsteps/pooling_size, output_dim)
