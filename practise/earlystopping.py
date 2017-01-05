#coding:utf-8
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.objectives import mse
from keras.optimizers import SGD,Adam
import numpy as np
import matplotlib.pyplot as plt

#生成数据
x=np.random.normal(0,1.0,size=(1000,1))
y=3*x+np.random.normal(0,1.0,size=(1000,1))
#绘制图像
plt.scatter(x,y,c='red')
plt.show()

model=Sequential()
model.add(Dense(1,bias=0,input_dim=1))
#model.compile(optimizer=SGD(lr=0.01,momentum=0.1),loss=mse,metrics=["acc"])
model.compile(optimizer=Adam(lr=0.1),loss=mse,metrics=["acc"])

#early stopping
earlystopping=EarlyStopping(monitor="val_loss",patience=2)
model.fit(x,y,validation_split=0.2,callbacks=[earlystopping],nb_epoch=1000)
'''
for i in range(10):
    model.fit(x,y,batch_size=10,nb_epoch=5)
    pre=model.predict(x)
    plt.scatter(x,y)
    plt.plot(x,pre,c='red')
    plt.show()
'''
pre=model.predict(x)
plt.scatter(x,y)
plt.plot(x,pre,c='red')
plt.show()