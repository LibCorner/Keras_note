#practise
import numpy as np
from keras.layers import Dense,Input,LSTM,merge
from keras.models import Model
from keras.layers import TimeDistributed
'''
keras的functional API是一种定义复杂模型的方式，比如多输出模型，有向无环图，或者有shared层的模型
'''
#1. 第一个例子：全连接的网络
#Sequential模型可能会是个更好的选择来实现这个网络，但是它可以帮助我们简单地开始学习一些东西。

#一个Layer实例是可调用的，参数是个tensor,返回一个tensor
#输入和输出的tensor可以用来定义一个Model
#这个Model可以像Sequential一样进行训练

#返回一个tensor
inputs=Input(shape=(6,))

#一个Layer实例在tensor上是可调用的，并返回一个tensor
x=Dense(3,activation='tanh')(inputs)
x=Dense(2,activation='tanh')(x)
predicts=Dense(2,activation='softmax')(x)

#创建一个Model，包含Input层和三个Dense层
model=Model(input=inputs,output=predicts)
model.compile(optimizer="rmsprop",loss='binary_crossentropy',metrics=['accuracy'])

data=np.array([[1,1,1,0,0,0],
               [0,0,0,1,1,1]])
labels=np.array([[0,0],[1,1]])
#开始训练
model.fit(data,labels,nb_epoch=100)

'''
所有的model和layer一样是可调用的，
使用functional API可以很容易重用训练好的模型：你可以把model看作一个layer,使用一个tensor参数来调用它。
注意，通过调用一个model,你不仅仅是重用model的结构，而且重用它的权值。
'''
in_x=Input(shape=(784,))
#这个会返回上面定义的model的输出
y=model(in_x)

model_1=Model(input=in_x,output=y)
'''先要compile才可以predcate'''
#model_1.predict(data) 

'''
这可以快速的处理输入序列。你可以把图片分类转换成视频分类，只需要一行。
'''


#输入的tensor,有2个时间step序列，每个timestep包含一个6维的向量。
input_sequences=Input(shape=(2,6))

#这个应用前面的model来处理输入序列里的每个timestep
#下面定义的这个layer的输出是2个2维的vector序列
processed_sequences=TimeDistributed(model)(input_sequences)

model_2=Model(input=input_sequences,output=processed_sequences)
model_2.compile(optimizer="rmsprop",loss='binary_crossentropy',metrics=['accuracy'])

data_1=np.array([[[1,1,1,0,0,0],
                  [0,0,0,1,1,1]]])
labels_1=np.array([[[0,0],
                   [1,1]]])

model_2.fit(data_1,labels_1)

'''
多输入和多输出的模型
'''



'''
Shared layers
可以让两个输入共享网络结构和权值。
'''
#两个输入的tweet,每个tweet是一个140个256维的向量序列
#256维的向量的每一维编码一个字符是否出现(0,1)
tweet_a=Input(shape=(140,256))
tweet_b=Input(shape=(140,256))

#要让不同的输入共享一个层，只需要实例化一个layer一次，然后在多个输入上调用它

#这个layer可以输入一个矩阵，返回一个大小为64的向量
shared_lstm=LSTM(64)

#当多次重用一个layer实例时，这个layer的权值也被重用了
encoded_a=shared_lstm(tweet_a)
encoded_b=shared_lstm(tweet_b)

#之后我们可以级联这两个向量
'''注意，这里的merge是小写的不是Merge'''
merged_vector=merge([encoded_a,encoded_b],mode='concat',concat_axis=-1)

#添加逻辑回归
predictions=Dense(1,activation='sigmoid')(merged_vector)

#定义可训练的模型
model_3=Model(input=[tweet_a,tweet_b],output=predictions)

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
#model.fit([data_a,data_b],labels,nb_epoch=10)



