# -*- coding: utf-8 -*-
#RNN
from keras.layers import LSTM
from keras.models import Model,Sequential
import numpy as np
'''
1.Recurrent
keras.layers.recurrent.Recurrent(weights=None, return_sequences=False, go_backwards=False, stateful=False, unroll=False, consume_less='cpu', input_dim=None, input_length=None)
Recurrent是循环神经网络层的抽象类，不要在model中使用它--它是不可用的。应该使用它的子类LSTM,GRU和SimpleRNN
'''
#example
#作为Sequential模型的第一层
model=Sequential()
#输入10个64维的向量，
#设置return_sequences=True，输出的是序列，即10个32维的向量，否则输出为向量
model.add(LSTM(32,input_shape=(10,64),return_sequences=True))
#现在,model.output_shape==(None,10,32)
#'None'是batch维

#下面的model和上面是一样的
model=Sequential()
#设置return_sequences=True，输出的是序列，即10个32维的向量，否则输出为向量
model.add(LSTM(32,input_dim=64,input_length=10,return_sequences=True))

#接下来的层不需要input size
model.add(LSTM(16))
#最后的输出是一个向量，而不是一个序列，因为没有设置return_sequences=True

data=np.random.random((5,10,64))
labels=np.random.randint(0,2,(5,16))
print data.shape
#编译模型
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
#训练
model.fit(data,labels,nb_epoch=10,batch_size=2)

'''
参数：
1.weights: numpy数组列表作为初始权值，这个列表应该有3个元素，他们的shape:[(input_dim,output_dim),(output_dim,output_dim),(output_dim,)]
2.return_sequences: Boolean.是否返回输出序列的最后一个向量，还是整个输出序列
3.go_backwards: Boolean(默认False),如果True,反向处理输入序列
4.stateful: Boolean(默认False)，如果True,batch中的每个样本上一次的状态将会用来作为下一个batch中相应样本的初始状态。
5.unroll: Boolean(默认False),如果True,网络会被展开，否则会使用符号loop.当使用TensorFlow是，总是unrolled的。unrolling可以为RNN提速，但是会更memory-intensive,只适合短序列。
6.consume_less: cpu,mem和gpu中的一个(LSTM/GRU only).
'''
#Masking

#SimpleRNN
'''
keras.layers.recurrent.SimpleRNN(output_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
全连接的RNN,输出fed back to 输入
'''

#GRU

#LSTM

