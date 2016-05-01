#Embedding Layer
from keras.models import Sequential
from keras.layers import Embedding
import numpy as np

'''
keras.layers.embeddings.Embedding(input_dim, output_dim, init='uniform',
                                  input_length=None, W_regularizer=None, 
                                  activity_regularizer=None, W_constraint=None,
                                  mask_zero=False, weights=None, dropout=0.0)
'''

'''
Embeding层用来把整数(index)转换成固定大小的向量。比如[[4],[20]] -> [[0.25,0.1],[0.6,-0.2]]

Embeding层只能作为model的第一层
'''
#例子
model=Sequential()
#输入的是1000维向量的下标index
#input_length是每个样本序列的长度
model.add(Embedding(1000,64,input_length=10))
#这个model将会被看作输入一个大小为(batch,input_length)的整形矩阵
#输入的最大整数小于1000(vocabulary size)
#现在model.output_shape==(None,10,64), None是batch维

#32是batch数，即输入的样本数，
#10是每个样本的序列长度
input_array=np.random.randint(1000,size=(32,10))


model.compile('rmsprop','mse')
output_array=model.predict(input_array)
assert output_array.shape==(32,10,64)