#coding:utf-8
import keras.backend as K
import numpy as np
import theano

x1=np.array([[1,1,1],[1,2,3]])
x2=np.array([[1,2,3],[3,2,1]])
a=K.variable(x1)
b=K.variable(x2)
#dot
out1=K.dot(a,b.T)
dot=theano.function([],out1)
#batch_dot
out2=K.batch_dot(a,b,axes=1)
batch_dot=theano.function([],out2)
print dot()
print batch_dot()


x1=np.random.rand(3,1,10)
x2=np.random.rand(3,5,10)

#定义变量
a=K.variable(x1)
b=K.variable(x2)
#batch_dot
out2=K.batch_dot(a,b,axes=2)

batch_dot=theano.function([],out2)
res=batch_dot()
print res
print res.shape
