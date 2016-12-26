#Sequential
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Merge
from keras.layers import LSTM
import numpy as np
from keras.utils.np_utils import to_categorical

'''
Sequential模型是层的线性栈。
'''
#1.创建Sequential模型

#1.1 通过向构造函数里传入一个layer列表来实例化

model= Sequential([
        Dense(32,input_dim=784),
        Activation('relu'),
        Dense(10),
        Activation('softmax')])

#1.2 也可以通过 .add() 方法来添加layers        
model=Sequential()
model.add(Dense(32,input_dim=784))
model.add(Activation('relu'))

'''
具体化输入的shape
模型需要知道输入的shape，因此，Sequential模型的第一层（只是第一层，后面的层的shape会自动推理）需要
接受关于输入shape的信息。这里有几种方法来实现：
'''
#2.1 给第一层传入一个 input_shape 参数，这是一个shape元组（integer或None的元组，
#     None表示可能是任何的整数）。在 input_shape 中，不需要batch维。

#2.2 传入一个 batch_input_shape 参数， 包含batch 维。这对具体化一个固定的batch大小很有用(比如，stateful RNN)

#2.3 对于2D layer,比如Dense, 通过 input_dim来具体化input shape。 对于3D 层支持参数input_dim和input_length。

#下面的三个片段是等价的
model=Sequential()
model.add(Dense(32,input_shape=(784,)))

model=Sequential()
model.add(Dense(32,batch_input_shape=(None,784)))
#注意这里的batch维是'None', 所以这个model可以处理任何大小的batch

model=Sequential()
model.add(Dense(32,input_dim=784))

#同样，下面三个也是等价的
model=Sequential()
model.add(LSTM(32,input_shape=(10,64)))

model=Sequential()
model.add(LSTM(32,batch_input_shape=(None,10,64)))

model=Sequential()
model.add(LSTM(32,input_length=10,input_dim=64))

'''
多个Sequential实例可以通过Merge layer合并成一个单一的输出。
输出是可以被添加为一个新的Sequential模型的第一层。
'''
#这里是一个合并两个独立的输入分支的例子
left_branch=Sequential()
left_branch.add(Dense(32,input_dim=784))

right_branch=Sequential()
right_branch.add(Dense(32,input_dim=784))

merged=Merge([left_branch,right_branch],mode='concat')

final_model=Sequential()
final_model.add(merged)
final_model.add(Dense(10,activation='softmax'))
#Merge 层支持几个预定义的mode

#sum : 逐项相加的和
#concat : 张量级联， 可以使用 concat_axis 来指定具体级联的axis坐标轴
#mul : 逐项相乘
#ave : 张量均值
#dot : 点乘， 可以通过 dot_axes 来指定具体哪一个axes
#cos : 两个2D张量间的余弦相似度
'''
也可以传入一个函数作为 mode 参数，允许任意的转换：
'''
#merged=Merge([left_branch,right_branch],mode=lambda x,y:x-y)

'''
现在你已经知道足够多的信息来定义大多数model。对于复杂的模型，可以通过Sequential和Merge来表达。
'''

'''
Compilation（编译）
训练模型之前，你需要配置学习过程，这是通过compile方法来做的。

#该方法接受3个参数：
#optimizer ： 优化方法。可以是已经存在的优化器的字符串标示(比如，’rmsporp‘或'adagrad')，也可以是Optimizer类的实例
#loss function : 损失函数，这是模型的要最小化的目标函数。可以是已经存在的损失函数的字符串标示(比如categorical_crossentropy或mse),
#也可以是一个objective函数，查看：objectives
#a list of metrics : 对于任何的分类问题，你可能想要设置为metrics=['accuracy']。可以是已存在的metric字符串，也可以是定制的metric函数
'''
#对于多类分类问题
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#对于二分类问题
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#对于均方差回归问题
model.compile(optimizer='rmsprop',loss='mse')

'''
训练
Keras模型是在Numpy数组的输入和标签上训练的。你可以使用fit函数来训练一个模型
'''
#对于一个单输入的两个类别的模型
model=Sequential()
model.add(Dense(1,input_dim=784,activation='softmax'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#生成dummy数据
data =np.random.random((1000,784))
labels=np.random.randint(2,size=(1000,1))

#训练模型，在数据上迭代，batches为32
model.fit(data,labels,nb_epoch=10,batch_size=32)

'''
对于多个输入10个类别的模型
'''
left_branch=Sequential()
left_branch.add(Dense(32,input_dim=784))

right_branch=Sequential()
right_branch.add(Dense(32,input_dim=784))

merged=Merge([left_branch,right_branch],mode='concat')

model=Sequential()
model.add(merged)
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#生成dummy数据
data_1=np.random.random((1000,784))
data_2=np.random.random((1000,784))

#0到9的整数
labels=np.random.randint(10,size=(1000,1))
#把labels转换成二维的矩阵，大小(1000,10)
#使用categorical_crossentropy
labels=to_categorical(labels,10)

#训练模型
#注意，我们传入了一个Numpy数组列表到训练数据
#因此，model有两个输入
model.fit([data_1,data_2],labels,nb_epoch=10,batch_size=32)
