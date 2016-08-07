# -*- coding: utf-8 -*-
#MyKerasLayer
'''
对于简单的，无状态的定制操作，建议使用layers.core.Lambda层来实现。但是对于包含可训练的权值的定制操作，应该实现自己的层。
这里是一个Keras层的骨架，你只需要实现三个方法：
1.build(input_shape) : 在这里可以定义你的权值，可训练的权值应该添加到self.trainable_weights列表里。
    结点的其他属性有：self.non_trainable_weigts(list)和self.updates (更新元组(tensor,new_tensor)的列表)。
    可以查看BatchNormalization层使如何使用non_trainable_weights和updates的。
2.call(x): 这是进行逻辑处理的地方。除非你想要你的层支持masking,否则你只需要关心传给call的第一个参数：输入的tensor

3.get_output_shape_for(input_shape):一旦你的层修改了输入的shape,你应该在这里实现shape转换的逻辑，以便于Keras自动推理shape

'''

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim=output_dim
        super(MyLayer,self).__init__(**kwargs)
        
    def build(self,input_shape):
        input_dim=input_shape[1]
        initial_weight_value=np.random.random((input_dim,self.ouput_dim))
        self.W=K.variable(initial_weight_value)
        self.trainable_weights=[self.W]
        
    def call(self,x,mask=None):
        return K.dot(x,self.W)
        
    def get_output_shape_for(self,input_shape):
        return (input_shape[0],self.output_dim)




