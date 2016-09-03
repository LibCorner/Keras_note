# -*- coding: utf-8 -*-
from keras.layers import LSTM,activations,Layer,Dense,Input,Activation,MaxPooling1D,Flatten,Convolution1D,Merge
from keras.models import Model
from keras.models import K
from keras import activations, initializations, regularizers, constraints
import numpy as np

class MemoryNet(Layer):
    def __init__(self,output_dim,mem_vec_dim,init='glorot_uniform', activation='linear', weights=None,
                 activity_regularizer=None,input_dim=None, **kwargs):
        '''
        Params:
            output_dim: 输出的维度
            mem_vec_dim: query向量的维度
            
        '''
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.mem_vector_dim=mem_vec_dim
        
        self.activity_regularizer = regularizers.get(activity_regularizer)


        self.initial_weights = weights

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MemoryNet,self).__init__(**kwargs)
        
    def build(self,input_shapes):
        input_shape=input_shapes[0]
        assert len(input_shape)==3
        input_dim=input_shape[2]
        self.input_batch=input_shape[0]
        self.input_num=input_shape[1]
        self.W_c=self.init((input_dim,self.output_dim),name='{}_W_c'.format(self.name))
        self.b_c=K.zeros((self.output_dim,),name='{}_b'.format(self.name))
        
        self.W_m=self.init((input_dim,self.mem_vector_dim),name='{}_W_c'.format(self.name))
        self.b_m=K.zeros((self.mem_vector_dim,),name='{}_b'.format(self.name))
        #可训练参数
        self.trainable_weights=[self.W_c,self.W_m,self.b_c,self.b_m]
        
        
    def call(self,inputs,mask=None):
        #w_c=K.repeat(self.W_c,self.input_num)
        #w_m=K.repeat(self.W_m,self.input_num)
    
        x=inputs[0]
        mem_vector=inputs[1]
        
        c=K.dot(x,self.W_c)+self.b_c #context向量
        m=K.dot(x,self.W_m)+self.b_m #memory向量
        mem_vec=K.repeat(mem_vector,self.input_num) #与问题进行内积
        m=K.sum(m*mem_vec,axis=2,keepdims=False)
        s=K.softmax(m)  #softmax
        s=K.reshape(s,(-1,self.input_num,1))
        ctx=self.activation(c*s)
        
        return ctx#self.activation(ctx)
        
    def get_output_shape_for(self,input_shape):
        shape=input_shape[0]
        return (shape[0],shape[1],self.output_dim)

if __name__=="__main__":
    input=Input((20,128))
    mem_input=Input((200,))
    mem=MemoryNet(output_dim=100,mem_vec_dim=200)([input,mem_input])
    mem_v=Flatten()(MaxPooling1D(20)(mem))
    mem=MemoryNet(output_dim=100,mem_vec_dim=100)([mem,mem_v])
    
    out=Flatten()(MaxPooling1D(20)(mem))
    model=Model([input,mem_input],output=out)
    model.compile(optimizer='sgd',loss='mse')
    
    data=np.random.rand(100,20,128)
    mem_vec=np.random.rand(100,200)
    label=np.random.rand(100,100)
    pre=model.predict([data[0:10],mem_vec[0:10]])
    model.fit([data,mem_vec],label)