# -*- coding: utf-8 -*-
from keras.layers import LSTM,activations,Layer,Dense,Input,Activation,MaxPooling1D,Flatten,Convolution1D,Merge,InputSpec,time_distributed_dense
from keras.models import Model
from keras.models import K
from keras import activations, initializations, regularizers, constraints
import numpy as np
class AttentionLSTM(LSTM):
    def __init__(self,output_dim,att_dim,attn_activation='tanh',
                 attn_inner_activation='tanh',
                 single_attn=False,**kwargs):
        '''
            attention_vec: 输入到这一层的attention向量，根据这个向量来计算这一层的attention输出
            single_attention_param: 每个时间t,的向量中的元素是否使用同一个attention值
        '''
        self.attn_activation=activations.get(attn_activation)
        self.attn_inner_activation=activations.get(attn_inner_activation)
        self.single_attention_param=single_attn
        self.input_spec=None
        self.att_dim=att_dim
        super(AttentionLSTM,self).__init__(output_dim,**kwargs)
    

    def build(self,input_shapes):
        '''
        build方法初始化权重矩阵    
        U_a: x到attention输出的权值矩阵
        U_m: attention_vec到attention输出的取值矩阵
        U_s: attention输出到softmax输出的权重矩阵
        '''
        input_shape=input_shapes[0]
        super(AttentionLSTM,self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shapes[0]),InputSpec(shape=input_shapes[1])]
        #attention_dim=self.input_spec[1].shape[1]
        attention_dim=self.att_dim
        input_dim = input_shape[2]
        #attention参数
        self.U_a=self.inner_init((input_dim,self.output_dim),
                                 name='{}_U_a'.format(self.name))
        self.b_a=K.zeros((self.output_dim,),name='{}_b_a'.format(self.name))
        
        self.U_m=self.inner_init((attention_dim,self.output_dim),
                                 name='{}_U_m'.format(self.name))
        self.b_m=K.zeros((self.output_dim,),name='{}_b_m'.format(self.name))
        
        if self.single_attention_param:
            self.U_s = self.inner_init((self.output_dim, 1),
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.output_dim,), name='{}_b_s'.format(self.name))
        
        self.trainable_weights+=[self.U_a,self.U_m,self.U_s,
                                 self.b_a,self.b_m,self.b_s]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
    def get_initial_states(self,inputs):
        return super(AttentionLSTM,self).get_initial_states(inputs[0])
    def preprocess_input(self,inputs):
        return super(AttentionLSTM,self).preprocess_input(inputs[0])
    def compute_mask(self, input, mask):
        '''计算mask'''
        if self.return_sequences:
            return mask[0]
        else:
            return None    
    def call(self, x, mask=None):
        mask=mask[0]
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output
    
    def step(self,x,states):
        '''
            step方法由父类RNN调用，定义每次输入在网络中的传播的运算
            states[4]存放attention_vec到attention层的输出状态        
        '''
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        if self.consume_less == 'cpu':
            x_i = x[:, :self.output_dim]
            x_f = x[:, self.output_dim: 2 * self.output_dim]
            x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
            x_o = x[:, 3 * self.output_dim:]
        else:
            x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
            x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
            x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
            x_o = K.dot(x * B_W[3], self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)
        
        attention=states[4]
        m = self.attn_inner_activation(K.dot(K.dot(x_i,self.W_i.T), self.U_a) +attention + self.b_a)
        # Intuitively it makes more sense to use a sigmoid (was getting some NaN problems
        # which I think might have been caused by the exponential function -> gradients blow up)
        s = self.attn_activation(K.dot(m, self.U_s) + self.b_s)

        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.output_dim, axis=1)
        else:
            h = h * s
        return h, [h, c]
        
    def get_constants(self,inputs):
        '''
         get_constants方法有父类LSTM调用，定义了在step函数外的组件，这些组件就不需要序列中的每次输入都重新计算        
        '''
        x=inputs[0]
        attention_vec=inputs[1]
        constants=super(AttentionLSTM,self).get_constants(x)
        constants.append(K.dot(attention_vec,self.U_m)+self.b_m)
        return constants    
    def get_output_shape_for(self,input_shapes):
        input_shape=input_shapes[0]
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

class LSTMLayer(LSTM):
    def __init__(self,**kwargs):
        super(LSTMLayer,self).__init__(**kwargs)

if __name__=="__main__":
    input=Input((20,100))
    att_vec=Input((100,))
    lstm=AttentionLSTM(output_dim=128,att_dim=100,return_sequences=True)([input,att_vec])
    flat=Flatten()(lstm)
    out=Dense(128)(flat)
    model=Model([input,att_vec],out)
    model.compile(optimizer='sgd',loss='mse')
    
    data=np.random.rand(30,20,100)
    att_vecs=np.random.rand(30,100)
    labels=np.random.rand(30,128)
    #pre=model.predict([data,att_vecs])
    model.fit([data,att_vecs],labels)
    