# -*- coding: utf-8 -*-
#cdssm_Q  question是cnn 
from keras.layers import Input,Dense,merge,Lambda,Dropout,Convolution1D,Flatten,MaxPooling1D,AveragePooling1D
from keras.models import Model,K
import numpy as np
import theano
#配置blas,防止报错
theano.config.blas.ldflags='-LC:\\OpenBLAS\\bin -lopenblas'

base_path='F:/QA/'
dim=4000
line=''
#加载数据
def load_data(path=base_path+'question_predicate/questions_vector.txt',samples=1000):
    f=open(path)
    X_train=[]
    for i in range(samples):
        line=f.readline()
        words=line.strip().split(" ")
        #print len(words)
        #temp=[float(words[j+1]) for j in range(len(words)-1)]
        temp=[float(words[j]) for j in range(len(words)) if words[j]!='' and words[j]!='\n']
        x=[0]*dim
        length=len(temp)
        if length<dim:
            x[0:length]=temp
        else:
            x[0:dim]=temp[0:dim]
        X_train.append(np.reshape(np.array(x),(20,200)))
    X_train=np.array(X_train)
    print X_train.shape
    return  X_train
#加载predicate    
def load_predicate_data(path=base_path+"question_predicate/predicates_vector.txt",samples=1000):
    f=open(path)
    X_train=[]
    for i in range(samples):
        line=f.readline()
        words=line.strip().split(" ")
        #print len(words)
        temp=[float(words[j]) for j in range(len(words)) if words[j]!='' and words[j]!='\n']
        x=[0]*1000
        length=len(temp)
        if length<1000:
            x[0:length]=temp
        else:
            x[0:1000]=temp[0:1000]
        X_train.append(x)
    X_train=np.array(X_train)
    return  X_train    


#cos函数
def cosine(x1,x2):
    return K.sum(x1*x2,axis=-1)/(K.sqrt(K.sum(x1*x1,axis=-1)*K.sum(x2*x2,axis=-1))+0.0000001) #cos
    
#定义cos目标函数
def cosine_error(x):  #x=[x1,x2,x3,x4] ,xi.shape=(batch_size,input_dim)
    cos1=cosine(x[0],x[1]) #cos shape=(batch_size,)
    cos2=cosine(x[0],x[2])
    cos3=cosine(x[0],x[3])
    cos4=cosine(x[0],x[4])
    cos5=cosine(x[0],x[5])
    cos6=cosine(x[0],x[6])
    delta=5 
    p=K.exp(cos1*delta)/(K.exp(cos1*delta)+K.exp(cos2*delta)+K.exp(cos3*delta)+K.exp(cos4*delta)+K.exp(cos5*delta)+K.exp(cos6*delta)) #softmax
    f=-K.log(p) #objective function：-log  #f.shape=(batch_size,)
    return K.reshape(f,(K.shape(p)[0],1))  #return.sahpe=(batch_size,1)


class CDSSM(object):
    def __init__(self,samples=1000):

        #问题Model
        input_1=Input(shape=(20,200))
        x=Convolution1D(400,3,border_mode='same',input_shape=(20,200))(input_1)
        #x=Dropout(0.5)(x)
        x=AveragePooling1D(20)(x)
        x=Dropout(0.25)(x)
        x=Flatten()(x)
        output_1=Dense(128,activation="tanh")(x)
        self.model_1=Model(input=input_1,output=output_1)
        self.model_1.compile(optimizer="sgd",loss="mse",metrics=['accuracy'])


        #谓语属性Model
        input_2=Input(shape=(1000,))
        x=Dense(400,activation="tanh")(input_2)
        x=Dropout(0.25)(x)
        output_2=Dense(128,activation="tanh")(x)
        #output_2=Lambda(lambda x:x*(-1))(output_2)
        self.model_2=Model(input=input_2,output=output_2)
        self.model_2.compile(optimizer="sgd",loss="mse",metrics=['accuracy'])

        #输入两个样本：正样本和负样本
        input_2_a=Input(shape=(1000,))
        input_2_b=Input(shape=(1000,))
        input_2_c=Input(shape=(1000,))
        input_2_d=Input(shape=(1000,))
        input_2_e=Input(shape=(1000,))
        input_2_f=Input(shape=(1000,))
        output_2_a=self.model_2(input_2_a)
        output_2_b=self.model_2(input_2_b)
        output_2_c=self.model_2(input_2_c)
        output_2_d=self.model_2(input_2_d)
        output_2_e=self.model_2(input_2_e)
        output_2_f=self.model_2(input_2_f)

        #合并,输出
        output=merge(inputs=[output_1,output_2_a,output_2_b,output_2_c,output_2_d,output_2_e,output_2_f],mode=cosine_error,output_shape=(None,1))

        #构造模型    
        self.model=Model([input_1,input_2_a,input_2_b,input_2_c,input_2_d,input_2_e,input_2_f],output=output)
        self.model.compile(optimizer="sgd",loss='mse',metrics=['accuracy'])
        

        #训练数据
        #rand=np.random
        self.X_train_1=load_data(samples=samples)  #question
        self.X_train_2_a=load_predicate_data(samples=samples) #正样本
        #self.X_train_2_b=np.array([self.X_train_2_a[rand.randint(999)] for i in range(1000)]) #负样本
        #self.X_train_2_c=np.array([self.X_train_2_a[rand.randint(999)] for i in range(1000)]) #负样本
    
    def train(self,samples_num=1000,nb_epoch=50,batch_size=5):
        labels=np.array([[0]]*samples_num)
        rand=np.random
        X_train_2_b=np.array([self.X_train_2_a[rand.randint(samples_num)] for i in range(samples_num)]) #负样本
        X_train_2_c=np.array([self.X_train_2_a[rand.randint(samples_num)] for i in range(samples_num)]) #负样本
        X_train_2_d=np.array([self.X_train_2_a[rand.randint(samples_num)] for i in range(samples_num)]) #负样本
        X_train_2_e=np.array([self.X_train_2_a[rand.randint(samples_num)] for i in range(samples_num)]) #负样本
        X_train_2_f=np.array([self.X_train_2_a[rand.randint(samples_num)] for i in range(samples_num)]) #负样本
        self.model.fit([self.X_train_1[0:samples_num],self.X_train_2_a[0:samples_num],X_train_2_b[0:samples_num],X_train_2_c[0:samples_num],
                        X_train_2_d[0:samples_num],X_train_2_e[0:samples_num],X_train_2_f[0:samples_num]],labels,nb_epoch=nb_epoch,batch_size=batch_size)
        

    def encode_question(self,question): #question.shape=(samples,4000)
        return self.model_1.predict(question) #return.shape=(samples,128)
    
    def encode_predicate(self,predicate): #question.shape=(samples,1000)
        return self.model_2.predict(predicate) #return.shape=(samples,128)
    
    #保存权值
    def save_weights(self):
        self.model.save_weights(base_path+'question_predicate/cdssm_weights.h5')
        self.model_1.save_weights(base_path+'question_predicate/cdssm_weights_question.h5')
        self.model_2.save_weights(base_path+'question_predicate/cdssm_weights_predicate.h5')
    #加载权值   
    def load_weights(self,path=base_path+'question_predicate/cdssm_weights.h5'):
        self.model.load_weights(path)
        self.model.compile(optimizer="sgd",loss='mse',metrics=['accuracy'])
#计算cos
def cos(x,y):
    return np.sum(x*y,axis=-1)/(np.sqrt(np.sum(x*x,axis=-1)*np.sum(y*y,axis=-1))+0.000000001)

#softmax概率    
def cos_probability(x_true,x_positive,x_neg):
     cos1=cos(x_true,x_positive)
     cos2=cos(x_true,x_neg)
     p=np.exp(cos1*5)/(np.exp(cos1*5)+np.exp(cos2*5))
     return -np.mean(np.log(p))
     
def mse(x,y):
    return np.mean((x-y)**2,axis=-1)
    
#保存权值
def save_weights(cdssm):
    cdssm.model.save_weights(base_path+'question_predicate/cdssm_weights.h5',overwrite=True)
    cdssm.model_1.save_weights(base_path+'question_predicate/cdssm_weights_question.h5',overwrite=True)
    cdssm.model_2.save_weights(base_path+'question_predicate/cdssm_weights_predicate.h5',overwrite=True)
    
cdssm=CDSSM(samples=5000)
#cdssm.load_weights()

def train_model(cdssm,iter_num=10,samples_num=5000,nb_epoch=50,batch_size=50):
    for i in range(iter_num):
        cdssm.train(samples_num=samples_num,nb_epoch=nb_epoch,batch_size=batch_size)
#train_model(cdssm,iter_num=10,samples_num=5000,nb_epoch=100,batch_size=50)
'''
cdssm.load_weights()
cdssm.train()
save_weights(cdssm)
questions=cdssm.encode_question(cdssm.X_train_1)
predicates=cdssm.encode_predicate(cdssm.X_train_2_a)
cos(questions[0],predicates[0])
'''
def write_encoded_data(encoded_data,path):
    f=open(path,'w')
    for data in encoded_data:
        for e in data:
            f.write(str(e)+" ")
        f.write('\n')
    f.close()
'''            
questions=load_data('F:/QA/question_predicate/questions_vector_all.txt',14609)
predicates=load_predicate_data('F:/QA/question_predicate/predicates_vector_all.txt',128837)
pre_question=cdssm.encode_question(questions)
pre_predicate=cdssm.encode_predicate(predicates)
write_encoded_data(pre_question,'F:/QA/question_predicate/cdssm.encoded_questions.txt')
write_encoded_data(pre_predicate,'F:/QA/question_predicate/cdssm.encoded_predicates.txt')
'''


