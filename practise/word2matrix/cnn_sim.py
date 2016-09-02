# -*- coding: utf-8 -*-
#cnn
from keras.models import Model
from keras.layers import Convolution2D,Input,MaxPooling2D,Flatten,Dense,merge,Lambda
from keras.regularizers import l2
import keras.backend as K
import theano
import pylab
import matplotlib.cm as cm
from word2matrix import *
import random

word_len=10
vocab_size=chars_size

def get_model(channel=10):
    input=Input(shape=(channel,word_len,vocab_size))
    conv1=Convolution2D(nb_filter=64,nb_row=3,nb_col=5,border_mode='same',activation='relu')(input)
    pool1=MaxPooling2D(pool_size=(1,2))(conv1)
    conv2=Convolution2D(128,3,5,border_mode='same',activation='relu')(pool1)
    pool2=MaxPooling2D(pool_size=(2,2))(conv2)
    conv3=Convolution2D(1,5,5,border_mode='same',activation='relu')(pool2)
    out=Flatten()(conv3)
    out=Dense(128,activation='tanh')(out)
    model=Model(input=input,output=out)
    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    return model

def get_cosine(sent_out,word_out):
    cos=K.sum(sent_out*word_out,axis=-1)/(K.sqrt(K.sum(sent_out*sent_out,axis=-1)*K.sum(word_out*word_out,axis=-1))+0.0000001)
    return cos

def build_model():
    sent_model=get_model(10)
    word_model=get_model(1)
    sent=Input((10,word_len,vocab_size))
    word=Input((1,word_len,vocab_size))
    sent_out=sent_model(sent)
    word_out=word_model(word)
    
    sim=merge([sent_out,word_out],mode=lambda x:get_cosine(x[0],x[1]),output_shape=lambda x:(x[0],1))
    
    model=Model([sent,word],sim)
    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    return sent_model,word_model,model


sent_model,word_model,model=build_model()

sentences=['IPython has a built-in mode to work cleanly with matplotlib figures']

def getData(sentences):
    datas=[]
    real_words=[]
    for s in sentences:
        sentence=s.split(' ')
        real_word=random.sample(sentence,1)
        data=sentence2matrix(sentence)
        datas.append(data)
        real_word=sentence2matrix(real_word,sent_len=1)
        real_words.append(real_word)
    #batch=len(sentences)
    return np.array(datas),np.array(real_words)

def train(sentences):
    data,real_word=getData(sentences)
    labels=np.ones((len(sentences),))
    model.fit([data,real_word],labels,nb_epoch=5)


data,real_word=getData(sentences)
fake_word=gen_model.predict(data)

def get_np_cos(x1,x2):
    return np.sum(x1*x2,axis=-1)/np.sqrt(np.sum(x1*x1,axis=-1)*np.sum(x2*x2,axis=-1)+0.0000000001)