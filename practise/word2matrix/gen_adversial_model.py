# -*- coding: utf-8 -*-
#cnn
from keras.models import Model
from keras.layers import Convolution2D,Input,MaxPooling2D,Flatten,Dense
from keras.regularizers import l2
import theano
import pylab
import matplotlib.cm as cm
from word2matrix import *
import random

word_len=10
vocab_size=chars_size

def get_gen_model():
    '''生成网络，输入句子矩阵，生成单词'''
    input=Input(shape=(10,word_len,vocab_size))
    conv1=Convolution2D(nb_filter=64,nb_row=5,nb_col=5,border_mode='same',activation='relu')(input)
    #pool1=MaxPooling2D(pool_size=(1,4),border_mode='same')(conv1)
    conv2=Convolution2D(3,5,5,border_mode='same',activation='relu')(conv1)
    #pool2=MaxPooling2D(pool_size=(1,2),border_mode='same')(conv2)
    conv3=Convolution2D(1,5,5,border_mode='same',activation='relu')(conv2)
    
    model=Model(input=input,output=conv3)
    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    return model
def get_discriminator_model():
    '''鉴别器，识别一个单词是生成的还是真实的'''
    input=Input(shape=(1,word_len,vocab_size))
    conv1=Convolution2D(nb_filter=64,nb_row=5,nb_col=5,border_mode='same',activation='relu')(input)
    pool1=MaxPooling2D(pool_size=(1,4))(conv1)
    conv2=Convolution2D(3,5,5,border_mode='same',activation='relu')(pool1)
    pool2=MaxPooling2D(pool_size=(1,2))(conv2)
    conv3=Convolution2D(1,5,5,border_mode='same',activation='relu')(pool2)
    flat=Flatten()(conv3)
    out=Dense(1,activation='tanh',W_regularizer=l2(0.01))(flat)
    
    model=Model(input=input,output=out)
    model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
    return model

def build_models():
    gen_model=get_gen_model()
    discriminator=get_discriminator_model()
    #real_input=Input(shape=(1,word_len,vocab_size))
    #real_out=discriminator(real_input,real_out)
    #discriminator_real=Model(real_input,real_out)    
    
    fake_input=Input(shape=(10,word_len,vocab_size))
    fake_word=gen_model(fake_input)
    out=discriminator(fake_word)
    discriminator_gen=Model(fake_input,out)
    discriminator_gen.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
    return discriminator_gen,discriminator,gen_model

discriminator_gen,discriminator,gen_model=build_models()


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

def train(sentences,discriminator_gen=discriminator_gen,discriminator=discriminator,gen_model=gen_model):
    
    data,real_word=getData(sentences)
    
    fake_word=gen_model.predict(data)
    pylab.imshow(fake_word[0][0], cmap=cm.Greens,origin='lower')
    
    real_label=np.zeros((1,))
    fake_label=np.ones((1,))
    
    #discriminator.predict(real_word)
    
    discriminator_gen.fit(data,fake_label,nb_epoch=5)
    fake_word=gen_model.predict(data)
    discriminator.fit(fake_word,real_label,nb_epoch=5)
    discriminator.fit(real_word,fake_label,nb_epoch=5)
for i in range(500):
    train(sentences)
fake_word=gen_model.predict(data)
pylab.imshow(fake_word[0][0], cmap=cm.Greens,origin='lower')
indices=np.argmax(fake_word[0][0],axis=1)
word=[vocab[i][0] for i in indices]
print ''.join(word)
