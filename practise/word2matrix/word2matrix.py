# -*- coding: utf-8 -*-
#word2matrix
import numpy as np
from PIL import Image
import pylab


path='E:\\Python Scripts\\insuranceQA-master\\V1\\vocabulary'
f=open(path)

vocab={}
for line in f:
    i_w=line.strip().lower().split('\t')
    i=i_w[0]
    w=i_w[1]
    chars=list(w)
    for c in chars:
        if c not in vocab.keys():
            vocab[c]=0
        vocab[c]+=1

vocab=sorted(vocab.items(),key=lambda d:d[1],reverse=True)
chars_index=dict([(vocab[e][0],e) for e in range(len(vocab))])
chars_size=len(chars_index)+1

def word2matrix(word,max_len=10,chars_size=chars_size):
    matrix=np.zeros((max_len,chars_size))
    chars=list(word)
    for c in range(min(len(chars),max_len)):
        i=chars_index.get(chars[c],chars_size-1)
        matrix[c][i]=1
        
    return matrix
def sentence2matrix(sentence,sent_len=10,word_len=10,chars_size=chars_size):
    words=np.zeros((sent_len,word_len,chars_size))
    for i in range(sent_len):
        words[i]=word2matrix(sentence[i],word_len,chars_size)
    return words
        
if __name__=='__main__':
    word='my'
    im=pylab.imshow(word2matrix(word))
    pylab.colorbar(im,orientation='horizontal')