#TextPreprocess
from keras.preprocessing import text

s='你 好 啊 哈 哈'

'''
文本预处理
'''

'''
keras.preprocessing.text.text_to_word_sequence(text, filters=base_filter(), lower=True, split=" ")
#text: 文本
#filters: 过滤器，过滤掉一些字符，比如标点。默认：base_filter(),包含标点，tab,和换行
#lower: boolean
#split: str
'''
#text_to_word_sequence
#文本切分成单词列表

words=text.text_to_word_sequence(s)
for w in words:
    print w.decode('utf-8')
    
'''
keras.preprocessing.text.ont_hot(text,n,filter=base_filter,lower=True,split=" ")
#返回：[1,n]之间的整数列表，每个整数编码一个词
参数：与text_to_word_sequence相同
'''
#one_hot
#把文本编码成大小为n的词汇库的index列表
one_hot=text.one_hot(s,1000)
print one_hot
print text.one_hot('还 好 把 你 忘 了',1000)

'''
Tokenizer
'''