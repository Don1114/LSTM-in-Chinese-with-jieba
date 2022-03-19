import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras
from keras.layers.core import Activation, Dense
from keras.layers import embeddings
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import *
from keras.models import *
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import jieba
import jieba.posseg as pseg
import nltk
import numpy as numpy
from keras.models import load_model
import pandas as pd

jieba.load_userdict('./userdict.txt')

#斷詞
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return ' '.join([word for word, flag in words if flag != 'x'])

save = ""
rank = ["1","2","3","4","5"] # 評論的分數

# 訓練資料的最大字句等
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
with open('./Training_set.csv','r+', encoding='UTF-8') as f:
    for line in f:
        #print(num_recs,"!!")
        if save != "":
            line = save + line
        label = line.split(",")
        if label[0] == 'error':
            continue
        if len(label) > 2 and label[len(label)-1].strip("\n") not in rank:
        # print(len(text),"==")
            for i in range(0,len(label)):
                # print(i,"~")
                if "\n" in label[i]:
                    label[i] = save
                else:
                    save += label[i]
            continue
        elif "\n" in label[0]:
            label[0] = save
            continue
        else:
            save = ""
        #print(label)
        words = jieba_tokenizer(label[0])
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
        #print(num_recs,"~")
# print('max_len ', maxlen)
# print('nb_words ', len(word_freqs))
# print('num_recs', num_recs)
print("phase 1\n")

# 數據
MAX_NUM_WORDS = 2000
MAX_SENTENCE_LENGTH = 600
vocab_size = min(MAX_NUM_WORDS, len(word_freqs)) + 2
word_index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_NUM_WORDS))}
word_index["PAD"] = 0
word_index["UNK"] = 1
index2word = {v:k for k, v in word_index.items()}
X = numpy.empty(num_recs,dtype=list)
y = numpy.zeros(num_recs)
k = 0

# 讀取訓練資料
with open('./comment.csv','r+', encoding='UTF-8') as f:
    for line in f:
        if save != "":
            line = save + line
        label = line.split(",")
        if label[0] == 'error':
            continue
        if len(label) > 2 and label[len(label)-1].strip("\n") not in rank:
        # print(len(text),"==")
            for i in range(0,len(label)):
                # print(i,"~")
                if "\n" in label[i]:
                    label[i] = save
                else:
                    save += label[i]
            continue
        elif "\n" in label[0]:
            label[0] = save
            continue
        else:
            save = ""
        #print(label)
        words = jieba_tokenizer(label[0])
        seqs = []
        for word in words:
            if word in word_index:
                seqs.append(word_index[word])
            else:
                seqs.append(word_index["UNK"])
        X[k] = seqs
        #print(X[i])
        # y[k] = int(label[len(label)-1])
        #y[k] = float(int(label[len(label)-1])/5)
        # #評分4以下為不推薦  
        if label[len(label)-1] <= '3':
            y[k] = int(0) 
        else:
            y[k] = int(1)
        
        k += 1
        #print(k,"~")
print("phase 2\n")
# print(X)
# print(len(X))

# 把長度不足的字句加上空白
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
# 測試組0.2 訓練組0.8
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=50)
# model 建構
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10
model = Sequential()

model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
#LSTM 層
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(1))
# model.add(Activation("relu"))
# model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
#only 0 1 
model.add(Activation("sigmoid"))
#二分法
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
#model 訓練
model.fit(Xtrain, ytrain, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS,validation_data = (Xtest, ytest))

score, acc = model.evaluate(Xtest, ytest, batch_size = BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('預測','真實','句子'))
for i in range(10):
    idx = numpy.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,MAX_SENTENCE_LENGTH)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))

model.save('RNNtest.h5')  # creates a HDF5 file 'model.h5'

# ##### 自己輸入測試
# INPUT_SENTENCES = []  # 在括號眾自行輸入語句
# XX = numpy.empty(len(INPUT_SENTENCES),dtype=list)
# # 轉換文字為數值
# i=0
# for sentence in  INPUT_SENTENCES:
#     words = jieba_tokenizer(sentence)
#     seq = []
#     for word in words:
#         if word in word_index:
#             seq.append(word_index[word])
#         else:
#             seq.append(word_index['UNK'])
#     XX[i] = seq
#     i+=1

# XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
# # 預測，並將結果四捨五入，轉換為 0 或 1
# labels = [int(round(x[0])) for x in model.predict(XX) ]
# label2word = {1:'正面', 0:'負面'}
# # 顯示結果
# for i in range(len(INPUT_SENTENCES)):
#     print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))