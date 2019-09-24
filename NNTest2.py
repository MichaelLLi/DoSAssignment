# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:57:47 2018

@author: Michael
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Conv1D,MaxPooling1D,LSTM, Embedding, Input, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, add,concatenate, subtract
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model
import math
from math import pi
from random import shuffle
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

max_features=50000
maxlen=5
embed_size=300

def loss_func(y_true,y_pred):
     return tf.reduce_mean(tf.squared_difference(tf.log(y_true +1.),tf.log(y_pred+1.)))


train_data=pd.read_csv("NN_input_V10_big_train_near.csv")
test_data=pd.read_csv("NN_input_V10_big_test_near1.csv")
#train_data=train_data.drop(labels=['X1'], axis=1)
#test_data=test_data.drop(labels=['X1'], axis=1)
#data1=pd.read_csv("NN_input_V8.csv")
#data1=data1.drop(labels=['X1'], axis=1)
#data=pd.concat([data,data1],axis=0)
# test_data=test_data[test_data.week_of_year<13]
k=list(range(train_data.shape[0]))
shuffle(k)
train_data=train_data.iloc[k,:]
descs=[str(x).lower() for x in train_data["item_desc"].values]
descs_test=[str(x).lower() for x in test_data["item_desc"].values]
train_c=round(train_data.shape[0]*0.95)
y=train_data.iloc[:,3:22].values
y_train = y[:train_c]
y_valid = y[train_c:]
y_test=test_data.iloc[:,3:22].values
aux=np.concatenate((train_data.iloc[:,22:].values,train_data.iloc[:,1:3].values),axis=1)
aux_train=aux[:train_c]
aux_valid = aux[train_c:]
aux_test = np.concatenate((test_data.iloc[:,22:].values,test_data.iloc[:,1:3].values),axis=1)
list_sentences_train=descs[:train_c]
list_sentences_valid=descs[train_c:]
list_sentences_test=descs_test
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_valid = tokenizer.texts_to_sequences(list_sentences_valid)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

x_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
x_valid = pad_sequences(list_tokenized_valid, maxlen=maxlen)
x_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float64')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('6B_300d_w2v.txt',encoding="utf-8") if len(o.strip().split())==301)
#proj_array=np.vstack((embeddings_index["biology"],embeddings_index["chemistry"],embeddings_index["computer-science"],
#           embeddings_index["economics"],embeddings_index["engineering"],embeddings_index["english"],
#           embeddings_index["history"],embeddings_index["mathematics"],embeddings_index["philosophy"],embeddings_index["physics"]))
#def get_coefs(word,*arr): 
#    return word, np.dot(proj_array,np.asarray(arr, dtype='float32'))
#embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('42B_300d_w2v.txt',encoding="utf-8") if len(o.strip().split())==301)
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index)+1)
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= min(max_features,embedding_matrix.shape[0]): continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
# sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
#with open('embedding.pickle', 'wb') as handle:
#    pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
with tf.device('/gpu:0'):
#    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-5)
    inp = Input(shape=(maxlen,))
    init = Embedding(min(max_features,embedding_matrix.shape[0]), embed_size, weights=[embedding_matrix])(inp)
    x1 = Conv1D(64,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(init)
    x = Dropout(0.5)(x1)
    x = Conv1D(64,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x)
    x2 = add([x,x1])
    x = Conv1D(64,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(x2)
    x = Dropout(0.5)(x)
    x = Conv1D(64,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(x)
    x = add([x,x2])
    x3 = Conv1D(128,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x3)
    x = Conv1D(128,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x)  
    x4 = add([x,x3])
    x = Conv1D(128,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(x4)  
    x = Dropout(0.5)(x)
    x = Conv1D(128,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(x)
    x = add([x,x4])
    x5 = Conv1D(256,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x5)
    x = Conv1D(256,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(x)
    x = Dropout(0.5)(x)  
    x6 = add([x,x5])
    x = Conv1D(256,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(x6)  
    x = Dropout(0.5)(x)
    x = Conv1D(256,4, activation="relu", padding='same',kernel_initializer='glorot_normal')(x)
    x = add([x,x6])
    y = Bidirectional(GRU(100, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(init)
    x = GlobalMaxPool1D()(x)
    y = GlobalMaxPool1D()(y)
    aux_inp=Input(shape=(aux_train.shape[1],))
    x=concatenate([x,aux_inp])
    y=concatenate([y,aux_inp])
    x = Dense(2000)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1000)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(500)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    y = Dense(2000)(y)
    y = Activation("relu")(y)
    y = Dropout(0.5)(y)
    y = Dense(1000)(y)
    y = Activation("relu")(y)
    y = Dropout(0.5)(y)
    y = Dense(500)(y)
    y = Activation("relu")(y)
    y = Dropout(0.5)(y)
    x0 = add([x,y])
#    x0 = Dense(500, activation="relu")(x0)
#    x1 = Dropout(0.5)(x1)
#    x2 = Dense(500, activation="relu")(x0)
#    x2 = Dropout(0.5)(x2)
#    x3 = Dense(500, activation="relu")(x0)
#    x3 = Dropout(0.5)(x3)    
#    x4 = Dense(500, activation="relu")(x0)
#    x4 = Dropout(0.5)(x4)
#    x5 = Dense(500, activation="relu")(x0)
#    x5 = Dropout(0.5)(x5)   
#    x6 = Dense(500, activation="relu")(x0)
#    x6 = Dropout(0.5)(x6)
#    x7 = Dense(500, activation="relu")(x0)
#    x7 = Dropout(0.5)(x7)
#    x8 = Dense(500, activation="relu")(x0)
#    x8 = Dropout(0.5)(x8)
#    x9 = Dense(500, activation="relu")(x0)
#    x9 = Dropout(0.5)(x9)
    
    x1 = Dense(1, activation="relu")(x0)
    x2 = Dense(1, activation="relu")(x0)
    x2 = add([x1,x2])
    x3 = Dense(1, activation="relu")(x0)
    x3 = add([x2,x3])
    x4 = Dense(1, activation="relu")(x0)
    x4 = add([x3,x4])
    x5 = Dense(1, activation="relu")(x0)
    x5 = add([x4,x5])
    x6 = Dense(1, activation="relu")(x0)
    x6 = add([x5,x6])
    x7 = Dense(1, activation="relu")(x0)
    x7 = add([x6,x7])
    x8 = Dense(1, activation="relu")(x0)
    x8 = add([x7,x8])
    x9 = Dense(1, activation="relu")(x0)
    x9 = add([x8,x9])
    x10 = Dense(1, activation="relu")(x0)
    x10 = add([x9,x10])
    x11 = Dense(1, activation="relu")(x0)
    x11 = add([x10,x11])
    x12 = Dense(1, activation="relu")(x0)
    x12 = add([x11,x12])
    x13 = Dense(1, activation="relu")(x0)
    x13 = add([x12,x13])
    x14 = Dense(1, activation="relu")(x0)
    x14 = add([x13,x14])
    x15 = Dense(1, activation="relu")(x0)
    x15 = add([x14,x15])
    x16 = Dense(1, activation="relu")(x0)
    x16 = add([x15,x16])
    x17 = Dense(1, activation="relu")(x0)
    x17 = add([x16,x17])
    x18 = Dense(1, activation="relu")(x0)
    x18 = add([x17,x18])
    x19 = Dense(1, activation="relu")(x0)
    x19 = add([x18,x19])
    x=concatenate([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19])
    model = Model(inputs=[inp,aux_inp], outputs=x)
    model.compile(loss='msle', optimizer='adam')
    history=model.fit([x_train,aux_train], y_train,
              epochs=10,
              batch_size=64,validation_data=([x_valid,aux_valid],y_valid))
    score = model.evaluate([x_test,aux_test], y_test, batch_size=64)
    y_pred=model.predict([x_test,aux_test],batch_size=64)
    output=np.concatenate((y_pred,y_test,np.reshape(list_sentences_test,(len(list_sentences_test),1))),axis=1)
    output_df=pd.DataFrame(output)
    output_df.to_csv("test_output_v3_near1.csv")