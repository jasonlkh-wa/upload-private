# %%
import os
import re
import sys
import pandas as pd
import numpy as np
from joblib import dump, load
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import transformers
from tensorflow import keras
from transformers import TFDistilBertModel, DistilBertTokenizerFast
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, TextVectorization, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub


def regex_tokenizer(text, stopwords, limit=False):
    """
    The customzied word tokenizer for emails.
    """
    output = []
    Tokenizer = RegexpTokenizer(
        r"n't|([a-z]+[-_]?[a-z]+)|[a-z]+[a-z0-9]*", gaps=False)
    for i in Tokenizer.tokenize(text.lower()):
        if i not in stopwords:
            output += [i]
        if type(limit) != bool and len(output) >= limit:
            break
    return output

def bidirectional_lstm(max_length=500, max_dict=50000):
    # Bi-directional LSTM
    tokenizer = Tokenizer(oov_token='oov')
    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')

    x_test =  tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

    nn_clf = keras.Sequential()
    nn_clf.add(Input(max_length,))
    nn_clf.add(Embedding(len(tokenizer.word_counts)+2, 32,
               name='embedding', input_length=max_length))
    nn_clf.add(LSTM(8, activation='relu', return_sequences=False))
    nn_clf.add(BatchNormalization())
    nn_clf.add(Dense(8, activation='relu'))
    nn_clf.add(BatchNormalization())
    nn_clf.add(Dense(1, activation='sigmoid'))

    nn_clf.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics='accuracy')
    print(nn_clf.summary())

    return nn_clf


def distilbert_tokenize(df : pd.DataFrame, text_col, max_length=500):
    tokenizer = DistilBertTokenizerFast.from_pretrained('./huggingface_model/distilbert', local_files_only=True)
    df[['attention_mask', 'input_ids']] = df[text_col].agg(tokenizer, max_length=max_length, padding='max_length', truncation=True, return_tensors='tf').tolist()
    return df


def build_model(transformer, max_length, lr=5e-5):
    weight_initializer = keras.initializers.GlorotNormal(seed=1)

    input_ids_layer = Input(shape=(max_length, ), name='input_ids', dtype='int32')
    
    input_attention_layer = Input(shape=(max_length, ), name='input_attention', dtype='int32')

    output = transformer(input_ids_layer, input_attention_layer).last_hidden_state[:, 0, :]

    output = Dense(16, activation='relu', kernel_initializer=weight_initializer, kernel_constraint=None, bias_initializer='zeros')(output)

    output = Dense(1, activation='sigmoid', kernel_initializer=weight_initializer, kernel_constraint=None, bias_initializer='zeros')(output)

    model = keras.Model([input_ids_layer, input_attention_layer], output)

    model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    return model

def hub_model(lr=5e-5):
    embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
    hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    model.summary()

    return model


if __name__ == '__main__':
    np.random.seed(1)
    stopwords = stopwords

    raw = pd.read_csv("./data/IMDB Dataset.csv")
    # x_train, x_test, y_train, y_test = train_test_split(
    #     raw['review'], raw['sentiment'], test_size=0.3, shuffle=True, random_state=1)

    # for i in [x_train, x_test, y_train, y_test]:
    #     print(i.shape)

    # y_train = y_train.agg(lambda x: 1 if x == 'positive' else 0)
    # y_test = y_test.agg(lambda x: 1 if x == 'positive' else 0)

    # # Bi-directional LSTM
    # nn_clf = bidirectional_lstm()
    # nn_clf.fit(x_train, y_train, shuffle=True, epochs=20,
    #            validation_data=(x_test, y_test), batch_size=16)

    # Distilbert implementation
    # Tokenize and train-test-split
    distil_maxlen = 300
    df_distil = distilbert_tokenize(raw, 'review', max_length=distil_maxlen)
    print('start split')
    df_train, df_test = train_test_split(
        df_distil, test_size=0.3, shuffle=True, random_state=1)
    df_train = df_train.iloc[:2000]
    df_test = df_test.iloc[:1000]
    # Convert target to binary
    print('start binary')
    tensor_train_label = df_train['sentiment'].agg(lambda x: 1 if x == 'positive' else 0)
    tensor_test_label = df_test['sentiment'].agg(lambda x: 1 if x == 'positive' else 0)
    
    # Convert to tf tensor
    print('convert_train')
    tensor_train = tf.convert_to_tensor(df_train['input_ids'].tolist())
    tensor_train_mask = tf.convert_to_tensor(df_train['attention_mask'].tolist())
    tensor_train_label = tf.convert_to_tensor(tensor_train_label)
    tensor_train = tf.reshape(tensor_train, (-1, distil_maxlen))
    tensor_train_mask = tf.reshape(tensor_train_mask, (-1, distil_maxlen))

    print('convert_test')
    tensor_test = tf.convert_to_tensor(df_test['input_ids'].tolist())
    tensor_test_mask = tf.convert_to_tensor(df_test['attention_mask'].tolist())
    tensor_test_label = tf.convert_to_tensor(tensor_test_label)
    tensor_test = tf.reshape(tensor_test, (-1, distil_maxlen))
    tensor_test_mask = tf.reshape(tensor_test_mask, (-1, distil_maxlen))

    # Distilbert Import
    distilbert = TFDistilBertModel.from_pretrained('./huggingface_model/distilbert', local_files_only=True)

    for layer in distilbert.layers:
        layer.trainable = False
    
    model = build_model(distilbert, distil_maxlen, lr=5e-5)
    model.summary()
    model.fit([tensor_train, tensor_train_mask], tensor_train_label, epochs=10, batch_size=128, validation_data=([tensor_test, tensor_test_mask], tensor_test_label))

    # Hub pretrained model
    x_train, x_test, y_train, y_test = train_test_split(
        raw['review'], raw['sentiment'], test_size=0.3, shuffle=True, random_state=1)

    for i in [x_train, x_test, y_train, y_test]:
        print(i.shape)

    y_train = y_train.agg(lambda x: 1 if x == 'positive' else 0)
    y_test = y_test.agg(lambda x: 1 if x == 'positive' else 0)

    x_train = tf.convert_to_tensor(x_train, dtype='string')
    y_train = tf.convert_to_tensor(y_train, dtype='int32')

    hub_clf = hub_model()
    hub_clf.fit(x_train, y_train, epochs=10, batch_size=512, validation_data=(x_test, y_test))
    