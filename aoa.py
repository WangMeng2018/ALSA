import argparse
import os
import jieba
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Softmax, Flatten, Dropout
from keras.layers import Bidirectional, Dense, Dot, Lambda, LeakyReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

jieba.setLogLevel(20)

senti_label_id = {'无关': 0, 0: 1, 1: 2, -1: 3}
aspect_label_id = {'动力': 0, '价格': 1, '内饰': 2, '配置': 3, '安全性': 4,
                   '外观': 5, '操控': 6, '油耗': 7, '空间': 8, '舒适性': 9}
N_SENTI, N_ASPECT = len(senti_label_id), len(aspect_label_id)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='./data/train.csv')
    parser.add_argument('--test_data_path', default='./data/test_public.csv')
    parser.add_argument('--embed_data_path', default='./output/embeddings_.p')
    parser.add_argument('--out_dir', default='./output')

    # model
    parser.add_argument('--max_sentence_length', default=256)
    parser.add_argument('--lstm_hidden_size', default=100)
    parser.add_argument('--dense_hidden_size', default=512)
    parser.add_argument('--leakyRelu_alpha', default=0.01)
    parser.add_argument('--drop_rate', default=0.5)
    parser.add_argument('--reg_rate', default=0.001)

    # train
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--lr_factor', default=0.65)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--epochs', default=30)
    args = parser.parse_args()
    return args


def my_model(opt, word_index, embedding_matrix):
    # sen = [batch, max_sentence_length]
    sen = Input(shape=(opt['max_sentence_length'],), name='Sentence')
    # asp = [1, N_ASPECT]
    asp = Lambda(
        lambda x: tf.constant(
            [[word_index[w] for w in aspect_label_id.keys()]]),
        name='Aspect')([])

    batch_size = K.shape(sen)[0]

    # Embedding module
    E = Embedding(
        *embedding_matrix.shape, trainable=False,
        embeddings_initializer=Constant(embedding_matrix),
        name='WordVec')

    # BiLSTM module
    # asen = [batch_size, max_sentence_len, 2*lstm_hidden_size]
    asen = Bidirectional(
        LSTM(opt['lstm_hidden_size'], return_sequences=True,
             dropout=opt['drop_rate'], recurrent_dropout=opt['drop_rate']),
        name='BLSTM-Sen'
    )(E(sen))
    # aasp = [1, N_ASPECT, 2*lstm_hidden_size]
    aasp = Bidirectional(
        LSTM(opt['lstm_hidden_size'], return_sequences=True,
             dropout=opt['drop_rate'], recurrent_dropout=opt['drop_rate']),
        name='BLSTM-Asp'
    )(E(asp))
    # aasp = [batch, N_ASPECT, 2*lstm_hidden_size]
    aasp = Lambda(
        lambda attn: tf.reshape(
            tf.tile(tf.reshape(attn, (-1,)), [batch_size]),
            (batch_size, N_ASPECT, 2 * opt['lstm_hidden_size'])),
        name='Repeat'
    )(aasp)

    # AOA module
    # X = [batch_size, max_sentence_len, N_ASPECT]
    X = Dot(-1, name='Project')([asen, aasp])
    # attn = [batch_size, max_sentence_len, N_ASPECT]
    attn = Softmax(1, name='Within-Aspect')(X)  # column-wise-softmax
    # X = [batch_size, N_ASPECT, 2*lstm_hidden_size]
    X = Dot(1, name='Attention')([attn, asen])
    X = Dropout(opt['drop_rate'], name='Dropout')(X)
    # X = [batch_size, N_ASPECT * 2 * lstm_hidden_size]
    X = Flatten(name='Flatten')(X)

    # Prediction module
    # X = [batch, dense_hidden_size]
    X = Dense(opt['dense_hidden_size'],
              kernel_regularizer=regularizers.l2(opt['reg_rate']),
              name='Asp-Senti-Clf-1')(X)
    X = LeakyReLU(alpha=opt['leakyRelu_alpha'], name='LeakyReLU')(X)
    # X = [batch, N_SENTI * N_ASPECT]
    X = Dense(N_SENTI * N_ASPECT,
              kernel_regularizer=regularizers.l2(opt['reg_rate']),
              name='Asp-Senti-Clf-2')(X)
    # X = [batch, N_ASPECT, N_SENTI]
    X = Lambda(
        lambda x: tf.reshape(
            K.softmax(tf.reshape(x, (batch_size, N_ASPECT, N_SENTI))),
            (batch_size, N_SENTI * N_ASPECT)),
        name='Aspect-Softmax'
    )(X)

    return Model(inputs=sen, outputs=X)


def get_training_data(opt, word_index):
    X_y = defaultdict(lambda: [1, 0, 0, 0] * N_ASPECT)
    for e in pd.read_csv(opt['train_data_path']).itertuples():
        i = aspect_label_id[e.subject] * N_SENTI
        X_y[e.content][i] = 0
        X_y[e.content][i + senti_label_id[e.sentiment_value]] = 1
    X, y = [], []
    for content, senti in X_y.items():
        X.append([word_index[w] for w in jieba.lcut_for_search(
            content) if w in word_index.keys()])
        y.append(senti)
    return pad_sequences(X, maxlen=opt['max_sentence_length']), np.array(y)


def reshape_senti(y):
    return K.reshape(y, (-1, N_SENTI))


def reshape_aspect(y):
    return K.reshape(y, (-1, N_ASPECT))


def to_class_label(y):
    """Four probabilities to most likely class."""
    return K.argmax(reshape_senti(y))


def cate_loss(y_true, y_pred):
    """Sum of cross entropy loss for all aspects."""
    y_true, y_pred = reshape_senti(y_true), reshape_senti(y_pred)
    loss = reshape_aspect(K.categorical_crossentropy(y_true, y_pred))
    return K.sum(loss, 1)


def acc(y_true, y_pred):
    """Sentiment-wise accuracy."""
    y_true, y_pred = to_class_label(y_true), to_class_label(y_pred)
    same = K.cast(K.equal(y_true, y_pred), K.floatx())
    size = K.cast(K.shape(y_true)[0], K.floatx())
    return K.sum(same) / size


def f1(y_true, y_pred):
    """Task-required F1 score."""
    y_true, y_pred = to_class_label(y_true), to_class_label(y_pred)
    zeros = K.zeros_like(y_true)

    true_not_zero = K.cast(K.not_equal(y_true, zeros), K.floatx())
    true_zero = K.cast(K.equal(y_true, zeros), K.floatx())
    pred_not_zero = K.cast(K.not_equal(y_pred, zeros), K.floatx())
    pred_zero = K.cast(K.equal(y_pred, zeros), K.floatx())
    pred_not_true = K.cast(K.not_equal(y_true, y_pred), K.floatx())
    pred_true = K.cast(K.equal(y_true, y_pred), K.floatx())

    tp = K.sum(true_not_zero * pred_not_zero * pred_true + K.epsilon())  # 判断正确
    fp = K.sum(true_not_zero * pred_not_zero * pred_not_true +  # 错判
               true_zero * pred_not_zero + K.epsilon())   # 多判
    fn = K.sum(true_not_zero * pred_zero + K.epsilon())   # 漏判

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    return f1


def main():
    args = set_args()
    opt = vars(args)

    # load embed
    word_index, embedding_matrix = pickle.load(
        open(opt['embed_data_path'], 'rb'))

    # get data
    X, y = get_training_data(opt, word_index)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=666)
    X_test, X_val, y_test, y_val = train_test_split(
        X_val, y_val, test_size=0.5, random_state=233)
    print('train data size = {}, val_data_size = {}, test_data_size = {}'.format(
        len(y_train), len(y_val), len(y_test)))  # 8523 1066 1065

    # model
    model = my_model(opt, word_index, embedding_matrix)
    model.summary()
    model.compile(loss=cate_loss, optimizer=opt['optimizer'], metrics=[acc, f1])

    # train
    checkpointer = ModelCheckpoint(
        os.path.join(opt['out_dir'], 'weights.hdf5'), 'val_f1',
        mode='max', verbose=1, save_best_only=True, save_weights_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    lrreducer = ReduceLROnPlateau(
        monitor='val_loss', factor=opt['lr_factor'], patience=3, verbose=1)

    model.fit(X_train, y_train,
              batch_size=opt['batch_size'], epochs=opt['epochs'],
              validation_data=(X_val, y_val),
              callbacks=[checkpointer, earlystopper, lrreducer])

    # evaluate
    model.load_weights(os.path.join(opt['out_dir'], 'weights.hdf5'))
    print(model.evaluate(X_test, y_test))
    print('done!')


if __name__ == "__main__":
    main()
