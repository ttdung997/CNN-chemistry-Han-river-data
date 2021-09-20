import csv
import numpy as np
import pandas as pd
import math
import os
from time import gmtime, strftime
import tensorflow as tf
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score, accuracy_score

from tensorflow.keras.models import Sequential,Model, model_from_json
from tensorflow.keras.layers import Activation, Dense, Conv1D, MaxPooling1D, Flatten,Permute,RepeatVector
from tensorflow.keras.layers import Dropout,Input
from tensorflow.keras.layers import Dense, Lambda, Dot, Activation, Concatenate
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt

class Attention(Layer):

    def __init__(self, units=128, **kwargs):
        self.units = units
        super().__init__(**kwargs)

    def __call__(self, inputs):
        """
        Many-to-one attention mechanism for Keras.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28, philipperemy.
        """
        hidden_states = inputs
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = Dot(axes=[1, 2], name='attention_score')([h_t, score_first_part])
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = Dot(axes=[1, 1], name='context_vector')([hidden_states, attention_weights])
        pre_activation = Concatenate(name='attention_output')([context_vector, h_t])
        attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

    def get_config(self):
        return {'units': self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def report_evaluation_metrics(y_true, y_pred):
    average_precision = average_precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    recall = recall_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    f1 = f1_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    acc = accuracy_score(y_true, y_pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Precision: {0:0.4f}'.format(precision))
    print('Recall: {0:0.4f}'.format(recall))
    print('F1: {0:0.4f}'.format(f1))
    print('acc: {0:0.4f}'.format(acc))


def get_data(dataset):
    data = []
    with open(dataset, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
        data = np.array(data)
    return data

# Attention Mechanism
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # values shape == (batch_size, max_len, hidden size)

    # we are doing this to broadcast addition along the time axis to calculate the score
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

def Build_seq_cnn_regression_model(input_shape, output_size, neurons, activ_func="linear",
                               dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(Conv1D(64, 5, activation='relu', input_shape=input_shape))

    model.add(MaxPooling1D(pool_size=2))

    # model.add(Attention(128))


    attentionModel = model.add(Attention(128))


    # model.add(Flatten())
    model.add(Dropout(0.1))


    model.add(Dense(512, activation='relu'))

    model.add(Dense(256, activation='relu'))


    model.add(Dense(128, activation='relu'))


    model.add(Dense(13, activation='linear'))

    model.compile(loss=loss,  # one may use 'mean_absolute_error' as  mean_squared_error
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]  # you can add several if needed
                  )
    model.summary()
    return model,attentionModel

def Build_cnn_regression_model(input_shape, output_size, neurons, activ_func="linear",
                               dropout=0.25, loss="mae", optimizer="adam"):
    input_layer = Input(input_shape)

    conv1D_layer = Conv1D(64, 5, activation='relu')(input_layer)

    maxPooling_layer = MaxPooling1D(pool_size=2)(conv1D_layer)

    attention_layer = Attention(128)(maxPooling_layer)

    encoder = Dense(512, activation='relu')(attention_layer)
    encoder = Dense(256, activation='relu')(encoder)
    encoder = Dense(128, activation='relu')(encoder)
    encoder = Dense(13, activation='linear')(encoder)

    model = Model(inputs=input_layer, outputs=encoder)

    model.compile(loss=loss,  # one may use 'mean_absolute_error' as  mean_squared_error
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]  # you can add several if needed
                  )
    model.summary()
    return model,attention_layer

def split_data(data: list, ratio=0.9):
    size = len(data)
    train_sample = int(size * ratio)
    train_dataset, test_dataset = data[: train_sample], data[train_sample:]
    return np.array(train_dataset), np.array(test_dataset)


def mse(label: np.ndarray, predict: np.ndarray):
    sum_square = (label - predict) ** 2
    return np.mean(sum_square)


def mpe(label: np.ndarray, predict: np.ndarray):
    sum_square = (label - predict) ** 2 / np.average(label)*100
    return np.mean(sum_square)

def _pprint(field, output_test, res):
    total_loss = 0
    for i in range(len(field)):
        loss = mse(output_test[:, i], res[:, i])
        total_loss += loss
        print(f"Field: {field[i]} Mse: {loss}")
    print(f"Total Loss MSE: {total_loss}")

    print("_____________________________")

    total_loss = 0
    for i in range(len(field)):
        loss = mpe(output_test[:, i], res[:, i])
        total_loss += loss
        print(f"Field: {field[i]} Mpe: {loss} %")

def main():
    # doc dũ liệu từ tệp ảnh
    folder_name = "data/EEM"
    col1 = 0
    col2 = 0
    files = [file for file in os.listdir(folder_name)]
    input_data = []
    for file in files:
        excel = pd.ExcelFile(os.path.join(folder_name, file))
        sheets = excel.sheet_names

        for sheet in sheets:
            if sheet != "18b":
                row = excel.parse(sheet_name=sheet).values
                # print(len(row))
                input_data.append(row)

    # các trường chính cần xét
    full_col =['pH', 'DO', 'BOD5', 'CODMn', 'TN', 'TP', 'TOC', 'DOC',
     'TN,', 'NH3-N', 'NO3-N', 'DTP', 'PO4-P']
    
    field = full_col


    # doc file ket quả (nong do cac chat)
    output_df = pd.read_excel("data/river_data.xlsx", sheet_name="Data(2018-2020)")
    print(output_df.columns)
    output = output_df[field].values

    # chia tap du lieu
    train_dataset, test_dataset = split_data(input_data)
    input_data = np.array(input_data)
    output_train = np.array(output[:len(train_dataset)])
    output_test = np.array(output[len(train_dataset):len(input_data)])


    # huan luyen mo hinh
    my_model,my_att = Build_cnn_regression_model((len(train_dataset[0]), len(train_dataset[0][0])), output_size=5, neurons=100)
    my_model.fit(train_dataset, output_train, epochs=5, batch_size=1)
    res = my_model.predict(test_dataset)

    #trich rut vector dac trung attention
    att_output = K.function([my_model.layers[0].input],
                                      [my_model.layers[8].output])
    layer_output = att_output(input_data)[0]


    # tinh do ruong quan
    res = np.concatenate((layer_output,output), axis=1)
    corr = np.corrcoef(np.transpose(res))
    final = []
    i = 0
    for line in corr:
        final.append(line[-13:])
        i = i + 1;
        if(i >128):
            break
    final = np.nan_to_num(np.array(final))

    # tong cac he so tuong quan
    total1 = np.sum(final,axis=0)  
    sort1 = (-total1).argsort()[:3]

    print("Các chỉ số ứng với vector attention")
    for i in sort1:
        print(full_col[i])

    total2 = np.sum(np.absolute(final),axis=0)  
    sort2 = (-total2).argsort()[:3]

    # tong cac he so tuong quan (tinh theo tri tuye doi)
    print("Các chỉ số ứng với vector attention (abs)")
    for i in sort2:
        print(full_col[i])


if __name__ == '__main__':
    main()
