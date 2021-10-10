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

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FactorAnalysis
from sklearn.tree import DecisionTreeRegressor


from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import Dropout


from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class Antirectifier(layers.Layer):
    def __init__(self, initializer="he_normal", **kwargs):
        super(Antirectifier, self).__init__(**kwargs)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.output_dim = output_dim
        self.a1 = self.add_weight(
            shape=(1, output_dim),
            initializer=self.initializer,
            name="a1",
            trainable=True,
        )
        self.b1 = self.add_weight(
            shape=(1 , output_dim),
            initializer=self.initializer,
            name="b1",
            trainable=True,
        )
        self.a2 = self.add_weight(
            shape=(1, output_dim),
            initializer=self.initializer,
            name="a2",
            trainable=True,
        )
        self.b2 = self.add_weight(
            shape=(1 , output_dim),
            initializer=self.initializer,
            name="b2",
            trainable=True,
        )
        self.a3 = self.add_weight(
            shape=(1, output_dim),
            initializer=self.initializer,
            name="a3",
            trainable=True,
        )
        self.b3 = self.add_weight(
            shape=(1 , output_dim),
            initializer=self.initializer,
            name="b3",
            trainable=True,
        )
        self.a1 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)
        self.a2 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)
        self.a3 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)

        self.b1 = K.repeat_elements(K.expand_dims(10.0, axis = -1), self.output_dim, -1)
        self.b2 = K.repeat_elements(K.expand_dims(10.0, axis = -1), self.output_dim, -1)
        self.b3 = K.repeat_elements(K.expand_dims(10.0, axis = -1), self.output_dim, -1)
    
    # @tf.function
    def call(self, inputs):
        # inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
        x = inputs
        # x1 = 1/(1+tf.math.exp(-self.a1*(x+self.b1)))
        # x2 = tf.math.exp(-tf.math.pow(x-self.b2,2)/(2*tf.math.pow(self.a2,2)))
        # x3 = 1/(1-tf.math.exp(self.a3*(x+self.b3)))
        # print(x)
        # print(self.a1)
        
        low = [0.0 for x in range(0,10)]
        low = [low for x in range(0,10)]
        low = tf.convert_to_tensor(low,dtype = "float")
        # print(x1)
        low = tf.reshape(low,(1,10,10))

        x1 = (x- self.a1)/(self.b1 - self.a1)
        x2 = (x- self.b1)/(self.a1 - self.b1)
        x3 = (x- self.b3)/(2*self.a3 - self.b3)
        # return x1

        high = [10.0 for x in range(0,10)]
        high = [high for x in range(0,10)]
        high = tf.convert_to_tensor(high,dtype = "float")
        # print(x1)
        high = tf.reshape(high,(1,10,10))

        
        return tf.concat([x1,x2,x3],-1)

        if tf.math.less_equal(x,self.a1):
            x1 = low
        elif tf.math.greater(x,self.a1)  and tf.math.greater(self.b1,x):
            x1 = (x- self.a1)/(self.b1 - self.a1)
        else:
            x1 = high
          
        if tf.math.less_equal(x,self.a2):
            x2 = low
        elif tf.math.greater(x,self.a2)  and tf.math.greater(self.b2,x):
            x2 = (x- self.a2)/(self.b2 - self.a2)
        else:
            x2 = high

        if tf.math.less_equal(x,self.a3) or tf.math.less_equal(2*self.a3,x) :
            x3 = low
        elif tf.math.greater(x,self.a3)  and tf.math.greater(self.b3,x):
            x3 = (x- self.a3)/(self.b3 - self.a3)
        elif tf.math.greater(x,self.b3)  and tf.math.greater(2*self.a3,x):
            x3 = (x- self.b3)/(2*self.a3 - self.b3)
        elif x == 2*self.a3:
            x3 = high
        else:
            x3 = low

        # x1 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)

        # x2 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)

        # x3 = K.repeat_elements(K.expand_dims(0.0, axis = -1), self.output_dim, -1)

        # x1 = (x- self.a1)/(self.b1 - self.a1)

        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)

        # print("WTD")
        # print(x)
        # print(x[0][0][0])
        # quit()
        # print(x1)
        # print(tf.concat([x1,x2,x3],-1))
        # print(tf.concat([x1,x2,x3],-1).shape)
        # quit()
        # xc = inputs*aligned_a
        # print(tf.concat([x1,x2,x3],-1).shape)
        return tf.concat([x1,x2,x3],-1)
        # return x1


def build_RNN_model(inputs, output_size, neurons, activ_func="linear",
    dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(Antirectifier(input_shape=inputs.shape))

    model.add(Dense(128, activation='linear'))
    # model.add(DefuzzyLayer(1))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model


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

    attention_layer = Attention(32)(maxPooling_layer)

    # encoder = Dense(512, activation='relu')(attention_layer)
    # encoder = Dense(256, activation='relu')(encoder)
    encoder = Dense(128, activation='relu')(attention_layer)
    encoder = Dense(13, activation='linear')(encoder)

    model = Model(inputs=input_layer, outputs=encoder)

    model.compile(loss=loss,  # one may use 'mean_absolute_error' as  mean_squared_error
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]  # you can add several if needed
                  )
    model.summary()
    return model,attention_layer
# Python program to get average of a list
def Average(lst):
    return sum(lst) / len(lst)

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


    full_col =['pH', 'DO', 'BOD5', 'CODMn', 'TN', 'TP', 'TOC', 'DOC',
     'TN,', 'NH3-N', 'NO3-N', 'DTP', 'PO4-P']
    
    field = full_col
    # field = ["TN,", "DTP", "TN", "TP"]

    output_df = pd.read_excel("data/river_data.xlsx", sheet_name="Data(2018-2020)")
    print(output_df.columns)
    output = output_df[field].values

    train_dataset, test_dataset = split_data(input_data)
    input_data = np.array(input_data)
    output_train = np.array(output[:len(train_dataset)])
    output_test = np.array(output[len(train_dataset):len(input_data)])


    my_model,my_att = Build_cnn_regression_model((len(train_dataset[0]), len(train_dataset[0][0])), output_size=5, neurons=100)
    my_model.fit(train_dataset, output_train, epochs=5, batch_size=1)
    res = my_model.predict(test_dataset)

    # print(output)
    print(input_data.shape)
    print(output_test.shape)
    print(res.shape)
    # _pprint(field, output_test, res)


    print("______________________")
    att_output = K.function([my_model.layers[0].input],
                                      [my_model.layers[8].output])
    layer_output = att_output(input_data)[0]
    print(layer_output)

    print(len(layer_output))
    print(len(output))


    output_train = np.array(output[:len(train_dataset)])
    output_test = np.array(output[len(train_dataset):len(input_data)])



    train_dataset, test_dataset = split_data(layer_output)
    linear = []
    svr = []
    dtr = []
    fuzzy = []
    for col in full_col:
        print(col)
        output = output_df[col].values
        output_train = np.array(output[:len(train_dataset)])
        output_test = np.array(output[len(train_dataset):len(input_data)])
      

        regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        regr.fit(train_dataset,output_train)
        prediction = regr.predict(test_dataset)
        print(mpe(output_test,prediction))
        svr.append(mpe(output_test,prediction))

        my_model = build_RNN_model(train_dataset, output_size=1, neurons = 100)
        my_model.fit(train_dataset, output_train, 
            epochs=3, batch_size=1, verbose=1, shuffle=True)

        prediction = my_model.predict(test_dataset)
        print(mpe(output_test,prediction))
        fuzzy.append(mpe(output_test,prediction))
    print(Average(svr))
    print(Average(fuzzy))



        # break

if __name__ == '__main__':
    main()
