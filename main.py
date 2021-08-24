import csv
import numpy as np
import pandas as pd
import math
import os
from time import gmtime, strftime
import tensorflow as tf
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt


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


def build_cnn_regression_model(input_shape, output_size, neurons, activ_func="linear",
                               dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(5, activation='linear'))
    model.compile(loss=loss,  # one may use 'mean_absolute_error' as  mean_squared_error
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.RootMeanSquaredError(), "accuracy"]  # you can add several if needed
                  )
    model.summary()
    return model


def split_data(data: list, ratio=0.9):
    size = len(data)
    train_sample = int(size * ratio)
    train_dataset, test_dataset = data[: train_sample], data[train_sample:]
    return np.array(train_dataset), np.array(test_dataset)


def mse(label: np.ndarray, predict: np.ndarray):
    sum_square = (label - predict) ** 2
    return np.mean(sum_square)


def mpe(label: np.ndarray, predict: np.ndarray):
    sum_square = (label - predict) ** 2 / np.average(label) * 100
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


def visualize(history, epochs):
    loss_train = history.history['loss']
    accuracy = history.history['accuracy']
    epochs = range(1, epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Loss')
    plt.plot(epochs, accuracy, 'b', label='Accuracy')
    plt.title('Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('')
    plt.legend()
    plt.show()


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
                input_data.append(row)
    field = ["TN,", "DTP", "TN", "TP", "pH"]
    field_add = ["DO", "BOD5", "CODMn", "TOC", "DOC", "NH3-N", "NO3-N", "PO4-P"]
    # field  += field_add
    output_df = pd.read_excel("data/river_data.xlsx", sheet_name="Data(2018-2020)")
    output = output_df[field].values

    train_dataset, test_dataset = split_data(input_data)
    input_data = np.array(input_data)
    epochs = 5
    output_train = np.array(output[:len(train_dataset)])
    output_test = np.array(output[len(train_dataset):len(input_data)])
    my_model = build_cnn_regression_model((len(train_dataset[0]), len(train_dataset[0][0])), output_size=5, neurons=100)
    history = my_model.fit(train_dataset, output_train, epochs=epochs, batch_size=1)
    res = my_model.predict(test_dataset)
    _pprint(field, output_test, res)
    visualize(history, epochs)


if __name__ == '__main__':
    main()
