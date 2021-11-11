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

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FactorAnalysis
from sklearn.tree import DecisionTreeRegressor


from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    # accuracy = history.history['root_mean_squared_error']
    epochs = range(1, epochs + 1)
    plt.plot(epochs, loss_train, 'g', label='Loss')
    # plt.plot(epochs, accuracy, 'b', label='root_mean_squared_error')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('')
    plt.legend()
    plt.show()

def Average(lst):
    return sum(lst) / len(lst)


def main():
    # doc file du lieu va bieu dien duoi dang heatmap
    folder_name = "data/EEM"
    col1 = 0
    col2 = 0
    files = [file for file in os.listdir(folder_name)]
    input_data = []
    atotal = []
    
    dv1 = []
    dv2 = []
    dv3 = []
    dv4 = []
    dv5 = []
    dvd = []
    dvd2 = []
    dvr = []

    for file in files:
        print("_____________________________")
        print(os.path.join(folder_name, file))
        excel = pd.ExcelFile(os.path.join(folder_name, file))
        sheets = excel.sheet_names
        avg_file  = []
        for sheet in sheets:
            if sheet != "18b":
                v1 = []
                v2 = []
                v3 = []
                v4 = []
                v5 = []

                thr = []


                row = excel.parse(sheet_name=sheet).values
                
                rdata = np.array(row)
                rdata = rdata.flatten()
                
                print(rdata)
                data1 = rdata[3400:4600]
                data2 = rdata[4800:7200]

                data1.sort()
                data2.sort()

                
                # print("how it gone")
                # print(data1)
                test1 = (data1[-1])
                test2 = (data2[-1])

                test3 = (rdata[-3])
                test4 = (rdata[-4])
                test5 = (rdata[-5])
                # print("how it be")

                rdata.sort()
                cut_point = int(0.95 * len(rdata))
                threshold = rdata[cut_point]
                
                threshold_array =[x for x in rdata   if x > threshold ]
                threshold_tolal = 0
                for x in threshold_array:
                    threshold_tolal = threshold_tolal +x

                thr.append(threshold_tolal)

                avg_matrix = []
                for i in range(0,len(row)):
                    for j in range(0,len(row[i])):
                        if row[i][j] == test1   and len(v1) < 1:
                            print("??????")
                            v1.append(row[i][j])
                            print(v1)
                            print(test1)
                            # print(data2[-1])
                            print(row[i][j])
                        elif row[i][j] == test2  and len(v2) < 1:
                            v2.append(row[i][j])
                        elif row[i][j] == test3  and len(v3) < 1:
                            v3.append(row[i][j])
                        elif row[i][j] == test4  and len(v4) < 1:
                            v4.append(row[i][j])
                        elif row[i][j] == test5 and len(v5) < 1:
                            v5.append(row[i][j])


                dv1.append(v1)
                dv2.append(v2)
                dv3.append(v3)
                dv4.append(v4)
                dv5.append(v5)

                dvr.append(thr)
                for i in range(0,len(v1)):
                    dvd.append([v1[i],v2[i]])
                    dvd2.append([v1[i],v2[i],thr[i]])

                input_data.append(row)

                # print("___________________________________")
                # print(avg_matrix)

    dv = []
    dv.append(dvr)
    dv.append(dv1)
    dv.append(dv2)
    dv.append(dv3)
    dv.append(dv4)
    dv.append(dv5)
    dv.append(dvd)
    dv.append(dvd2)


    dvt = []
    dvt.append("dvr")
    dvt.append("dv1")
    dvt.append("dv2")
    dvt.append("dv3")
    dvt.append("dv4")
    dvt.append("dv5")
    dvt.append("dvd")
    dvt.append("dvd2")


    full_col =['pH', 'DO', 'BOD5', 'CODMn', 'TN', 'TP', 'TOC', 'DOC',
     'TN,', 'NH3-N', 'NO3-N', 'DTP', 'PO4-P']
    
    field = full_col
    # field = ["TN,", "DTP", "TN", "TP"]

    output_df = pd.read_excel("data/river_data.xlsx", sheet_name="Data(2018-2020)")
    print(output_df.columns)
    output = output_df[field].values

    i = 1
    dv_count = 0
    for dataset in dv:
        print(dvt[dv_count])
        dv_count = dv_count  +1
        try:
            print("**********************")
            print("DV "+str(i))
            i = i+1
            print("**********************")
            dataset = np.array(dataset)
            # print(dataset)
            train_dataset, test_dataset = split_data(dataset)
            input_data = np.array(input_data)


            linear = []
            svr = []
            dtr = []

            for col in full_col:
                output = output_df[col].values
                output_train = np.array(output[:len(train_dataset)])
                output_test = np.array(output[len(train_dataset):len(dataset)])
                
                # print(train_dataset.shape)
                # print(output_train.shape)


                model = LinearRegression()  

                # print("*************")
                # print(train_dataset)
                # print(output_train)
                # print("*************")

                model.fit(train_dataset,output_train)
                # print(output_train)
                prediction = model.predict(test_dataset)
                # print(col)
                # print("*************")
                # # print("linear")
                # print(output_test)
                # print(prediction)
                # print("????????")
                # continue
                a = (mpe(output_test,prediction))
                linear.append(mpe(output_test,prediction))

                # a = (r2_score(output_test,prediction))
                # linear.append(r2_score(output_test,prediction))

                # if a < -1:
                #     a = 0
                # else:
                #     a = a+ 0.5

                # print("SVR")

                regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
                regr.fit(train_dataset,output_train)
                prediction = regr.predict(test_dataset)

                b = (mpe(output_test,prediction))
                svr.append(mpe(output_test,prediction))

                # b = (r2_score(output_test,prediction))
                # svr.append(r2_score(output_test,prediction))

                # if b < -1:
                #     b = 0
                # else:
                #     b = a+ 0.5
                # print("DTR")

                regr = DecisionTreeRegressor(random_state=0)
                regr.fit(train_dataset,output_train)
                prediction = regr.predict(test_dataset)
                
                c = (mpe(output_test,prediction))
                dtr.append(mpe(output_test,prediction))



                # c = (r2_score(output_test,prediction))
                # dtr.append(r2_score(output_test,prediction))


                # if c < -1:
                #     c = 0
                # else:
                #     c = a+ 0.5
                # print("_____________________________")
                print(col+","+str(round(a,2))+","+str(round(b,2))+","+str(round(c,2)))

        except:
            print("wtd")
            continue
        print("final")
        print("linear")
        print(Average(linear))
        print("SVR")
        print(Average(svr))
        print("DTR")
        print(Average(dtr))



if __name__ == '__main__':
    main()

