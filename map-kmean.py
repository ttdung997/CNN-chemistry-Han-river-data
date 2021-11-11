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

import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FactorAnalysis
from sklearn.tree import DecisionTreeRegressor


from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

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

import numpy as np

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



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

    for file in files:
        print("_____________________________")
        print(os.path.join(folder_name, file))
        excel = pd.ExcelFile(os.path.join(folder_name, file))
        sheets = excel.sheet_names
        avg_file  = []
        for sheet in sheets:
            # print("new sheet")
            if sheet != "18b":
                v1 = []
                v2 = []
                v3 = []
                v4 = []
                v5 = []



                row = excel.parse(sheet_name=sheet).values
                
                rdata = np.array(row)
                rdata = rdata.flatten()

                # print("show flatter") 
                # sns.set_style("darkgrid")
                # plt.plot(np.array(rdata))
                # # fig = plt.get_figure()

                column_sums = [sum([x[i] for x in row]) for i in range(0,len(row[0]))]
                column_sums = NormalizeData(column_sums)


                row_sums = [sum(x) for x in row]
                row_sums = NormalizeData(row_sums)


                fig=plt.figure()
                ax=fig.add_subplot(111)

                ax.plot([i+200 for i in range(0, len(column_sums))],column_sums, label = "Excitation (mm)")
                ax.plot([i+200 for i in range(0, len(row_sums))],row_sums,  label = "Emission (mm)")
                ax.legend()

                ax.set_xlim(200, 500)
                ax.set_ylim(0, 1.2)
                # continue


                fig.savefig('Heatmap/distribute/'+file+"-"+sheet+".png")
                fig.clf()

                # continue

                data1 = rdata[3400:4600]
                data2 = rdata[4800:7200]


                data1.sort()
                print(data1)
                data2.sort()

                l = len(rdata)
                # print(l)
                rdata.sort()
                X = []
                for point in rdata:
                    X.append([0,point])
                kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
                peak2D = kmeans.cluster_centers_

                closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
                # print(closest)



                peak = []
                for index in closest:
                    peak.append(rdata[index])
                peak.sort()
                # print(peak)
                # quit()

                test =  peak[-1]

             
                avg_matrix = []
                print("begin loop: ")
                f = open("peak/"+file+"-"+str(sheet)+".txt", "w")
                for i in range(0,len(row)):
                    for j in range(0,len(row[i])):
                        flag = 0
                        if row[i][j] == peak[-1]:
                            print("PEAK 1")
                            f.write("Peak 1: ["+str(i) +", " +str(j) +"]\n")
                            print(row[i][j])
                            flag = peak[-1]
                            test = flag
                            for d1 in range(0,len(row)):
                                    for d2 in range(0,len(row[d1])):
                                        if (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 10:
                                            row[d1][d2]= test*1.2
                                        elif (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 20:
                                            row[d1][d2]= test*1.05
                                        elif (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 30:
                                            row[d1][d2]= test*0.8
                        if row[i][j] == peak[-2]:
                            print("PEAK 2")
                            f.write("Peak 2: ["+str(i) +", " +str(j) +"]\n")
                            print(row[i][j])
                            flag = peak[-2]
                            test = flag
                            for d1 in range(0,len(row)):
                                    for d2 in range(0,len(row[d1])):
                                        if (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 10:
                                            row[d1][d2]= test*1.2
                                        elif (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 20:
                                            row[d1][d2]= test*1.05
                                        elif (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 30:
                                            row[d1][d2]= test*0.8
                        if row[i][j] == peak[-3]:
                            print("PEAK 3")
                            f.write("Peak 3: ["+str(i) +", " +str(j) +"]\n")
                            print(row[i][j])
                            flag = peak[-3]
                            test = flag
                            for d1 in range(0,len(row)):
                                    for d2 in range(0,len(row[d1])):
                                        if (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 10:
                                            row[d1][d2]= test*1.2
                                        elif (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 20:
                                            row[d1][d2]= test*1.05
                                        elif (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 30:
                                            row[d1][d2]= test*0.8
                        if row[i][j] == peak[-4]:
                            print("PEAK 4")
                            f.write("Peak 4: ["+str(i) +", " +str(j) +"]\n")
                            print(row[i][j])
                            flag = peak[-4]
                            test = flag
                            for d1 in range(0,len(row)):
                                    for d2 in range(0,len(row[d1])):
                                        if (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 10:
                                            row[d1][d2]= test*1.2
                                        elif (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 20:
                                            row[d1][d2]= test*1.05
                                        elif (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 30:
                                            row[d1][d2]= test*0.8
                        if row[i][j] ==  peak[-5]:
                            print("PEAK 5")
                            f.write("Peak 5: ["+str(i) +", " +str(j) +"]\n")
                            print(row[i][j])
                            flag = peak[-5]
                            test = flag
                            for d1 in range(0,len(row)):
                                    for d2 in range(0,len(row[d1])):
                                        if (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 10:
                                            row[d1][d2]= test*1.2
                                        elif (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 20:
                                            row[d1][d2]= test*1.05
                                        elif (d1-i)*(d1-i)/25+ (d2-j)*(d2-j) < 30:
                                            row[d1][d2]= test*0.8
                        # if(flag > 0):
                        #     print("_____________________________")
                        #     print(rdata[-1])
                        #     print(rdata[-100])
                        #     print(rdata[-200])
                        #     print(rdata[-300])
                        #     print(rdata[-400])
                        #     print("*******")
                        #     print(test)
                        #     print("_____________________________")


                        # if flag > 0:
                        #     print("okoek")
                        #     # print(flag)
                        #     try:
                        #         atem = flag
                        #         for d1 in range(0,len(row)):
                        #             for d2 in range(0,len(row[d1])):
                        #                 if (d1-i)*(d1-i)/16+ (d2-j)*(d2-j) < 10:
                        #                     row[d1][d2]= test + test*0.1*(10- (d1-i)*(d1-i)/16+ (d2-j)*(d2-j))
                        #         # row[i-6][j] = atem*1.3
                        #         # row[i-5][j] = atem*1.3
                        #         # row[i-4][j] = atem*1.3
                        #         # row[i-3][j] = atem*1.3
                        #         # row[i-2][j] = atem*1.3
                        #         # row[i-1][j-1] = atem*1.3
                        #         # row[i-1][j] = atem*1.3
                        #         # row[i-1][j+1] = atem*1.3
                        #         # row[i-1][j] = atem*1.3
                        #         # row[i][j] = atem*1.3
                        #         # row[i+1][j] = atem*1.3
                        #         # row[i+1][j-1] = atem*1.3
                        #         # row[i+1][j] = atem*1.3
                        #         # row[i+1][j+1] = atem*1.3
                        #         # row[i+2][j] = atem*1.3
                        #         # row[i+3][j] = atem*1.3
                        #         # row[i+4][j] = atem*1.3
                        #         # row[i+5][j] = atem*1.3
                        #         # row[i+6][j] = atem*1.3
                        #     except:
                        #         continue
                        # else:
                            # row[i][j] = 0


                # scaler = MinMaxScaler()
                # scaler.fit(row)
                # heat = scaler.transform(row)

                f.close()
                heat = []
                # print()
                for i in range(len(row)-1,0,-1):
                    heat.append(row[i])

                heat = np.array(heat)

                print(heat.shape)

                # yticklabels = range(200, 500, 20)
                # # the index position of the tick labels
                # yticks = []
                # for label in yticklabels:
                #     idx_pos = df.index.get_loc(label)
                #     yticks.append(idx_pos)

               
                column_sums = [sum([x[i] for x in row]) for i in range(0,len(row[0]))]
                column_sums = NormalizeData(column_sums)


                row_sums = [sum(x) for x in row]
                row_sums = NormalizeData(row_sums)


                fig=plt.figure()
                ax=fig.add_subplot(111)

                ax.plot([i+200 for i in range(0, len(column_sums))],column_sums, label = "Excitation (mm)")
                ax.plot([i+200 for i in range(0, len(row_sums))],row_sums,  label = "Emission (mm)")
                ax.legend()

                ax.set_xlim(200, 500)
                ax.set_ylim(0, 1.2)
                # continue


                fig.savefig('Heatmap/distribute2/'+file+"-"+sheet+".png")
                fig.clf()


                # x_labels = [200,250,300,350,400,450,500] # Labels for x-axis
                # y_labels = [500,450,400,350,300,250,200] # Labels for y-axis
                # hmap  = sns.heatmap(heat, cmap = "mako")
                
                # # hmap = sns.heatmap(heat, yticklabels=y_labels)
                # hmap.set_ylabel('Excitation (mm)')
                # hmap.set_xlabel('Emission (mm)')

                # yticks = np.linspace(0, len(heat),7 , dtype=np.int)
                # hmap.set_yticks(yticks)
                # hmap.set_yticklabels(y_labels)


                # xticks = np.linspace(0, len(heat[0]),7 , dtype=np.int)
                # hmap.set_xticks(xticks)
                # hmap.set_xticklabels(x_labels)


                # # plt.title('Heatmap/pp2/'+file+"-"+sheet+".png");
                # fig = hmap.get_figure()
                # fig.savefig('Heatmap/pp3/'+file+"-"+sheet+".png")
                
                # plt.clf()

if __name__ == '__main__':
    main()
