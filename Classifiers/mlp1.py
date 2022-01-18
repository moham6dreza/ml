import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime as time


class Model:
    def build_data(self, dataset_name, dataset_num):
        print("#-----------------------------------------------------------------------------------------")
        # DataFlair - Read the data
        dataset_location = "C:\\Users\CODEX\Downloads\PycharmProjects\pythonProject\Datasets\\"
        df = pd.read_csv(dataset_location + dataset_name)
        row_count = df.shape[0]
        col_count = df.shape[1]
        print(" [ + ] dataset preview")
        print(df)
        # print(df.head())
        print("#-----------------------------------------------------------------------------------------")
        # DataFlair - Get the features and labels
        if dataset_num == '1':
            features = df.loc[:, df.columns != 'status'].values[:, 1:]
            labels = df.loc[:, 'status'].values
        elif dataset_num == '2':
            features = df.iloc[:, :col_count - 1].values
            labels = df.iloc[:, col_count - 1].values
        elif dataset_num == '3':
            features = df.iloc[:, :col_count - 1].values
            labels = df.iloc[:, col_count - 1].values
        print(" [ + ] one sample row of the features[0] array(1D) and label array(1D)")
        # print(len(features[0]))
        print(features[0])
        print(labels)
        print("#-----------------------------------------------------------------------------------------")
        print("Execute the following command to see the number of rows and columns in our dataset:")
        print(df.shape)
        # DataFlair - Get the count of each label (0 and 1) in labels
        print(" [ + ] Get the count of each label (1 and 0) in labels")
        print(labels[labels == 1].shape[0], labels[labels == 0].shape[0])

        # DataFlair - Scale the features to between -1 and 1
        scaler = MinMaxScaler((-1, 1))
        x = scaler.fit_transform(features)
        y = labels
        print("#-----------------------------------------------------------------------------------------")
        print(" [ + ] Scale the features to between -1 and 1")
        print(x[0])
        return x, y

    def mine(self, x, y, algorithm):

        # DataFlair - Split the dataset
        # random_state = Controls the shuffling applied to the data before applying the split.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)

        # DataFlair - Train the model
        if algorithm == "xgb":
            model = XGBClassifier(use_label_encoder=False)
        elif algorithm == "dt":
            model = DecisionTreeClassifier()
        elif algorithm == "rf":
            model = RandomForestClassifier()
        model.fit(x_train, y_train)

        # DataFlair - Calculate the accuracy
        y_pred = model.predict(x_test)
        print("#-----------------------------------------------------------------------------------------")
        print(" [ + ] Calculate the accuracy for test data")
        print(accuracy_score(y_test, y_pred) * 100)
        from sklearn.metrics import classification_report, confusion_matrix
        print("#-----------------------------------------------------------------------------------------")
        from sklearn import metrics
        print(" [ + ] Error Metrics")
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print("#-----------------------------------------------------------------------------------------")
        print(" [ + ] confusion_matrix")
        print(confusion_matrix(y_test, y_pred))
        print("#-----------------------------------------------------------------------------------------")
        print(" [ + ] classification_report")
        print(classification_report(y_test, y_pred))
        print("#-----------------------------------------------------------------------------------------")


if __name__ == '__main__':
    while True:
        print("\n\t[ + ] Choice Dataset")
        print("\n\n\t\t[ 1 ] Spambase Dataset")
        print("\t\t[ 2 ] Parkinson’s Disease Dataset")
        print("\t\t[ 3 ] Bill Authentication Dataset")
        select = input("\n\n\tEnter ... ")
        if select == '3':
            print("Bill Authentication Dataset")
            l = Model()
            x, y = l.build_data("bill_authentication.csv", '3')
            print(" [ + ] XGBOOST Algorithm")
            time1 = time.now()
            l.mine(x, y, "xgb")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in xgboost algorithm is : " + str(elapsed) + " ms ")
            input("Press any key to continue ... ")
            print(" [ + ] Decision Tree Algorithm")
            time1 = time.now()
            l.mine(x, y, "dt")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in decision tree algorithm is : " + str(elapsed) + " ms ")
            input("Press any key to continue ... ")
            print(" [ + ] Random Forest Algorithm")
            time1 = time.now()
            l.mine(x, y, "rf")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in random forest algorithm is : " + str(elapsed) + " ms ")
            print("Finish")
            input("Press any key to continue ... ")
        elif select == '2':
            print("Parkinson’s Disease Dataset")
            l = Model()
            x, y = l.build_data("parkinsons.data", '1')
            print(" [ + ] XGBOOST Algorithm")
            time1 = time.now()
            l.mine(x, y, "xgb")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in xgboost algorithm is : " + str(elapsed) + " ms ")
            input("Press any key to continue ... ")
            time1 = time.now()
            print(" [ + ] Decision Tree Algorithm")
            l.mine(x, y, "dt")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in decision tree algorithm is : " + str(elapsed) + " ms ")
            input("Press any key to continue ... ")
            time1 = time.now()
            print(" [ + ] Random Forest Algorithm")
            l.mine(x, y, "rf")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in random forest algorithm is : " + str(elapsed) + " ms ")
            print("Finish")
            input("Press any key to continue ... ")
        elif select == '1':
            print("Spambase Dataset")
            i = Model()
            x, y = i.build_data("spambase.data", '2')
            print(" [ + ] XGBOOST Algorithm")
            time1 = time.now()
            i.mine(x, y, "xgb")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in xgboost algorithm is : " + str(elapsed) + " ms ")
            input("Press any key to continue ... ")
            time1 = time.now()
            print(" [ + ] Decision Tree Algorithm")
            i.mine(x, y, "dt")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in decision tree algorithm is : " + str(elapsed) + " ms ")
            input("Press any key to continue ... ")
            time1 = time.now()
            print(" [ + ] Random Forest Algorithm")
            i.mine(x, y, "rf")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in random forest algorithm is : " + str(elapsed) + " ms ")
            print("Finish")
            input("Press any key to continue ... ")
