import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime as time


def xgboost(dataset):
    print("#-----------------------------------------------------------------------------------------")
    # DataFlair - Read the data
    df = pd.read_csv("C:\\Users\CODEX\Downloads\PycharmProjects\pythonProject\Datasets\\" + dataset)
    print(" [ + ] dataset preview")
    # print(df)
    print(df.head())
    print("#-----------------------------------------------------------------------------------------")
    # DataFlair - Get the features and labels
    features = df.loc[:, df.columns != 'status'].values[:, 1:]
    labels = df.loc[:, 'status'].values
    print(" [ + ] one sample row of the features")
    print(features)
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
    # DataFlair - Split the dataset
    # random_state = Controls the shuffling applied to the data before applying the split.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)

    # DataFlair - Train the model
    model = XGBClassifier(use_label_encoder=False)
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


def dt(dataset):
    print("#-----------------------------------------------------------------------------------------")
    # DataFlair - Read the data
    df = pd.read_csv("C:\\Users\CODEX\Downloads\PycharmProjects\pythonProject\Datasets\\" + dataset)
    print(" [ + ] dataset preview")
    print(df.head())
    print("#-----------------------------------------------------------------------------------------")
    # DataFlair - Get the features and labels
    features = df.loc[:, df.columns != 'status'].values[:, 1:]
    labels = df.loc[:, 'status'].values
    print(" [ + ] one sample row of the features")
    print(features[0])
    # print(labels)
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
    # DataFlair - Split the dataset
    # random_state = Controls the shuffling applied to the data before applying the split.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)

    # DataFlair - Train the model
    model = DecisionTreeClassifier()
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


def rf(dataset):
    print("#-----------------------------------------------------------------------------------------")
    # DataFlair - Read the data
    df = pd.read_csv("C:\\Users\CODEX\Downloads\PycharmProjects\pythonProject\Datasets\\" + dataset)
    print(" [ + ] dataset preview")
    print(df.head())
    print("#-----------------------------------------------------------------------------------------")
    # DataFlair - Get the features and labels
    features = df.loc[:, df.columns != 'status'].values[:, 1:]
    labels = df.loc[:, 'status'].values
    print(" [ + ] one sample row of the features")
    print(features[0])
    # print(labels)
    print("#-----------------------------------------------------------------------------------------")
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
    # DataFlair - Split the dataset
    # random_state = Controls the shuffling applied to the data before applying the split.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)

    # DataFlair - Train the model
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


def xgb(dataset1):
    print("#-----------------------------------------------------------------------------------------")
    # "method to read our CSV data file."
    dataset = pd.read_csv("C:\\Users\CODEX\Downloads\PycharmProjects\pythonProject\Datasets\\" + dataset1)
    print("Execute the following command to see the number of rows and columns in our dataset:")
    print(dataset.shape)
    print("Execute the following command to inspect the first five records of the dataset:")
    print(dataset.head())
    # To divide data into attributes and labels, execute the following code
    # Here the X variable contains all the columns from the dataset,
    # except the "Class" column, which is the label.
    # The y variable contains the values from the "Class" column.
    # The X variable is our attribute set and y variable contains corresponding labels.
    X = dataset.drop('Class', axis=1)
    y = dataset['Class']
    from sklearn.model_selection import train_test_split
    # "we'll use to randomly split the data into training and testing sets. Execute the following code to do so"
    # " the test_size parameter specifies the ratio of the test set,
    # which we use to split up 20% of the data in to the test set and 80% for training."
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # "the final step is to train the decision tree algorithm on this data and make predictions."
    classifier = XGBClassifier()
    # "The fit method of this class is called to train the algorithm on the training data"
    classifier.fit(X_train, y_train)
    # "Now that our classifier has been trained, let's make predictions on the test data"
    y_pred = classifier.predict(X_test)
    # For classification problems the metrics used to evaluate an algorithm are accuracy,
    # confusion matrix, precision recall, and F1 values. Execute the following script to find these values:
    from sklearn.metrics import accuracy_score
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


def decision_tree(dataset1):
    print("#-----------------------------------------------------------------------------------------")
    # "method to read our CSV data file."
    dataset = pd.read_csv("C:\\Users\CODEX\Downloads\PycharmProjects\pythonProject\Datasets\\" + dataset1)
    print("Execute the following command to see the number of rows and columns in our dataset:")
    print(dataset.shape)
    print("Execute the following command to inspect the first five records of the dataset:")
    print(dataset.head())
    # To divide data into attributes and labels, execute the following code
    # Here the X variable contains all the columns from the dataset,
    # except the "Class" column, which is the label.
    # The y variable contains the values from the "Class" column.
    # The X variable is our attribute set and y variable contains corresponding labels.
    X = dataset.drop('Class', axis=1)
    y = dataset['Class']
    from sklearn.model_selection import train_test_split
    # "we'll use to randomly split the data into training and testing sets. Execute the following code to do so"
    # " the test_size parameter specifies the ratio of the test set,
    # which we use to split up 20% of the data in to the test set and 80% for training."
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    from sklearn.tree import DecisionTreeClassifier
    # "the final step is to train the decision tree algorithm on this data and make predictions."
    classifier = DecisionTreeClassifier()
    # "The fit method of this class is called to train the algorithm on the training data"
    classifier.fit(X_train, y_train)
    # "Now that our classifier has been trained, let's make predictions on the test data"
    y_pred = classifier.predict(X_test)
    # "Now we'll see how accurate our algorithm is"
    # For classification problems the metrics used to evaluate an algorithm are accuracy,
    # confusion matrix, precision recall, and F1 values. Execute the following script to find these values:
    from sklearn.metrics import accuracy_score

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


def random_forest(dataset1):
    # "Execute the following command to import the dataset:"
    dataset = pd.read_csv('C:\\Users\CODEX\Downloads\PycharmProjects\pythonProject\Datasets\\' + dataset1)
    print("To get a high-level view of what the dataset looks like, execute the following command:")
    print(dataset.head())
    # "Two tasks will be performed in this section. The first task is to divide data into 'attributes' and 'label' sets.
    # The resultant data is then divided into training and test sets"
    X = dataset.iloc[:, 0:4].values
    y = dataset.iloc[:, 4].values
    # "Finally, let's divide the data into training and testing sets:"
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # "We can see that the values in our dataset are not very well scaled.
    # We will scale them down before training the algorithm."
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)




    # n_estimators : int, default=100
    #         The number of trees in the forest.
    # max_depth : int, default=None
    #         The maximum depth of the tree. If None, then nodes are expanded until
    #         all leaves are pure or until all leaves contain less than
    #         min_samples_split samples.
    # min_samples_split : int or float, default=2
    #         The minimum number of samples required to split an internal node
    # min_samples_leaf : int or float, default=1
    #         The minimum number of samples required to be at a leaf node.
    # random_state : int or RandomState, default=None
    #         Controls both the randomness of the bootstrapping of the samples used
    #         when building trees (if ``bootstrap=True``) and the sampling of the
    #         features to consider when looking for the best split at each node
    #         (if ``max_features < n_features``).
    # "And again, now that we have scaled our dataset,
    # we can train our random forests to solve this classification problem. To do so,
    # execute the following code"
    # RandomForestClassifier class also takes n_estimators as a parameter
    # "Like before, this parameter defines the number of trees in our random forest.
    # We will start with 20 trees again."
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # For classification problems the metrics used to evaluate an algorithm are accuracy,
    # confusion matrix, precision recall, and F1 values. Execute the following script to find these values:
    from sklearn.metrics import accuracy_score

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
        print("\n [ + ] Choice Dataset")
        print(" [ 1 ] Parkinson’s Disease Dataset")
        print(" [ 2 ] Bill Authentication Dataset")
        select = input("Enter ... ")
        if select == '1':
            print("Parkinson’s Disease Dataset")
            print(" [ + ] XGBOOST Algorithm")
            time1 = time.now()
            xgboost("parkinsons.data")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in xgboost algorithm is : " + str(elapsed) + " ms ")
            input("Press any key to continue ... ")
            time1 = time.now()
            print(" [ + ] Decision Tree Algorithm")
            dt('parkinsons.data')
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in decision tree algorithm is : " + str(elapsed) + " ms ")
            input("Press any key to continue ... ")
            time1 = time.now()
            print(" [ + ] Random Forest Algorithm")
            rf('parkinsons.data')
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in random forest algorithm is : " + str(elapsed) + " ms ")
            print("Finish")
            input("Press any key to continue ... ")
        elif select == '2':
            print(" Dataset")
            print(" [ + ] XGBOOST Algorithm")
            time1 = time.now()
            xgb("bill_authentication.csv")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in xgboost algorithm is : " + str(elapsed) + " ms ")
            input("Press any key to continue ... ")
            print(" [ + ] Decision Tree Algorithm")
            time1 = time.now()
            decision_tree("bill_authentication.csv")
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in decision tree algorithm is : " + str(elapsed) + " ms ")
            input("Press any key to continue ... ")
            print(" [ + ] Random Forest Algorithm")
            time1 = time.now()
            random_forest('bill_authentication.csv')
            time2 = time.now()
            elapsed = int((time2 - time1).total_seconds() * 1000)
            print(" [ + ] Total Elapsed time in random forest algorithm is : " + str(elapsed) + " ms ")
            print("Finish")
            input("Press any key to continue ... ")
