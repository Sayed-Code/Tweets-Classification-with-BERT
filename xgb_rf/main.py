import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score,plot_confusion_matrix
from xgboost import XGBClassifier
import xgboost as xgb
from path_to_data import *


def get_training_data(year):
    if year == '2020':
        data_train = pd.read_excel(PATH_TRAINING_2020, engine='openpyxl')
        data_validation = pd.read_excel(PATH_VALIDATION_2020, engine='openpyxl')
    if year == '2019':
        data_train = pd.read_excel(PATH_TRAINING_2019, engine='openpyxl')
        data_validation = pd.read_excel(PATH_VALIDATION_2019, engine='openpyxl')
    if year == '2019-2020':
        data_train = pd.read_excel(PATH_TRAINING_2019_2020, engine='openpyxl')
        data_validation = pd.read_excel(PATH_VALIDATION_2019_2020, engine='openpyxl')
    return data_train, data_validation


def perpare_data_with_count_vectorizer(df):
    df_columns_0 = ['task2']

    cv = CountVectorizer(max_features=500, encoding="utf-8",
                         ngram_range=(1, 3),
                         token_pattern="[A-Za-z_][A-Za-z\d_]*")

    small_df0 = df.loc[:, df_columns_0]
    labelencoder = LabelEncoder()
    small_df0['task2'] = labelencoder.fit_transform(small_df0['task2'])

    small_df1 = pd.DataFrame(cv.fit_transform(
        df.text.values.astype('U')).toarray())

    df_total = pd.concat([small_df1, small_df0], axis=1)
    return df_total


def visualize_result(ytest, ypred):
    x_ax = range(len(ytest))
    plt.scatter(x_ax, ypred, color='red', alpha=0.5, label="predicted")
    plt.scatter(x_ax, ytest, color='blue', alpha=0.5, label="original")
    plt.title("Twitter impact test and predicted data")
    plt.legend()
    plt.show()


def plot_xgb_tree_and_feature_importance(xgb_model):
    xgb.plot_tree(xgb_model, num_trees=0)
    plt.rcParams['figure.figsize'] = [50, 10]
    plt.show()


def split_data_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=123)
    return X_train, X_test, y_train, y_test


def train_xgb_model(data, yr):

    if not data.empty:
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        X_train, X_test, y_train, y_test = split_data_train_test(X, y)

        params = {
            "training data time zone": yr,
            "training data size": X_train.shape,
            "testing data size": X_test.shape,
            "learning_rate": 0.1,
            "num estimators": 30,
            "max depth": 15,
            "reg lambda": 6,
            "random state": 123
        }


        xgb_class = XGBClassifier(random_state=123)
        xgb_class.set_params(n_estimators=30, learning_rate=0.1,
                             reg_lambda=6, max_depth=15)

        xgb_class.fit(X_train,y_train)
        plot_confusion_matrix(xgb_class, X_test, y_test)
        plt.show()

        predictions = xgb_class.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("[+] ----- Accuracy: %.2f%%" % (accuracy * 100.0))
        model_name =   f'/xgb_classification_model_{yr}'+ '.pickle.dat'

        # save model
        pickle.dump(xgb_class,open(PATH_SAVED_MODELS + model_name , "wb"))

        visualize_result(ytest=y_test, ypred=predictions)
        plot_xgb_tree_and_feature_importance(xgb_class)


def train_rf_model(data, yr):
    if not data.empty:
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        X_train, X_test, y_train, y_test = split_data_train_test(X, y)

        clf = RandomForestClassifier(n_estimators=500)
        clf.fit(X_train, y_train)

        plot_confusion_matrix(clf, X_test, y_test)
        plt.show()
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("[+] ----- Accuracy: %.2f%%" % (accuracy * 100.0))
        model_name =   f'/xgb_classification_rf_model_{yr}'+ '.pickle.dat'

        # save model
        pickle.dump(clf,open(PATH_SAVED_MODELS + model_name , "wb"))


def combine_data_vertically(data1, data2):
    data1.append(data2, ignore_index=True)
    return data1


def train(yr):
    data1, data2 = get_training_data(yr)
    data12 = combine_data_vertically(data1, data2)

    encoded_df = perpare_data_with_count_vectorizer(data12)
    train_xgb_model(encoded_df, yr)
    # train_rf_model(encoded_df, yr)


def get_test_data(year):
    if year == '2020':
        data_test = pd.read_excel(PATH_TESTING_2020, engine='openpyxl')
    if year == '2019':
        data_test = pd.read_excel(PATH_TESTING_2019, engine='openpyxl')
    if year == '2019-2020':
        data_test = pd.read_excel(PATH_TESTING_2019_2020, engine='openpyxl')
    return data_test


def test(yr):
    data = get_test_data(yr)
    encoded_df = perpare_data_with_count_vectorizer(data)
    if not encoded_df.empty:
        model_name =   f'/xgb_classification_model_{yr}'+ '.pickle.dat'
        loaded_model = pickle.load(
            open(PATH_SAVED_MODELS +model_name, "rb"))
        X, y = encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]
        y_predicted = loaded_model.predict(X)
        accuracy = accuracy_score(y, y_predicted)
        print("[+] ----- Testing Accuracy: %.2f%%" % (accuracy * 100.0))


def trained_on_test_on(param, param1):
    if param == '2019' and param1 == '2019':
        test_data = get_test_data(param1)
        encoded_df = perpare_data_with_count_vectorizer(test_data)
        model_name = f'/xgb_classification_rf_model_{param}'+ '.pickle.dat'

        loaded_model = pickle.load(
            open(PATH_SAVED_MODELS + model_name, "rb"))

        X, y = encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]
        y_predicted = loaded_model.predict(X)
        accuracy = accuracy_score(y, y_predicted)
        print("[+] ----- Testing Accuracy: %.2f%%" % (accuracy * 100.0))
    if param == '2019' and param1 == '2020':
        test_data = get_test_data(param1)
        encoded_df = perpare_data_with_count_vectorizer(test_data)
        model_name = f'/xgb_classification_rf_model_{param}' + '.pickle.dat'

        loaded_model = pickle.load(
            open(PATH_SAVED_MODELS + model_name, "rb"))

        X, y = encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]
        y_predicted = loaded_model.predict(X)
        accuracy = accuracy_score(y, y_predicted)
        print("[+] ----- Testing Accuracy: %.2f%%" % (accuracy * 100.0))
    if param == '2020' and param1 == '2020':
        test_data = get_test_data(param1)
        encoded_df = perpare_data_with_count_vectorizer(test_data)
        model_name = f'/xgb_classification_rf_model_{param}' + '.pickle.dat'

        loaded_model = pickle.load(
            open(PATH_SAVED_MODELS + model_name, "rb"))

        X, y = encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]
        y_predicted = loaded_model.predict(X)
        accuracy = accuracy_score(y, y_predicted)
        print("[+] ----- Testing Accuracy: %.2f%%" % (accuracy * 100.0))
    if param == '2020' and param1 == '2019':
        test_data = get_test_data(param1)
        encoded_df = perpare_data_with_count_vectorizer(test_data)
        model_name = f'/xgb_classification_rf_model_{param}' + '.pickle.dat'

        loaded_model = pickle.load(
            open(PATH_SAVED_MODELS + model_name, "rb"))

        X, y = encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]
        y_predicted = loaded_model.predict(X)
        accuracy = accuracy_score(y, y_predicted)
        print("[+] ----- Testing Accuracy: %.2f%%" % (accuracy * 100.0))
    if param == '2019-2020' and param1 == '2019-2020':
        test_data = get_test_data(param1)
        encoded_df = perpare_data_with_count_vectorizer(test_data)
        model_name = f'/xgb_classification_rf_model_{param}' + '.pickle.dat'

        loaded_model = pickle.load(
            open(PATH_SAVED_MODELS + model_name, "rb"))

        X, y = encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]
        y_predicted = loaded_model.predict(X)
        accuracy = accuracy_score(y, y_predicted)
        print("[+] ----- Testing Accuracy: %.2f%%" % (accuracy * 100.0))
    if param == '2019-2020' and param1 == '2019':
        test_data = get_test_data(param1)
        encoded_df = perpare_data_with_count_vectorizer(test_data)
        model_name = f'/xgb_classification_rf_model_{param}' + '.pickle.dat'

        loaded_model = pickle.load(
            open(PATH_SAVED_MODELS + model_name, "rb"))

        X, y = encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]
        y_predicted = loaded_model.predict(X)
        accuracy = accuracy_score(y, y_predicted)
        print("[+] ----- Testing Accuracy: %.2f%%" % (accuracy * 100.0))
    if param == '2019-2020' and param1 == '2020':
        test_data = get_test_data(param1)
        encoded_df = perpare_data_with_count_vectorizer(test_data)
        model_name = f'/xgb_classification_rf_model_{param}' + '.pickle.dat'

        loaded_model = pickle.load(
            open(PATH_SAVED_MODELS + model_name, "rb"))

        X, y = encoded_df.iloc[:, :-1], encoded_df.iloc[:, -1]
        y_predicted = loaded_model.predict(X)
        accuracy = accuracy_score(y, y_predicted)
        print("[+] ----- Testing Accuracy: %.2f%%" % (accuracy * 100.0))


if __name__ == '__main__':



    # train('2019') # 68.26%
    train('2020') # 79.12%
    train('2019-2020') # 74.84%

    # test('2019')
    # test('2020')
    # test('2019-2020')

    # trained_on_test_on('2019', '2019')
    # trained_on_test_on('2019', '2020')
    # trained_on_test_on('2020', '2020')
    # trained_on_test_on('2020', '2019')
    # trained_on_test_on('2019-2020', '2019')
    # trained_on_test_on('2019-2020', '2020')
    # trained_on_test_on('2019-2020', '2019-2020')