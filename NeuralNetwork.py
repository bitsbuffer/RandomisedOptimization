import mlrose_hiive as mlrose
import pandas as pd
import numpy as np
import pickle
from itertools import product
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def calculate_metrics(clf, test, y_test):
    yhat_test = clf.predict(test)
    _roc_auc_score = roc_auc_score(y_test, yhat_test)
    _f1_score = f1_score(y_test, yhat_test)
    _acc = accuracy_score(y_test, yhat_test)
    print("********************************************************")
    print("Accuracy on Test data ", _acc)
    print("********************************************************")
    print("F1 on Test data ", _f1_score)
    print("********************************************************")
    print("AUC ROC on Test data ", _roc_auc_score)
    print("********************************************************")
    print("Confusion matrix \n", confusion_matrix(y_test, yhat_test))
    print("********************************************************")
    print(classification_report(y_test, yhat_test))
    print("********************************************************")
    return _acc, _roc_auc_score, _f1_score


def nn_rhc(x_train, y_train, x_test, y_test):
    print("********************Executing Randomized Hill Climbing****************")
    max_attempts = [50, 100]
    learning_rates = [0.001, 0.005, 0.01, 0.1]
    train_acc_s = []
    test_acc_s = []
    train_f1_s = []
    test_f1_s = []
    train_roc_s = []
    test_roc_s = []
    for lr, ma in product(learning_rates, max_attempts):
        print(f"Running for learning rate {lr} and max attempt {ma}")
        nn_model = mlrose.NeuralNetwork(hidden_nodes=[16, 8],
                                        activation='relu',
                                        algorithm='random_hill_climb',
                                        max_iters=1000,
                                        bias=True,
                                        is_classifier=True,
                                        learning_rate=lr,
                                        early_stopping=True,
                                        clip_max=5,
                                        restarts=2,
                                        max_attempts=ma,
                                        random_state=1234)
        nn_model.fit(x_train, y_train)
        train_acc, train_roc_auc, train_f1 = calculate_metrics(nn_model, x_train, y_train)
        test_acc, test_roc_auc, test_f1 = calculate_metrics(nn_model, x_test, y_test)
        train_acc_s.append(train_acc)
        train_roc_s.append(train_roc_auc)
        train_f1_s.append(train_f1)
        test_acc_s.append(test_acc)
        test_roc_s.append(test_roc_auc)
        test_f1_s.append(test_f1)
    obj = {
        "max_attempts": max_attempts,
        "lr": learning_rates,
        "train_acc": train_acc_s,
        "test_acc": test_acc_s,
        "train_roc": train_roc_s,
        "test_roc": test_roc_s,
        "train_f1": train_f1_s,
        "test_f1": test_f1_s
    }
    with open("./output/nn-rhc.pkl", "wb") as fp:
        pickle.dump(obj, fp)


def nn_sa(x_train, y_train, x_test, y_test):
    print("********************Executing Simulated Annealing****************")
    max_attempts = np.linspace(10, 100, 10)
    learning_rates = [0.001, 0.01]
    train_acc_s = []
    test_acc_s = []
    train_f1_s = []
    test_f1_s = []
    train_roc_s = []
    test_roc_s = []
    for lr, max_attempt in product(learning_rates, max_attempts):
        print(f"Running for learning rate {lr}")
        nn_model = mlrose.NeuralNetwork(hidden_nodes=[8, 4],
                                        activation='relu',
                                        algorithm='simulated_annealing',
                                        max_iters=1000,
                                        bias=True,
                                        is_classifier=True,
                                        learning_rate=lr,
                                        early_stopping=True,
                                        clip_max=5,
                                        max_attempts=max_attempt,
                                        schedule=mlrose.ExpDecay(),
                                        random_state=1234)
        nn_model.fit(x_train, y_train)
        train_acc, train_roc_auc, train_f1 = calculate_metrics(nn_model, x_train, y_train)
        test_acc, test_roc_auc, test_f1 = calculate_metrics(nn_model, x_test, y_test)
        train_acc_s.append(train_acc)
        train_roc_s.append(train_roc_auc)
        train_f1_s.append(train_f1)
        test_acc_s.append(test_acc)
        test_roc_s.append(test_roc_auc)
        test_f1_s.append(test_f1)
    obj = {
        "max_attempts": max_attempts,
        "scheduler": ["Geometric", "Exponential"],
        "lr": learning_rates,
        "train_acc": train_acc_s,
        "test_acc": test_acc_s,
        "train_roc": train_roc_s,
        "test_roc": test_roc_s,
        "train_f1": train_f1_s,
        "test_f1": test_f1_s
    }
    with open("./output/nn-sa.pkl", "wb") as fp:
        pickle.dump(obj, fp)


def nn_ga(x_train, y_train, x_test, y_test):
    print("********************Executing Genetic Algorithm****************")
    pop_sizes = [50, 100, 150, 200]
    learning_rates = [0.001, 0.01]
    train_acc_s = []
    test_acc_s = []
    train_f1_s = []
    test_f1_s = []
    train_roc_s = []
    test_roc_s = []
    for lr, ps in product(learning_rates, pop_sizes):
        print(f"Running for learning rate {lr} and population size {ps}")
        nn_model = mlrose.NeuralNetwork(hidden_nodes=[32, 16],
                                        activation='relu',
                                        algorithm='genetic_alg',
                                        max_iters=1000,
                                        bias=True,
                                        is_classifier=True,
                                        learning_rate=lr,
                                        early_stopping=True,
                                        clip_max=5,
                                        max_attempts=100,
                                        pop_size=ps,
                                        random_state=1234)
        nn_model.fit(x_train, y_train)
        train_acc, train_roc_auc, train_f1 = calculate_metrics(nn_model, x_train, y_train)
        test_acc, test_roc_auc, test_f1 = calculate_metrics(nn_model, x_test, y_test)
        train_acc_s.append(train_acc)
        train_roc_s.append(train_roc_auc)
        train_f1_s.append(train_f1)
        test_acc_s.append(test_acc)
        test_roc_s.append(test_roc_auc)
        test_f1_s.append(test_f1)
    obj = {
        "pop_sizes": pop_sizes,
        "lr": learning_rates,
        "train_acc": train_acc_s,
        "test_acc": test_acc_s,
        "train_roc": train_roc_s,
        "test_roc": test_roc_s,
        "train_f1": train_f1_s,
        "test_f1": test_f1_s
    }
    with open("./output/nn-ga.pkl", "wb") as fp:
        pickle.dump(obj, fp)


def nn_gd(x_train, y_train, x_test, y_test):
    print("********************Executing Gradient Descent****************")
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    train_acc_s = []
    test_acc_s = []
    train_f1_s = []
    test_f1_s = []
    train_roc_s = []
    test_roc_s = []
    for lr in learning_rates:
        print(f"Running for learning rate {lr}")
        nn_model = mlrose.NeuralNetwork(hidden_nodes=[8, 4],
                                        activation='relu',
                                        algorithm='gradient_descent',
                                        max_iters=5000,
                                        bias=True,
                                        is_classifier=True,
                                        learning_rate=lr,
                                        early_stopping=True,
                                        clip_max=5,
                                        random_state=1234)
        nn_model.fit(x_train, y_train)
        train_acc, train_roc_auc, train_f1 = calculate_metrics(nn_model, x_train, y_train)
        test_acc, test_roc_auc, test_f1 = calculate_metrics(nn_model, x_test, y_test)
        train_acc_s.append(train_acc)
        train_roc_s.append(train_roc_auc)
        train_f1_s.append(train_f1)
        test_acc_s.append(test_acc)
        test_roc_s.append(test_roc_auc)
        test_f1_s.append(test_f1)
    obj = {
        "lr": learning_rates,
        "train_acc": train_acc_s,
        "test_acc": test_acc_s,
        "train_roc": train_roc_s,
        "test_roc": test_roc_s,
        "train_f1": train_f1_s,
        "test_f1": test_f1_s
    }
    with open("./output/nn-gd.pkl", "wb") as fp:
        pickle.dump(obj, fp)


if __name__ == '__main__':
    train = pd.read_csv("./dataset/train_filtered.csv.zip", compression="zip")
    valid = pd.read_csv("./dataset/valid_filtered.csv.zip", compression="zip")
    y_train = train.pop('target')
    y_valid = valid.pop('target')

    min_max_scaler = MinMaxScaler()
    train_scaled = min_max_scaler.fit_transform(train)
    valid_scaled = min_max_scaler.transform(valid)
    nn_ga(train_scaled, y_train, valid_scaled, y_valid)
