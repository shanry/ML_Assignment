import numpy as np
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def load_data_train():
    X = []
    Y = []
    with open('./adult_dataset/adult_train_feature.txt') as f:
        lines = f.readlines()
        for line in lines:
            feature = line.split()
            X.append(list(map(float, feature)))
    with open('./adult_dataset/adult_train_label.txt') as f:
        lines = f.readlines()
        Y.extend(list(map(int, lines)))
    return np.array(X), np.array(Y)


def load_data_test():
    X = []
    Y = []
    with open('./adult_dataset/adult_test_feature.txt') as f:
        lines = f.readlines()
        for line in lines:
            feature = line.split()
            X.append(list(map(float, feature)))
    with open('./adult_dataset/adult_test_label.txt') as f:
        lines = f.readlines()
        Y.extend(list(map(int, lines)))
    return np.array(X), np.array(Y)


class RandomForest:
    def __init__(self, n_estimators=50, max_depth=None):
        assert n_estimators >= 1
        self.T = n_estimators
        self.estimators = []
        self.max_depth = max_depth

    def fit(self, x_train, y_train):
        H = 0
        for t in range(self.T):
            dtc = DecisionTreeClassifier(max_depth=self.max_depth, max_features="log2")
            x_bs, y_bs = resample(x_train, y_train)
            dtc.fit(x_bs, y_bs)
            h = dtc.predict(x_train)
            H += h
            H[H == 0] = 1
            self.estimators.append(dtc)
        return np.sign(H)

    def predicate(self, x_test):
        H = 0
        for dtc in self.estimators:
            H += dtc.predict(x_test)
        return np.sign(H)

    def score(self, x_test, y_test):
        pred = self.predicate(x_test)
        return (pred == y_test).sum() / len(y_test)

    def predict_proba(self, x_test):
        proba = 0
        for dtc in self.estimators:
            proba += dtc.predict_proba(x_test)
        return proba


def valid_randomforest(max_T):
    train_x, train_y = load_data_train()
    train_y[train_y == 0] = -1
    test_x, test_y = load_data_test()
    test_y[test_y == 0] = -1
    kf = KFold(n_splits=5, shuffle=True)
    train_set = [(train_index, test_index) for train_index, test_index in kf.split(train_x)]

    AUC = []
    for T in range(max_T):
        accs = []
        aucs = []
        for train_index, test_index in train_set:
            X_train, X_test = train_x[train_index], train_x[test_index]
            Y_train, Y_test = train_y[train_index], train_y[test_index]
            RF = RandomForest(T + 1, 14)
            RF.fit(X_train, Y_train)
            acc = RF.score(X_test, Y_test)
            score = RF.predict_proba(X_test)
            auc = roc_auc_score(Y_test, score[:, 1])
            accs.append(acc)
            aucs.append(auc)
            # print("test acc", acc)
            # print("test auc", auc)
        # print("average acc:", np.average(accs))
        print("average auc:", np.average(aucs))
        AUC.append(np.average(aucs))
    print(np.argmax(AUC), np.max(AUC))
    x = list(range(1, len(AUC) + 1))
    # plt.plot(x, AUC)
    plt.show()
    return np.argmax(AUC) + 1, x, AUC


def test_randomforest(best_T):
    X_train, Y_train = load_data_train()
    Y_train[Y_train == 0] = -1
    X_test, Y_test = load_data_test()
    Y_test[Y_test == 0] = -1

    RF = RandomForest(best_T, 14)
    RF.fit(X_train, Y_train)
    acc = RF.score(X_test, Y_test)
    score = RF.predict_proba(X_test)
    auc = roc_auc_score(Y_test, score[:, 1])

    return auc, acc


if __name__ == '__main__':
    best_T, x, AUC = valid_randomforest(60)
    auc, acc = test_randomforest(best_T)
    print("RandomForest:", auc, acc)
    plt.plot(x, AUC, 'b')
    plt.show()
