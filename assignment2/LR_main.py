import numpy as np
from random import sample


def load_data_x(filename):
    x = []
    with open(filename) as f:
        lines = f.readlines();
        x = np.ones((len(lines), 10))
        for i in range(len(lines)):
            line = lines[i]
            features = line.split();
            for j in range(len(features)):
                x[i][j] = float(features[j])
    return x


def load_data_y(filename):
    y = []
    with open(filename) as f:
        lines = f.readlines();
        y = np.zeros(len(lines), dtype=np.int)
        for i in range(len(lines)):
            y[i] = int(lines[i])
    return y


def kmean_center(x_train, y_train, K):
    label_augment = 1
    x_label = x_train[y_train == label_augment]
    diff = len(y_train) - 2 * len(x_label)
    if diff < 0:
        x_label = x_train[y_train != label_augment]
        diff = len(y_train) - 2 * len(x_label)
        label_augment = 1 - label_augment
    x_augment = np.zeros((diff, x_train.shape[1]))
    y_augment = np.zeros(diff) + label_augment
    center = x_label.mean(axis=0)
    for i in range(diff):
        x_seed = x_label[sample(range(0, len(x_label)), K)]
        coef = np.random.rand(x_seed.shape[1])
        x_augment[i] = coef * x_seed.mean(axis=0) + (1 - coef) * center
    return np.vstack((x_train, x_augment)), np.hstack((y_train, y_augment))


class LogisticRegression:
    def __init__(self):
        self.alpha = 0.001  # learning reate
        self.coefficient = None
        self.loss = 0
        self.X_train = None
        self.Y_train = None
        self.norm = None
        self.label = None
        self.mean = None
        self.std = None

    def init(self, learning_reate=0.01):
        self.alpha = learning_reate          # learning reate
        self.loss = 0
        self.X_train = None
        self.Y_train = None
        self.norm = True
        self.label = -1

    def load_data_train(self, x_train, y_train, label, norm=True):
        self.X_train = x_train.copy()
        self.norm = norm
        self.label = label   # 1 2 3 4 5
        self.Y_train = y_train.copy()
        self.mean = self.X_train.mean(axis=0)
        self.std = self.X_train.std(axis=0)

    def oversampling(self, K):
        self.X_train, self.Y_train = kmean_center(self.X_train, self.Y_train, K)

    def label2binary(self):
        for i in range(len(self.Y_train)):
            if self.Y_train[i] == self.label:
                self.Y_train[i] = 1
            else:
                self.Y_train[i] = 0

    def __normalization_data(self):
        # self.mean = self.X_train.mean(axis=0)
        # self.std = self.X_train.std(axis=0)
        self.X_train = (self.X_train - self.mean)/self.std

    def update_loss(self):
        dot = np.dot(self.X_train, self.coefficient)
        self.loss = -dot * self.Y_train + np.log(1 + np.exp(dot))
        return self.loss.sum()

    def update_accuracy(self):
        dot = np.dot(self.X_train, self.coefficient)
        p0 = 1/(1 + np.exp(dot))
        p = np.array([p0, 1-p0])
        predicate = p.argmax(axis=0)
        return (predicate == self.Y_train).sum()/len(self.Y_train)

    def preprocess_data(self):
        if self.norm:
            self.__normalization_data()
        else:
            self.X_train = self.X_train/self.X_train.sum()
        intercept = np.ones(len(self.X_train))
        self.X_train = np.hstack((self.X_train, intercept.reshape(len(intercept), 1)))
        self.coefficient = np.random.random(self.X_train.shape[1])

    def fit(self, epoch=1000, batch_size=32):
        # print("shape:", self.X_train.shape)
        index_shuffle = np.arange(len(self.Y_train))
        for ep in range(epoch):
            np.random.shuffle(index_shuffle)
            for episode in range(int(np.ceil(len(self.Y_train)//batch_size))):
                index = index_shuffle[episode*batch_size: (episode+1)*batch_size]
                p1 = 1 - 1/(1 + np.exp(np.dot(self.X_train[index], self.coefficient)))
                factor = self.Y_train[index] - p1
                gradient = -self.X_train[index] * factor.reshape((len(factor), 1))
                self.coefficient -= self.alpha * gradient.sum(axis=0)/len(factor)
            epoch -= 1
        return self.update_loss()

    def predicate(self, X_test, Y_test):
        if self.norm:
            X_test = (X_test - self.mean) / self.std
        intercept = np.ones(len(X_test))
        X_test = np.hstack((X_test, intercept.reshape(len(intercept), 1)))
        Y_test = Y_test == self.label
        dot = np.dot(X_test, self.coefficient)
        p0 = 1 / (1 + np.exp(dot))
        p = np.array([p0, 1 - p0])
        predicate = p.argmax(axis=0)
        accu = (predicate == Y_test).sum() / len(Y_test)
        print("test", len(Y_test), "examples", "label", self.label, "acuu", accu)
        return accu

    def predicate_prob(self, X_test):
        if self.norm:
            X_test = (X_test - self.mean) / self.std
        intercept = np.ones(len(X_test))
        X_test = np.hstack((X_test, intercept.reshape(len(intercept), 1)))
        dot = np.dot(X_test, self.coefficient)
        p0 = 1 / (1 + np.exp(dot))
        return 1 - p0

    def classify_accu(predict, truth):
        accu_test = (predict == truth).sum()/len(truth)
        print("test acuu", accu_test)
        precision = np.zeros(5)
        recall = np.zeros(5)
        for i in range(5):
            index_p = predict==i
            precision[i] = (predict == truth)[index_p].sum()/index_p.sum()
            index_r = truth==i
            recall[i] = (predict == truth)[index_r].sum()/index_r.sum()
        print("precision:", precision)
        print("recall:", recall)
        return accu_test


def main(K=2, F=0):
    assert K >= 0 and type(K) == int
    assert F >= 0 and F < 2
    x = load_data_x('./assign2_dataset/page_blocks_train_feature.txt')
    y = load_data_y('./assign2_dataset/page_blocks_train_label.txt')
    x_test = load_data_x('./assign2_dataset/page_blocks_test_feature.txt')
    y_test = load_data_y('./assign2_dataset/page_blocks_test_label.txt')

    mylr = LogisticRegression()
    prob = np.zeros((5, len(y_test)))
    for label in range(5):
        mylr.init(learning_reate=0.02)
        mylr.load_data_train(x, y, label+1, norm=True)
        mylr.label2binary()
        if K > 0:
            mylr.oversampling(K)
        mylr.preprocess_data()
        mylr.fit(epoch=1000, batch_size=32)
        mylr.predicate(x_test, y_test)
        prob[label] = mylr.predicate_prob(x_test)
    if F > 0:
        prob[0] *= F
    labels = prob.argmax(axis=0)
    LogisticRegression.classify_accu(labels, y_test-1)


if __name__ == '__main__':
    # main(0, 0)      # raw
    main(5, 0)          # oversample
    # main(0, 0.4)    # move-threshold


