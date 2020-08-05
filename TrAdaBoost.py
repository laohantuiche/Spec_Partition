import numpy as np
from sklearn import svm
from sklearn import tree
import joblib
import pandas as pd


class TrAdaboost:
    def __init__(self, base_classifier=svm.SVC(), iter=10):
        self.base_classifier = base_classifier
        self.N = iter
        self.beta_all = np.zeros([1, self.N])
        self.classifiers = []

    def fit(self, x_source, x_target, y_source, y_target):
        x_train = np.concatenate((x_source, x_target), axis=0)
        y_train = np.concatenate((y_source, y_target), axis=0)
        x_train = np.asarray(x_train, order='C')
        y_train = np.asarray(y_train, order='C')
        y_source = np.asarray(y_source, order='C')
        y_target = np.asarray(y_target, order='C')

        row_source = x_source.shape[0]
        row_target = x_target.shape[0]

        # 初始化权重
        weight_source = np.ones([row_source, 1]) / row_source
        weight_target = np.ones([row_target, 1]) / row_target
        weights = np.concatenate((weight_source, weight_target), axis=0)

        beta = 1 / (1 + np.sqrt(2 * np.log(row_source / self.N)))

        result = np.ones([row_source + row_target, self.N])
        for i in range(self.N):
            weights = self._calculate_weight(weights)
            self.base_classifier.fit(x_train, y_train, sample_weight=weights[:, 0])
            self.classifiers.append(self.base_classifier)

            result[:, i] = self.base_classifier.predict(x_train)
            error_rate = self._calculate_error_rate(y_target,
                                                    result[row_source:, i],
                                                    weights[row_source:, :])

            print("Error Rate in target data: ", error_rate, 'round:', i+1, 'all_round:', self.N)

            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                self.N = i
                print("Early stopping...")
                break
            self.beta_all[0, i] = error_rate / (1 - error_rate)

            # 调整 target 样本权重 正确样本权重变大
            for t in range(row_target):
                weights[row_source + t] = weights[row_source + t] * np.power(self.beta_all[0, i], -np.abs(result[row_source + t, i] - y_target[t]))
            # 调整 source 样本 错分样本变大
            for s in range(row_source):
                weights[s] = weights[s] * np.power(beta, np.abs(result[s, i] - y_source[s]))

    def predict(self, x_test):
        result = np.ones([x_test.shape[0], self.N + 1])
        predict = []

        i = 0
        for classifier in self.classifiers:
            y_pred = classifier.predict(x_test)
            result[:, i] = y_pred
            i += 1

        for i in range(x_test.shape[0]):
            left = np.sum(result[i, int(np.ceil(self.N / 2)): self.N] *
                          np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)):self.N]))

            right = 0.5 * np.sum(np.log(1 / self.beta_all[0, int(np.ceil(self.N / 2)): self.N]))

            if left >= right:
                predict.append(1)
            else:
                predict.append(0)
        return predict

    def _calculate_weight(self, weights):
        sum_weight = np.sum(weights)
        return np.asarray(weights / sum_weight, order='C')

    def _calculate_error_rate(self, y_target, y_predict, weight_target):
        sum_weight = np.sum(weight_target)
        temp_arr = []
        for i in range(len(y_target)):
            temp_arr.append(y_target[i][0])
        y_target = np.array(temp_arr)
        return np.sum(weight_target[:, 0] / sum_weight * np.abs(y_target - y_predict))

    def save_model(self):
        head = 'Stream_or_not_{index}.pkl'
        for i in range(len(self.classifiers)):
            joblib.dump(self.classifiers[i], head.format(idex=i+1))


original_data = pd.read_csv('DT_dataset\\213\\stream_213.csv', sep=',')

features_arr = ["Frequency", "IPC", "Misses", "LLC", "MBL", "Memory_Footprint",
                "Virt_Memory", "Res_Memory", "Allocated_Cache", "stream"]
test_data = pd.read_csv('DT_dataset\\43\\stream_43.csv', sep=',')
test_data = (test_data-original_data.min())/(original_data.max()-original_data.min())
input_test = test_data.loc[:, features_arr[:9]].values
output_test = test_data.loc[:, features_arr[9:]].values

train_data = pd.read_csv('DT_dataset\\43\\stream_43_train.csv', sep=',')
train_data = (train_data-original_data.min())/(original_data.max()-original_data.min())
input_train = train_data.loc[:, features_arr[:9]].values
output_train = train_data.loc[:, features_arr[9:]].values

original_train_data = pd.read_csv('DT_dataset\\213\\stream_213_small_train.csv', sep=',')
original_train_data = (original_train_data-original_data.min())/(original_data.max()-original_data.min())
# data.to_csv("normalized_ST_or_CS.csv")
original_train_input = train_data.loc[:, features_arr[:9]].values
original_train_output = train_data.loc[:, features_arr[9:]].values

decision_tree = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random", max_depth=15)
transfer_learning = TrAdaboost(decision_tree, 10)
transfer_learning.fit(original_train_input, input_train, original_train_output, output_train)
predicted_result = transfer_learning.predict(input_test)
acc = 0
for i in range(len(predicted_result)):
    if predicted_result[i] == output_test[i]:
        acc += 1
print(acc/len(predicted_result))
