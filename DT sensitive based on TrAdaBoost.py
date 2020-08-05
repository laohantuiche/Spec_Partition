import numpy as np
from sklearn import tree
from sklearn import svm
import random
import pandas as pd
import joblib

# H 测试样本分类结果
# TrainS 原训练样本 np数组
# TrainA 辅助训练样本
# LabelS 原训练样本标签
# LabelA 辅助训练样本标签
# Test  测试样本
# N 迭代次数
# """


def tradaboost(trans_S, trans_A, label_S, label_A, test, N):
    trans_data = np.concatenate((trans_A, trans_S), axis=0)
    trans_label = np.concatenate((label_A, label_S), axis=0)

    row_A = trans_A.shape[0]
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    test_data = np.concatenate((trans_data, test), axis=0)

    # 初始化权重
    weights_A = np.ones([row_A, 1]) / row_A
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0)

    bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

    # 存储每次迭代的标签和bata值？
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])

    predict = np.zeros([row_T])

    print('params initial finished.')
    trans_data = np.asarray(trans_data, order='C')
    trans_label = np.asarray(trans_label, order='C')
    test_data = np.asarray(test_data, order='C')

    for i in range(N):
        P = calculate_P(weights, trans_label)

        result_label[:, i] = train_classify(trans_data, trans_label, test_data, P, i+1)
        print('result,{r},{ra},{rs},{idex},{s}'.format(r=result_label[:, i], ra=row_A, rs=row_S, idex=i, s=result_label.shape))
        for j in range(len(result_label[row_A:row_A + row_S, i])):
            print(result_label[j, i])
        error_rate = calculate_error_rate(label_S, result_label[row_A:row_A + row_S, i],
                                          weights[row_A:row_A + row_S, :])
        print('Error rate:{er}'.format(er=error_rate))
        if error_rate > 0.5:
            error_rate = 0.5
        if error_rate == 0:
            N = i
            break  # 防止过拟合
            # error_rate = 0.001

        bata_T[0, i] = error_rate / (1 - error_rate)

        # 调整源域样本权重
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i],
                                                               (-np.abs(result_label[row_A + j, i] - label_S[j])))

        # 调整辅域样本权重
        for j in range(row_A):
            weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - label_A[j]))
    print(bata_T)
    for i in range(row_T):
        # 跳过训练数据的标签
        left = np.sum(
            result_label[row_A + row_S + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))

        if left >= right:
            predict[i] = 1
        else:
            predict[i] = 0
            # print left, right, predict[i]

    return predict


def calculate_P(weights, label):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


def train_classify(trans_data, trans_label, test_data, P, index):
    clf = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random", max_depth=15)
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    # joblib.dump(clf, '43_model_file\\sensitive_{item}.pkl'.format(item=index))
    return clf.predict(test_data)


def calculate_error_rate(label_R, label_H, weight):
    total = np.sum(weight)

    print(weight[:, 0] / total)

    temp_arr = []
    for i in range(len(label_R)):
        temp_arr.append(label_R[i][0])
    label_R = np.array(temp_arr)
    print(np.abs(label_R - label_H))
    return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))
# """


"""
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

    def predict_prob(self, x_test):
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
            predict.append([left, right])
        return predict

    def _calculate_weight(self, weights):
        sum_weight = np.sum(weights)
        return np.asarray(weights / sum_weight, order='C')

    def _calculate_error_rate(self, y_target, y_predict, weight_target):
        sum_weight = np.sum(weight_target)
        return np.sum(weight_target[:, 0] / sum_weight * np.abs(y_target - y_predict))
"""


# 数据处理
"""
data_file = open('DT_dataset\\43\\sensitive_43.csv', 'r')
test_data_file = open('DT_dataset\\43\\sensitive_43_test.csv', 'w')
train_data_file = open('DT_dataset\\43\\sensitive_43_train.csv', 'w')
data = data_file.readlines()
train_data_file.write(data[0])
test_data_file.write(data[0])

for i in range(1, len(data)):
    random_number = random.randint(1, 50)
    if random_number == 25:
        random_number = random.randint(0, 9)
        if random_number < 7:
            train_data_file.write(data[i])
        else:
            test_data_file.write(data[i])

data_file.close()
train_data_file.close()
test_data_file.close()

small_data_file = open('DT_dataset\\213\\stream_213_small_train.csv', 'w')
data_file = open('DT_dataset\\213\\stream_213_train.csv', 'r')
data = data_file.readlines()
small_data_file.write(data[0])
for i in range(1, len(data)):
    random_num = random.randint(1, 20)
    if random_num == 10:
        small_data_file.write(data[i])
small_data_file.close()
"""

# 测试

original_data = pd.read_csv('DT_dataset\\213\\sensitive_213_test.csv', sep=',')

features_arr = ["Frequency", "IPC", "Misses", "LLC", "MBL", "Memory_Footprint",
                "Virt_Memory", "Res_Memory", "Allocated_Cache", "sensitive"]
test_data = pd.read_csv('DT_dataset\\43\\sensitive_43.csv', sep=',')
test_data = (test_data-original_data.min())/(original_data.max()-original_data.min())
input_test = test_data.loc[:, features_arr[:9]].values
output_test = test_data.loc[:, features_arr[9:]].values

train_data = pd.read_csv('DT_dataset\\43\\sensitive_43_train.csv', sep=',')
train_data = (train_data-original_data.min())/(original_data.max()-original_data.min())
input_train = train_data.loc[:, features_arr[:9]].values
output_train = train_data.loc[:, features_arr[9:]].values

original_train_data = pd.read_csv('DT_dataset\\213\\sensitive_213_train.csv', sep=',')
original_train_data = (original_train_data-original_data.min())/(original_data.max()-original_data.min())
# data.to_csv("normalized_ST_or_CS.csv")
original_train_input = train_data.loc[:, features_arr[:9]].values
original_train_output = train_data.loc[:, features_arr[9:]].values

svm_classifier = joblib.load('sensitive_or_not_213.pkl')
print(svm_classifier.score(input_test, output_test))
dt_classifier = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random", max_depth=15)
dt_classifier.fit(input_train, output_train)
print(svm_classifier.score(input_test, output_test))

"""
classfier = TrAdaboost(svm.SVC(), 10)
classfier.fit(original_train_input, input_train, original_train_output, output_train)
predicted_result = classfier.predict(input_test)
"""
predicted_result = tradaboost(original_train_input, input_train, original_train_output, output_train, input_train, 10)
dt_classifier = joblib.load('sensitive_or_not_43.pkl')
print(dt_classifier.score(input_train, output_train))
acc = 0
for i in range(len(predicted_result)):
    if predicted_result[i] == output_train[i]:
        acc += 1
print(acc/len(predicted_result))

clfs = []
for i in range(10):
    clf = joblib.load('43_model_file\\sensitive_{item}.pkl'.format(item=i+1))
    clfs.append(clf)


def predict(x_test):
    result = np.ones([x_test.shape[0], 11])
    predict = []

    i = 0
    for classifier in clfs:
        y_pred = classifier.predict(x_test)
        result[:, i] = y_pred
        i += 1

    beta_all = np.array([9.95272456e-04, 9.96264010e-04, 1.24548512e-04, 1.86857677e-04, 6.07656472e-04, 7.79052828e-06, 3.99981619e-03, 1.95546048e-05, 9.77731194e-07, 1.29752359e-03])
    beta_all = np.array([beta_all])

    for i in range(x_test.shape[0]):
        left = np.sum(result[i, int(np.ceil(10 / 2)): 10] *
                      np.log(1 / beta_all[0, int(np.ceil(10 / 2)):10]))

        right = 0.5 * np.sum(np.log(1 / beta_all[0, int(np.ceil(10 / 2)): 10]))

        if left >= right:
            predict.append(1)
        else:
            predict.append(0)
    return predict

predicted_result = predict(input_test)
acc = 0
for i in range(len(predicted_result)):
    if predicted_result[i] == output_test[i]:
        acc += 1
print(acc/len(predicted_result))
