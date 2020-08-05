import pandas as pd
from sklearn import svm
from sklearn import tree
import joblib

data = pd.read_csv("SVM_dataset\\213\\stream_213.csv", sep=',')
"""
# profile = data.profile_report(title="relation")
# profile.to_file(output_file="classifier analysis.html")
"""

features_arr = ["Frequency", "IPC", "Misses", "LLC", "MBL", "Memory_Footprint",
                "Virt_Memory", "Res_Memory", "Allocated_Cache", "stream"]

train_data = pd.read_csv('SVM_dataset\\213\\stream_213_train.csv', sep=',')
train_data = (train_data-data.min())/(data.max()-data.min())
print('normalization finished')
# data.to_csv("normalized_ST_or_CS.csv")
train_input_features = train_data.loc[:, features_arr[:9]].values
train_output_target = train_data.loc[:, features_arr[9:]].values

test_data = pd.read_csv('SVM_dataset\\213\\stream_213_test.csv', sep=',')
test_data = (test_data-data.min())/(data.max()-data.min())
print('normalization finished')
# data.to_csv("normalized_ST_or_CS.csv")
test_input_features = test_data.loc[:, features_arr[:9]].values
test_output_target = test_data.loc[:, features_arr[9:]].values

classifier = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random")
classifier.fit(train_input_features, train_output_target)
print(classifier.score(test_input_features, test_output_target))

joblib.dump(classifier, "stream_or_not_213.pkl")

"""
overall_file = open('SVM_dataset\\213\\sensitive_213.csv', 'r')
data = overall_file.readlines()
train_file = open('SVM_dataset\\213\\sensitive_213_train.csv', 'w')
test_file = open('SVM_dataset\\213\\sensitive_213_test.csv', 'w')
train_file.write(data[0])
test_file.write(data[0])
import random
for i in range(1, len(data)):
    random_num = random.randint(0, 9)
    if random_num < 7:
        train_file.write(data[i])
    else:
        test_file.write(data[i])
"""
