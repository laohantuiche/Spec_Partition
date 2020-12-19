import random
import pandas as pd
from sklearn import svm
from sklearn import metrics
import joblib

"""
overall_file = open('SVM_Dataset\\sensitive.csv', 'r')
all_data = overall_file.readlines()

train_file = open("SVM_Dataset\\sensitive_train.csv", 'w')
test_file = open("SVM_Dataset\\sensitive_test.csv", 'w')

train_file.write(all_data[0])
test_file.write(all_data[0])

for i in range(1, len(all_data)):
    random_number = random.randint(0, 9)
    if random_number < 7:
        train_file.write(all_data[i])
    else:
        test_file.write(all_data[i])
"""

features_arr = ["Frequency", "IPC", "Misses", "LLC", "MBL", "Memory_Footprint",
                "Virt_Memory", "Res_Memory", "Allocated_Cache", "sensitive"]

train_data = pd.read_csv('SVM_Dataset\\sensitive_train.csv', sep=',')
train_input_features = train_data.loc[:, features_arr[:9]].values
train_output_target = train_data.loc[:, features_arr[9:]].values

test_data = pd.read_csv('SVM_Dataset\\sensitive_test.csv', sep=',')
test_input_features = test_data.loc[:, features_arr[:9]].values
test_output_target = test_data.loc[:, features_arr[9:]].values
print("initialize done")

model = svm.SVC()
model.fit(train_input_features, train_output_target)
print("train successfully")
prediction = model.predict(test_input_features)
print(metrics.accuracy_score(test_output_target, prediction))
print(metrics.precision_score(test_output_target, prediction, average='macro'))
print(metrics.recall_score(test_output_target, prediction, average='macro'))

joblib.dump(model, 'Sensitive_or_not_SVM.pkl')
print("save successfully")

