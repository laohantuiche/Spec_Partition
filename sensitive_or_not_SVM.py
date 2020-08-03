import pandas as pd
from sklearn import svm
import joblib

data = pd.read_csv("SVM_dataset\\213\\sensitive_213.csv", sep=',')

features_arr = ["Frequency", "IPC", "Misses", "LLC", "MBL", "Memory_Footprint",
                "Virt_Memory", "Res_Memory", "Allocated_Cache", "sensitive"]

data = pd.read_csv('SVM_dataset\\213\\sensitive_213.csv', sep=',')

train_data = pd.read_csv('SVM_dataset\\213\\sensitive_213_train.csv', sep=',')
train_data = (train_data-data.min())/(data.max()-data.min())
print('train data normalization finished')
# data.to_csv("normalized_ST_or_CS.csv")
train_input_features = train_data.loc[:, features_arr[:9]].values
train_output_target = train_data.loc[:, features_arr[9:]].values

test_data = pd.read_csv('SVM_dataset\\213\\sensitive_213_test.csv', sep=',')
test_data = (test_data-data.min())/(data.max()-data.min())
print('test data normalization finished')
# data.to_csv("normalized_ST_or_CS.csv")
test_input_features = test_data.loc[:, features_arr[:9]].values
test_output_target = test_data.loc[:, features_arr[9:]].values

classifier = svm.SVC()
classifier.fit(train_input_features, train_output_target)
print(classifier.score(test_input_features, test_output_target))

joblib.dump(classifier, "sensitive_or_not_213.pkl")



