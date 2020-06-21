import pandas as pd
from sklearn import svm
import joblib

data = pd.read_csv("need_protection_44.csv", sep=',')

features_arr = ["CPU_Utilization", "Frequency", "IPC", "Misses", "LLC", "MBL",
                "Memory_Footprint", "Virt_Memory", "Res_Memory", "Allocated_Cache", "Protected"]

data = (data-data.min())/(data.max()-data.min())
data.to_csv("normalized_need_protected_dataset.csv")
input_features = data.loc[:, features_arr[:10]].values
output_target = data.loc[:, features_arr[10:]].values

classifier = svm.SVC()
classifier.fit(input_features, output_target)
print(classifier.score(input_features, output_target))

joblib.dump(classifier, "protected_or_not.pkl")
