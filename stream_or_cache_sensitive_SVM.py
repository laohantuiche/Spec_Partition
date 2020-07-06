import pandas as pd
from sklearn import svm
import joblib

data = pd.read_csv("ST_or_CS_44.csv", sep=',')
"""
profile = data.profile_report(title="relation")
profile.to_file(output_file="classifier analysis.html")
"""

features_arr = ["CPU_Utilization", "Frequency", "IPC", "Misses", "LLC", "MBL",
                "Memory_Footprint", "Virt_Memory", "Res_Memory", "Allocated_Cache", "ST"]

data = (data-data.min())/(data.max()-data.min())
# data.to_csv("normalized_ST_or_CS.csv")
input_features = data.loc[:, features_arr[:10]].values
output_target = data.loc[:, features_arr[10:]].values

classifier = svm.SVC()
classifier.fit(input_features, output_target)
print(classifier.score(input_features, output_target))

joblib.dump(classifier, "stream_or_cache_sensitive.pkl")
