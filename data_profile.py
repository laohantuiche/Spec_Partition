import pandas as pd
import math
# import pandas_profiling

data = pd.read_csv('DT_dataset\\213\\stream_213.csv', sep=',')
"""
profile = data.profile_report(title='stream or cache sensitive')
profile.to_file(output_file='profile.html')
"""


# 计算特征和类的平均值
def calc_mean(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean, y_mean


#计算Pearson系数
def calc_pearson(x, y):
    x_mean, y_mean = calc_mean(x, y) # 计算x,y向量平均值
    n = len(x)
    sum_top = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sum_top += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean, 2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean, 2)
    sum_bottom = math.sqrt(x_pow*y_pow)
    p = sum_top/sum_bottom
    return p


features_arr = ["Frequency", "IPC", "Misses", "LLC", "MBL", "Memory_Footprint",
                "Virt_Memory", "Res_Memory", "Allocated_Cache", "stream"]
"""
for features_i in features_arr:
    temp_str = ''
    for features_j in features_arr:
        temp_str += str(calc_pearson(data[features_i], data[features_j]))+','
    temp_str = temp_str[:-1]
    print(temp_str)
"""

pearson_correlation = [
    [1.0, 0.07305463309403963, -0.051832717608371985, 0.2770743599299148, -0.03254166662323912, -0.001337135438886153, 0.001088381064087852, 0.002324513486492664, 0.26750653077877995, -0.00963029869950182],
    [0.07305463309403963, 1.0, -0.04443755438766835, -0.032920171855532726, 0.041266648322769765, -0.20892991607469333, -0.1846101346475898, -0.18939881824468144, -0.05688234635960439, -0.025628250409627962],
    [-0.051832717608371985, -0.04443755438766835, 1.0, 0.3486948074656889, 0.9360393126153073, 0.2503295433181082, 0.18304529694854396, 0.20569510459903512, 0.08714954676933695, 0.19870218928006897],
    [0.2770743599299148, -0.032920171855532726, 0.3486948074656889, 1.0,0.3769494989410002, 0.3106295849847942, 0.27842439793292384, 0.29257838844312056, 0.6941446142113252, 0.3492304233791552],
    [-0.03254166662323912, 0.041266648322769765, 0.9360393126153073, 0.3769494989410002, 1.0,0.2716310884624986, 0.21248032303888578, 0.23470397773820142, 0.09543620902656268, 0.2699675277576073],
    [-0.001337135438886153, -0.20892991607469333, 0.2503295433181082, 0.3106295849847942, 0.2716310884624986, 1.0,0.977158483486253, 0.9929534749348781, 0.2402317998367734, 0.5665973625343593],
    [0.001088381064087852, -0.1846101346475898, 0.18304529694854396, 0.27842439793292384, 0.21248032303888578, 0.977158483486253, 1.0,0.9858885457310885, 0.2270641443802927, 0.5878736373911931],
    [0.002324513486492664, -0.18939881824468144, 0.20569510459903512, 0.29257838844312056, 0.23470397773820142, 0.9929534749348781, 0.9858885457310885, 1.0, 0.2333202433354069, 0.5837383277380639],
    [0.26750653077877995, -0.05688234635960439, 0.08714954676933695, 0.6941446142113252, 0.09543620902656268, 0.2402317998367734, 0.2270641443802927, 0.2333202433354069, 1.0, 0.3040660885831414],
    [-0.00963029869950182, -0.025628250409627962, 0.19870218928006897, 0.3492304233791552, 0.2699675277576073, 0.5665973625343593, 0.5878736373911931, 0.5837383277380639, 0.3040660885831414, 1.0]
]

for features in features_arr:
    print(features)

for index in pearson_correlation[-1]:
    print(index)

print(calc_pearson(data['CPU_Utilization'], data[features_arr[-1]]))