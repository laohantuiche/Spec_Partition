import pandas as pd
import numpy as np
import random
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector

column_index = [
    "CPU_Utilization_1_cs", "Frequency_1_cs", "IPC_1_cs", "Misses_1_cs", "LLC_1_cs", "MBL_1_cs", "Memory_Footprint_1_cs", "Virt_Memory_1_cs", "Res_Memory_1_cs", "Weighted_Speedup_1_cs",
    "CPU_Utilization_1_s", "Frequency_1_s", "IPC_1_s", "Misses_1_s", "LLC_1_s", "MBL_1_s", "Memory_Footprint_1_s", "Virt_Memory_1_s", "Res_Memory_1_s", "Weighted_Speedup_1_s",
    "CPU_Utilization_2_cs", "Frequency_2_cs", "IPC_2_cs", "Misses_2_cs", "LLC_2_cs", "MBL_2_cs", "Memory_Footprint_2_cs", "Virt_Memory_2_cs", "Res_Memory_2_cs", "Weighted_Speedup_2_cs",
    "CPU_Utilization_2_s", "Frequency_2_s", "IPC_2_s", "Misses_2_s", "LLC_2_s", "MBL_2_s", "Memory_Footprint_2_s", "Virt_Memory_2_s", "Res_Memory_2_s", "Weighted_Speedup_2_s",
    "CPU_Utilization_3_cs", "Frequency_3_cs", "IPC_3_cs", "Misses_3_cs", "LLC_3_cs", "MBL_3_cs", "Memory_Footprint_3_cs", "Virt_Memory_3_cs", "Res_Memory_3_cs", "Weighted_Speedup_3_cs",
    "CPU_Utilization_3_s", "Frequency_3_s", "IPC_3_s", "Misses_3_s", "LLC_3_s", "MBL_3_s", "Memory_Footprint_3_s", "Virt_Memory_3_s", "Res_Memory_3_s", "Weighted_Speedup_3_s",
    "Weighted_Speedup_4", "Weighted_Speedup_5", "Weighted_Speedup_6", "Weighted_Speedup_7", "Weighted_Speedup_8", "Weighted_Speedup_9", "Weighted_Speedup_10"
]

data = pd.read_csv("seq2seq.csv", sep=',')
data = (data-data.min())/(data.max()-data.min())
# data.to_csv("normalized_seq2seq.csv")

state_input_1 = data.loc[:, column_index[:20]].values
state_input_2 = data.loc[:, column_index[20:40]].values
state_input_3 = data.loc[:, column_index[40:60]].values
target_4 = data.loc[:, column_index[60]].values
target_5 = data.loc[:, column_index[61]].values
target_6 = data.loc[:, column_index[62]].values
target_7 = data.loc[:, column_index[63]].values
target_8 = data.loc[:, column_index[64]].values
target_9 = data.loc[:, column_index[65]].values
target_10 = data.loc[:, column_index[66]].values

input_train = []
input_test = []
output_train = []
output_test = []

for i in range(len(state_input_1)):
    random_number = random.randint(0, 9)
    input_temp = np.array([state_input_1[i], state_input_2[i], state_input_3[i]])
    output_temp = np.array(
        [
            np.array([target_4[i]]), np.array([target_5[i]]), np.array([target_6[i]]), np.array([target_7[i]]),
            np.array([target_8[i]]), np.array([target_9[i]]), np.array([target_10[i]])
        ]
    )
    if random_number < 7:
        input_train.append(input_temp)
        output_train.append(output_temp)
    else:
        input_test.append(input_temp)
        output_test.append(output_temp)

input_train = np.array(input_train)
input_test = np.array(input_test)
output_train = np.array(output_train)
output_test = np.array(output_test)

encode_steps = 4
encode_features = 20
target = 1
decode_steps = 13

model = Sequential()

# Encoder(第一个 LSTM)
model.add(LSTM(input_shape=(encode_steps, encode_features), units=1, return_sequences=False, dropout=0.2))

model.add(Dense(target, activation="relu"))

# 使用 "RepeatVector" 将 Encoder 的输出(最后一个 time step)复制 N 份作为 Decoder 的 N 次输入
model.add(RepeatVector(target*decode_steps))

# Decoder(第二个 LSTM)
model.add(LSTM(input_shape=(decode_steps, target), units=1, return_sequences=True, dropout=0.2))

# TimeDistributed 是为了保证 Dense 和 Decoder 之间的一致
model.add(TimeDistributed(Dense(units=1, activation="linear")))

model.compile(loss="mse", optimizer='adam')

model.summary()
"""
model.fit(
    input_train, output_train,
    batch_size=1000, epochs=80,
    validation_data=(input_test, output_test),
    verbose=1, shuffle=True
)

model.save("seq2seq.h5")
"""