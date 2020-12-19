import pandas as pd
import numpy as np
import random
import re
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector
from keras import optimizers

"""
overall_file = open('Seq2seq_dataset\\43\\seq2seq_43.csv', 'r')
data = overall_file.readlines()
new_data = []
for i, item in enumerate(data):
    element = re.split(',', item[:-1])
    if 'inf' not in element:
        print(item[:-1])
        new_data.append(item)
overall_file.close()
overall_file = open('Seq2seq_dataset\\43\\seq2seq_43.csv', 'w')
for item in new_data:
    overall_file.write(item)

train_file = open('Seq2seq_dataset\\43\\seq2seq_43_train.csv', 'w')
test_file = open('Seq2seq_dataset\\43\\seq2seq_43_test.csv', 'w')
train_file.write(data[0])
test_file.write(data[0])
for i in range(1, len(new_data)):
    random_num = random.randint(0, 9)
    if random_num < 7:
        train_file.write(new_data[i])
    else:
        test_file.write(new_data[i])
"""

column_index = [
    'CPU_Utilization_1_sen', 'Frequency_1_sen', 'IPC_1_sen', 'Misses_1_sen', 'LLC_1_sen', 'MBL_1_sen', 'Memory_Footprint_1_sen', 'Virt_Memory_1_sen', 'Res_Memory_1_sen', 'Weighted_Speedup_1_sen',
    'CPU_Utilization_1_stream', 'Frequency_1_stream', 'IPC_1_stream', 'Misses_1_stream', 'LLC_1_stream', 'MBL_1_stream', 'Memory_Footprint_1_stream', 'Virt_Memory_1_stream', 'Res_Memory_1_stream', 'Weighted_Speedup_1_stream',
    'CPU_Utilization_1_insen', 'Frequency_1_insen', 'IPC_1_insen', 'Misses_1_insen', 'LLC_1_insen', 'MBL_1_insen', 'Memory_Footprint_1_insen', 'Virt_Memory_1_insen', 'Res_Memory_1_insen', 'Weighted_Speedup_1_insen',
    'CPU_Utilization_2_sen', 'Frequency_2_sen', 'IPC_2_sen', 'Misses_2_sen', 'LLC_2_sen', 'MBL_2_sen', 'Memory_Footprint_2_sen', 'Virt_Memory_2_sen', 'Res_Memory_2_sen', 'Weighted_Speedup_2_sen',
    'CPU_Utilization_2_stream', 'Frequency_2_stream', 'IPC_2_stream', 'Misses_2_stream', 'LLC_2_stream', 'MBL_2_stream', 'Memory_Footprint_2_stream', 'Virt_Memory_2_stream', 'Res_Memory_2_stream', 'Weighted_Speedup_2_stream',
    'CPU_Utilization_2_insen', 'Frequency_2_insen', 'IPC_2_insen', 'Misses_2_insen', 'LLC_2_insen', 'MBL_2_insen', 'Memory_Footprint_2_insen', 'Virt_Memory_2_insen', 'Res_Memory_2_insen', 'Weighted_Speedup_2_insen',
    'CPU_Utilization_3_sen', 'Frequency_3_sen', 'IPC_3_sen', 'Misses_3_sen', 'LLC_3_sen', 'MBL_3_sen', 'Memory_Footprint_3_sen', 'Virt_Memory_3_sen', 'Res_Memory_3_sen', 'Weighted_Speedup_3_sen',
    'CPU_Utilization_3_stream', 'Frequency_3_stream', 'IPC_3_stream', 'Misses_3_stream', 'LLC_3_stream', 'MBL_3_stream', 'Memory_Footprint_3_stream', 'Virt_Memory_3_stream', 'Res_Memory_3_stream', 'Weighted_Speedup_3_stream',
    'CPU_Utilization_3_insen', 'Frequency_3_insen', 'IPC_3_insen', 'Misses_3_insen', 'LLC_3_insen', 'MBL_3_insen', 'Memory_Footprint_3_insen', 'Virt_Memory_3_insen', 'Res_Memory_3_insen', 'Weighted_Speedup_3_insen',
    'CPU_Utilization_4_sen', 'Frequency_4_sen', 'IPC_4_sen', 'Misses_4_sen', 'LLC_4_sen', 'MBL_4_sen', 'Memory_Footprint_4_sen', 'Virt_Memory_4_sen', 'Res_Memory_4_sen', 'Weighted_Speedup_4_sen',
    'CPU_Utilization_4_stream', 'Frequency_4_stream', 'IPC_4_stream', 'Misses_4_stream', 'LLC_4_stream', 'MBL_4_stream', 'Memory_Footprint_4_stream', 'Virt_Memory_4_stream', 'Res_Memory_4_stream', 'Weighted_Speedup_4_stream',
    'CPU_Utilization_4_insen', 'Frequency_4_insen', 'IPC_4_insen', 'Misses_4_insen', 'LLC_4_insen', 'MBL_4_insen', 'Memory_Footprint_4_insen', 'Virt_Memory_4_insen', 'Res_Memory_4_insen', 'Weighted_Speedup_4_insen',
    'Weighted_Speedup_5', 'Weighted_Speedup_6', 'Weighted_Speedup_7', 'Weighted_Speedup_8', 'Weighted_Speedup_9', 'Weighted_Speedup_10', 'Weighted_Speedup_11',
    'Weighted_Speedup_12', 'Weighted_Speedup_13', 'Weighted_Speedup_14', 'Weighted_Speedup_15', 'Weighted_Speedup_16', 'Weighted_Speedup_17'
]

data = pd.read_csv("Seq2seq_dataset\\213\\seq2seq_213.csv", sep=',')

test_data = pd.read_csv("Seq2seq_dataset\\43\\seq2seq_43_test.csv", sep=',')
# test_data = (test_data-data.min())/(data.max()-data.min())
print(test_data)

state_input_1 = test_data.loc[:, column_index[:30]].values
state_input_2 = test_data.loc[:, column_index[30:60]].values
state_input_3 = test_data.loc[:, column_index[60:90]].values
state_input_4 = test_data.loc[:, column_index[90:120]].values
target_5 = test_data.loc[:, column_index[120]].values
target_6 = test_data.loc[:, column_index[121]].values
target_7 = test_data.loc[:, column_index[122]].values
target_8 = test_data.loc[:, column_index[123]].values
target_9 = test_data.loc[:, column_index[124]].values
target_10 = test_data.loc[:, column_index[125]].values
target_11 = test_data.loc[:, column_index[126]].values
target_12 = test_data.loc[:, column_index[127]].values
target_13 = test_data.loc[:, column_index[128]].values
target_14 = test_data.loc[:, column_index[129]].values
target_15 = test_data.loc[:, column_index[130]].values
target_16 = test_data.loc[:, column_index[131]].values
target_17 = test_data.loc[:, column_index[132]].values

input_test = []
output_test = []

for i in range(len(state_input_1)):
    input_test.append(np.array([state_input_1[i], state_input_2[i], state_input_3[i], state_input_4[i]]))
    output_test.append(np.array(
        [
            np.array([target_5[i]]), np.array([target_6[i]]), np.array([target_7[i]]), np.array([target_8[i]]),
            np.array([target_9[i]]), np.array([target_10[i]]), np.array([target_11[i]]), np.array([target_12[i]]),
            np.array([target_13[i]]), np.array([target_14[i]]), np.array([target_15[i]]), np.array([target_16[i]]),
            np.array([target_17[i]])
        ]
    ))

input_test = np.array(input_test)
output_test = np.array(output_test)

train_data = pd.read_csv("Seq2seq_dataset\\43\\seq2seq_43_train.csv", sep=',')
# train_data = (train_data-data.min())/(data.max()-data.min())
print(train_data)

state_input_1 = train_data.loc[:, column_index[:30]].values
state_input_2 = train_data.loc[:, column_index[30:60]].values
state_input_3 = train_data.loc[:, column_index[60:90]].values
state_input_4 = train_data.loc[:, column_index[90:120]].values
target_5 = train_data.loc[:, column_index[120]].values
target_6 = train_data.loc[:, column_index[121]].values
target_7 = train_data.loc[:, column_index[122]].values
target_8 = train_data.loc[:, column_index[123]].values
target_9 = train_data.loc[:, column_index[124]].values
target_10 = train_data.loc[:, column_index[125]].values
target_11 = train_data.loc[:, column_index[126]].values
target_12 = train_data.loc[:, column_index[127]].values
target_13 = train_data.loc[:, column_index[128]].values
target_14 = train_data.loc[:, column_index[129]].values
target_15 = train_data.loc[:, column_index[130]].values
target_16 = train_data.loc[:, column_index[131]].values
target_17 = train_data.loc[:, column_index[132]].values

input_train = []
output_train = []

for i in range(len(state_input_1)):
    input_train.append(np.array([state_input_1[i], state_input_2[i], state_input_3[i], state_input_4[i]]))
    output_train.append(np.array(
        [
            np.array([target_5[i]]), np.array([target_6[i]]), np.array([target_7[i]]), np.array([target_8[i]]),
            np.array([target_9[i]]), np.array([target_10[i]]), np.array([target_11[i]]), np.array([target_12[i]]),
            np.array([target_13[i]]), np.array([target_14[i]]), np.array([target_15[i]]), np.array([target_16[i]]),
            np.array([target_17[i]])
        ]
    ))

input_train = np.array(input_train)
output_train = np.array(output_train)

encode_steps = 4
encode_features = 30
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

model.compile(loss="mse", optimizer='adam', metrics=['mae'])

model.load_weights('seq2seq_weights.h5')

for layer in model.layers[:3]:
    layer.trainable = False
for layer in model.layers[3:]:
    layer.trainable = True

model.fit(
    input_train, output_train,
    batch_size=250, epochs=400,
    validation_data=(input_test, output_test),
    verbose=1, shuffle=True
)

print(model.evaluate(input_test, output_test))
model.save_weights("seq2seq_weights_43.h5")
