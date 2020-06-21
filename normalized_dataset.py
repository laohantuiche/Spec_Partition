import pandas as pd


def normalized_svm_data_stream():
    svm_data_stream = pd.read_csv("ST_or_CS_44.csv", sep=',')
    svm_data_stream = (svm_data_stream-svm_data_stream-svm_data_stream.min())/(svm_data_stream.max()-svm_data_stream.min())
    svm_data_stream.to_csv("normalized_ST_or_CS.csv")


def normalized_svm_data_protected():
    svm_data_protected = pd.read_csv("need_protection_44.csv", sep=',')
    svm_data_protected = (svm_data_protected-svm_data_protected.min())/(svm_data_protected.max()-svm_data_protected.min())
    svm_data_protected.to_csv("normalized_need_protected_dataset.csv")


def normalized_seq2seq_dataset():
    seq2seq_data = pd.read_csv("seq2seq.csv", sep=',')
    seq2seq_data = (seq2seq_data-seq2seq_data.min())/(seq2seq_data.max()-seq2seq_data.min())
    seq2seq_data.to_csv("normalized_seq2seq.csv")


normalized_svm_data_stream()
normalized_svm_data_protected()
normalized_seq2seq_dataset()
