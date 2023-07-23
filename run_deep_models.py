import tensorflow as tf
seed = 0
tf.random.set_seed(seed)


from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.regularizers import L1L2

import numpy as np
seed_np = 42
np.random.seed(seed_np)
import pandas as pd
import statistics as st
import pickle
import sys
import numpy.random as random
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score


def my_model(n_steps, n_features, hid):
    model = Sequential()
    #model.add(LSTM(hid, activation='relu', input_shape=(n_steps, 1))) #LSTM
    model.add(LSTM(hid, activation='relu', return_sequences=True, input_shape=(n_steps, 1))) #stacked LSTM
    model.add(LSTM(hid, activation='relu')) #stacked LSTM
    #model.add(Bidirectional(LSTM(hid, activation='relu'), input_shape=(n_steps,  1))) #Bi-LSTM
    #model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, 1))) #CNN-LSTM
    #model.add(TimeDistributed(MaxPooling1D(pool_size=2))) #CNN-LSTM
    #model.add(TimeDistributed(Flatten())) #CNN-LSTM
    #model.add(LSTM(hid, activation='relu')) #CNN-LSTM
    #model.add(Flatten()) #MLP
    #model.add(Dense(hid, activation='relu')) #MLP
    model.add(Dense(1, activation='sigmoid'))
    return model


pat_enc_dict_p = {} 

# positive annotated dataset
data_p = pd.read_csv('POS_Data.csv', sep=',', encoding='Latin-1')
k_lis_p = data_p['patient_id'].tolist() 

for i in range(len(k_lis_p)): 
    pat_dk_p = k_lis_p[i]
    pat_enc_dict_p[pat_dk_p] = ''

pat_enc_dict_n = {} 

# negative annotated dataset
data_n = pd.read_csv('NEG_Data.csv', sep=',', encoding='Latin-1')
k_lis_n = data_n['patient_id'].tolist() 

for i in range(len(k_lis_n)): 
    pat_dk_n = k_lis_n[i] 
    pat_enc_dict_n[pat_dk_n] = ''

pat_lab_dict = {} 

#lab test EHR data
data = pd.read_csv('LAB_Data.csv', sep=',', encoding='Latin-1') 
k_lis = data['patient_id'].tolist() 
e_lis = data['enc_number'].tolist() 
d_lis = data['DBP'].tolist() #dbp
s_lis = data['SBP'].tolist() #sbp

unq_vocab_lis = []

for i in range(len(k_lis)):
    pat_dk = k_lis[i]
    enc = e_lis[i]
    dbp = d_lis[i]
    #dbp = s_lis[i]

    dbp = dbp.replace('[', ' ')
    dbp = dbp.replace(']', ' ')
    dbp = dbp.strip()
    dbp_lis = dbp.split(',')
    if len(dbp_lis) != 1 and '' not in dbp_lis:
        dbp_lis = [float(tok) for tok in dbp_lis]
        dbp_lis = [st.mean(dbp_lis)] 
        if pat_dk in pat_lab_dict:
            pat_lab_dict[pat_dk] = pat_lab_dict[pat_dk] + dbp_lis
        else:
            pat_lab_dict[pat_dk] = dbp_lis
        unq_vocab_lis = unq_vocab_lis + dbp_lis
        unq_vocab_lis = list(set(unq_vocab_lis))
    

docs = []
# define class labels as (1)effective/not effective(0)
labels_ = []

pos_lis_ = []
for ptk_p, encs in pat_enc_dict_p.items():
    if ptk_p in pat_lab_dict:
        pos_lis_.append(pat_lab_dict[ptk_p])     
        labels_.append(1)

pos_lis = []
labels = []
pos_duration_lis = []
count = 0
count_2 = 0
avg_dur = 2


for i in range(len(pos_lis_)):
    pos_pat = pos_lis_[i]
    if pos_pat:
        first_dbp = pos_pat[0]
        lab = labels_[i]
        ind = next((i for (i,dbp_v) in enumerate(pos_pat) if dbp_v < first_dbp), None)
        if ind is not None:
            temp_dbp_lis = [pos_pat[i] for i in range(ind+1)]
            pos_lis.append(temp_dbp_lis) 
            labels.append(lab)
            pos_duration_lis.append(ind+1)
            count +=1
        else:
            count_2 +=1

pos_duration_lis_unq = list(set(pos_duration_lis))
avg_drug_eff_dur = st.mean(pos_duration_lis) 

labels__ = []
neg_lis_ = []
for ptk_n, encs in pat_enc_dict_n.items():
    if ptk_n in pat_lab_dict:
        neg_lis_.append(pat_lab_dict[ptk_n])    
        labels__.append(0)

neg_lis = []
neg_duration_lis = []
count_neg = 0
count_neg_2 = 0
avg_dur_neg = 2


for i in range(len(neg_lis_)):
    neg_pat = neg_lis_[i]
    if neg_pat:
        first_dbp_neg = neg_pat[0]
        lab_neg = labels__[i]
        ind_neg = next((i for (i,dbp_v) in enumerate(neg_pat) if dbp_v > first_dbp_neg), None)
        if ind_neg is not None:
            temp_dbp_lis_neg = [neg_pat[i] for i in range(ind_neg+1)]
            neg_lis.append(temp_dbp_lis_neg) 
            labels.append(lab_neg)
            neg_duration_lis.append(ind_neg+1)
            count_neg +=1
        else:
            count_neg_2 +=1

neg_duration_lis_unq = list(set(neg_duration_lis))
avg_drug_eff_dur_neg = st.mean(neg_duration_lis) 

pos_true_lis = [tok for tok in labels if tok == 1]   
neg_true_lis = [tok for tok in labels if tok == 0]   

docs = pos_lis + neg_lis

max_doc_len_lis = [len(d) for d in docs]
max_doc_len = max(max_doc_len_lis)
 
labels = np.asarray(labels)

scaler = MinMaxScaler() #for normalization to (0-1) range

from collections import Counter

vocab_size = len(unq_vocab_lis)
encoded_docs = docs[:]
enc_doc_len_here = [len(doc) for doc in encoded_docs]
mean_len_ = st.mean(enc_doc_len_here)
max_len_ = max(enc_doc_len_here)
min_len_ = min(enc_doc_len_here)
freq_dict_len_ = Counter(enc_doc_len_here)
enc_doc_len_here = list(set(enc_doc_len_here))

max_length = max_doc_len
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

X = padded_docs[:]
Y = labels[:]

## compile the model and train via kfold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores_acc = []
cvscores_loss = []
epc = 100
output_dim = 8
hid = 50
acc_tr_lis = []
acc_vl_lis = []
loss_tr_lis = []
loss_vl_lis = []

prec_pos = [] #class 1
prec_neg = [] #class 0
prec_all = []
rec_pos = []
rec_neg = []
rec_all = []
f1_pos = []
f1_neg = []
f1_all = []
y_true_all = []
y_pred_all = []
threshold = 0.5
y_pred_all_orig = []
acc_all = []
auroc_all = []
aucpr_all_2 = []

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

for train, test in kfold.split(X, Y):
    model = my_model(vocab_size, output_dim, hid)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    X_train = scaler.fit_transform(X[train])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #for lstm variants
    history = model.fit(X_train, Y[train], epochs=epc, validation_split = 0.2, verbose=0)
    acc_tr = history.history['accuracy']
    loss_tr = history.history['loss']
    acc_vl = history.history['val_accuracy']
    loss_vl = history.history['val_loss']
    acc_tr_lis.append(acc_tr)
    acc_vl_lis.append(acc_vl)
    loss_tr_lis.append(loss_tr)
    loss_vl_lis.append(loss_vl)

    X_test = scaler.transform(X[test])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))#for lstm variants
    scores = model.evaluate(X_test, Y[test], verbose=0) 
    cvscores_acc.append(scores[1] * 100)
    cvscores_loss.append(scores[0])
    y_pred_ = model.predict(X_test) 
    y_pred_new = [tok[0] for tok in y_pred_]
    y_pred = np.where(np.array(y_pred_new) > threshold, 1,0) 
    prec_all_fold, rec_all_fold, f1_all_fold, _ = precision_recall_fscore_support(Y[test], y_pred, labels=[0,1])
    prec_all_fold = st.mean(prec_all_fold)
    rec_all_fold = st.mean(rec_all_fold)
    f1_all_fold = st.mean(f1_all_fold)
    acc_fold = accuracy_score(Y[test], y_pred)
    acc_all.append(acc_fold)
    average_precision = average_precision_score(Y[test], y_pred)
    aucpr_all_2.append(average_precision)
    prec_all.append(prec_all_fold)
    rec_all.append(rec_all_fold)
    f1_all.append(f1_all_fold)
    y_true_all = y_true_all + Y[test].tolist()
    y_pred_all = y_pred_all + y_pred.tolist()
    y_pred_all_orig = y_pred_all_orig + y_pred_new


print('prec_all', prec_all)
print('len(prec_all)', len(prec_all))
print('rec_all', rec_all)
print('len(rec_all)', len(rec_all))
print('f1_all', f1_all)
print('len(f1_all)', len(f1_all))
print('acc_all', acc_all)
print('len(acc_all)', len(acc_all))
print('aucpr_all_2', aucpr_all_2)
print('len(aucpr_all_2)', len(aucpr_all_2))

prec_all_score = st.mean(prec_all)
print('avg prec', prec_all_score)

rec_all_score = st.mean(rec_all)
print('avg rec', rec_all_score)

f1_all_score = st.mean(f1_all)
print('avg f1', f1_all_score)

acc_score = st.mean(acc_all)
print('acc_score', acc_score)

auc_roc_both = roc_auc_score(y_true_all, y_pred_all, average = 'micro', labels=[0,1])
print('auc_score', auc_roc_both)

avg_prec_mean = st.mean(aucpr_all_2)
print('avg_prec_mean', avg_prec_mean)

print('finish')
