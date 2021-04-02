import argparse
import collections
import csv
import json

import load

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_recall_fscore_support
from tensorflow import keras
import scipy.stats as sst
import numpy as np
import sklearn.metrics as skm
from tensorflow.python.keras import models

import architecture

def predict(parser):
    val = load.load_dataset("data/validation_2.json")
    preproc = load.preproc(*val)

    args = parser.parse_args()
    print("args model : ", args.model)

    model = architecture.build_model()
    model.load_weights(args.model)

    with open("data/validation_2.json", "rb") as fid:
        val_labels = [json.loads(l)['labels'] for l in fid]

    counts = collections.Counter(preproc.class_to_int[l[0]] for l in val_labels)
    counts = sorted(counts.most_common(), key=lambda x: x[0])
    counts = list(zip(*counts))[1]

    print("counts : " , counts)

    smooth = 500
    counts = np.array(counts)[None, None, :]
    total = np.sum(counts) + counts.shape[1]
    print("total : ", total)
    prior = (counts + smooth) / float(total)    # ???
    print("prior : ", prior)

    ecgs, committee_labels = preproc.process(*val)
    m_probs = model.predict(ecgs)

    committee_labels = np.argmax(committee_labels, axis=2)
    committee_labels = committee_labels[:, 0]

    print("===================")
    temp = []
    preds = np.argmax(m_probs / prior, axis = 2)
    for i, j in zip(preds, val_labels):
        t = sst.mode(i[:len(j)-1])[0][0]
        temp.append(t)
        #print(i[:len(j)-1])

    preds = temp

    #print("preds : \n", preds)

    report = skm.classification_report(committee_labels, preds, target_names=preproc.classes, digits=3)
    scores = skm.precision_recall_fscore_support(committee_labels, preds, average=None)
    print("report : \n", report)


    cm = confusion_matrix(committee_labels, preds)
    print("confusion matrix : \n", cm)

    f1 = f1_score(committee_labels, preds, average='micro')
    #print("f1_score : ", f1)

    
    # ***roc_auc_score - m_probs***
    s_probs = np.sum(m_probs, axis=1)
    s_probs = s_probs / 71  # one data set max size (element count) -> normalization


    #ovo_auroc = roc_auc_score(committee_labels, s_probs, multi_class='ovo')
    ovr_auroc = roc_auc_score(committee_labels, s_probs, multi_class='ovr')

    print("ovr_auroc : ", ovr_auroc)
    #print("ovo_auroc : ", ovo_auroc)

    '''
        bootstrapping
    '''
    n_bootstraps = 100
    np.random.seed(3033)

    total_precision = []
    total_recall = []
    total_f1 = []
    total_auroc = []

    precision = []
    recall = []
    f1 = []

    total = []

    for j in range(n_bootstraps):
        indices = np.random.random_integers(0, len(m_probs) -1, 100)

        #print("indices : ", len(indices))

        if len(np.unique(committee_labels[indices])) < 2:
            continue


        sub_labels = []
        sub_result = []
        sub_probs = []

        #print(indices)

        for i in indices:
            sub_labels.append(committee_labels[i])
            sub_result.append(preds[i])
            sub_probs.append(m_probs[i])

        s_scores = precision_recall_fscore_support(sub_labels, sub_result, labels=[0, 1, 2, 3], average=None)

        # ***roc_auc_score - m_probs***
        s_p = np.sum(sub_probs, axis=1)
        s_p = s_p / 71  # one data set max size (element count) -> normalization

        # ovo_auroc = roc_auc_score(committee_labels, s_probs, multi_class='ovo')
        #print(sub_labels)
        #print(s_p)

        try:
            s_auroc = roc_auc_score(sub_labels, s_p, multi_class='ovr')
        except:
            s_auroc = -1

        #print(s_scores)
        precision.append(np.array(s_scores[0]))
        recall.append(np.array(s_scores[1]))
        f1.append(np.array(s_scores[2]))
        #auroc.append(s_auroc)

        total_precision.append(np.average(s_scores[0]))
        total_recall.append(np.average(s_scores[1]))
        total_f1.append(np.average(s_scores[2]))
        total_auroc.append(s_auroc)

    total_precision.sort()
    total_recall.sort()
    total_f1.sort()
    total_auroc.sort()

    total_auroc.remove(-1)
    #print(total_auroc)

    '''
        bootstrapping 시 클래스가 존재하지 않는 케이스가 있을수도 있음 
    '''
    precision = np.array(precision)
    precision[precision == .0] = np.nan
    recall = np.array(recall)
    recall[recall == .0] = np.nan
    f1 = np.array(f1)
    f1[f1 == .0] = np.nan

    #print(total_auroc)

    for i in range(4):
        pre = precision[:, i]
        pre.sort()
        rec = recall[:, i]
        rec.sort()
        f = f1[:, i]
        f.sort()

        pre = np.round(pre[int(len(pre) * 0.025): int(len(pre) * 0.975)], 3)
        rec = np.round(rec[int(len(rec) * 0.025): int(len(rec) * 0.975)], 3)
        f = np.round(pre[int(len(f) * 0.025): int(len(f) * 0.975)], 3)

        '''
        print(i,
              " : ", "{0} ({1}, {2})".format(np.round(np.nanmean(pre), 3), round(pre[0], 3), round(pre[-1], 3)),
              " : ", "{0} ({1}, {2})".format(np.round(np.nanmean(rec), 3), round(rec[0], 3), round(rec[-1], 3)),
              " : ", "{0} ({1}, {2})".format(np.round(np.nanmean(f), 3), round(f[0], 3), round(f[-1], 3)))
        '''

        item = [i,
                "{0} ({1}, {2})".format(np.round(np.nanmean(pre), 3), round(np.nanmin(pre), 3), round(np.nanmax(pre), 3)),
                "{0} ({1}, {2})".format(np.round(np.nanmean(rec), 3), round(np.nanmin(rec), 3), round(np.nanmax(rec), 3)),
                "{0} ({1}, {2})".format(np.round(np.nanmean(f), 3), round(np.nanmin(f), 3), round(np.nanmax(f), 3))]

        total.append(item)

    total_auroc = np.round(total_auroc[int(len(total_auroc) * 0.025): int(len(total_auroc) * 0.975)], 3)
    total_precision = np.round(total_precision[int(len(total_precision) * 0.025): int(len(total_precision) * 0.975)], 3)
    total_recall = np.round(total_recall[int(len(total_recall) * .025): int(len(total_recall) * .975)], 3)
    total_f1 = np.round(total_f1[int(len(total_f1) * .025): int(len(total_f1) * .975)], 3)


    with open(args.file_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["", "precision", "recall", "f1-score", "auroc"])
        writer.writerow(["",
                         "{0} ({1}, {2})".format(np.round(np.average(scores[0]), 3), total_precision[0], total_precision[-1]),
                         "{0} ({1}, {2})".format(np.round(np.average(scores[1]), 3), total_recall[0], total_recall[-1]),
                         "{0} ({1}, {2})".format(np.round(np.average(scores[2]), 3), total_f1[0], total_f1[-1]),
                         "{0} ({1}, {2})".format(np.round(ovr_auroc, 3), total_auroc[0], total_auroc[-1]),
                         ])
        for i in total:
            writer.writerow(i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/model/base.hmd5")
    parser.add_argument("--file_name", default="ecg.csv")

    predict(parser)
