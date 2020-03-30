import tensorflow as tf
import numpy as np
from scipy.stats import stats
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255, test_images / 255

def build_nn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def bootstrapping():
    model = build_nn_model()
    #model.load_weights("../result/model/20200118-085651-496.h5")   sample
    model.load_weights("E:/experiments/MNIST_FL_1/model/20200317-171952-491-0.9456.h5")
    print("==> bootstrapping start")

    n_bootstraps = 10000
    rng_seed = 3033  # control reproducibility

    bootstrapped_auroc = []
    bootstrapped_auprc = []
    bootstrapped_sen = []
    bootstrapped_spe = []
    bootstrapped_bac = []
    bootstrapped_f1 = []
    bootstrapped_pre = []
    bootstrapped_NLR = []
    bootstrapped_PLR = []
    final = {}

    result = model.predict(test_images)
    auroc = metrics.roc_auc_score(test_labels, result, multi_class='ovr')
    print("auroc ovr : ", auroc)
    auroc_ovo = metrics.roc_auc_score(test_labels, result, multi_class='ovo')
    print("auroc ovo : ", auroc_ovo)

    result = np.argmax(result, axis=1)
    auprc = metrics.auc(test_labels, result)
    print("auprc : ", auprc)





    '''
    fpr = dict()
    tpr = dict()

    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], result[:, i])

    print(fpr, tpr)
    
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, result)
    #roc_auc = metrics.auc(fpr, tpr)
    '''
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(test_labels, result)

    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    result = np.argmax(result, axis=1)

    cf = metrics.confusion_matrix(test_labels, result)
    print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])

    t = pd.concat([pd.DataFrame(thresholds), pd.DataFrame(tpr), pd.DataFrame(1-fpr), pd.DataFrame(((1-fpr+tpr)/2))], axis=1)
    t.columns = ['threshold', 'sensitivity', 'specificity', 'bac']
    t_ = t.iloc[np.min(np.where(t['bac'] == max(t['bac']))), :]
    y_pred_ = (result >= t_['threshold']).astype(bool)

    cm_ = metrics.confusion_matrix(test_labels, result)
    tp = cm_[1, 1]
    fn = cm_[1, 0]
    fp = cm_[0, 1]
    tn = cm_[0, 0]

    bac = t_['bac']  # balanced accuracy
    sensitivity = t_['sensitivity']  # sensitivity
    specificity = t_['specificity']  # specificity
    precision = tp / (tp + fp)  # precision
    f1 = 2 * ((sensitivity * precision) / (sensitivity + precision))  # f1 score
    plr = sensitivity / (1 - specificity)  # PLR
    nlr = (1 - sensitivity) / specificity  # NLR

    rng = np.random.RandomState(rng_seed)

    y_true = np.array(test_labels)
    for j in range(n_bootstraps):
        indices = rng.random_integers(0, len(result)-1, len(result))

        if len(np.unique(y_true[indices])) < 2:
            continue

        auroc_ = metrics.roc_auc_score(y_true[indices], result[indices])
        precision_, recall_, thresholds_ = metrics.precision_recall_curve(y_true[indices], result[indices])
        auprc_ = metrics.auc(recall_, precision_)
        CM = metrics.confusion_matrix(np.array(y_true)[indices], result.argmax(axis=1))
        TP = CM[1, 1]
        FN = CM[1, 0]
        FP = CM[0, 1]
        TN = CM[0, 0]

        TPV = TP / (TP + FN)  # sensitivity
        TNV = TN / (TN + FP)  # specificity
        PPV = TP / (TP + FP)  # precision
        BAAC = (TPV + TNV) / 2  # balanced accuracy
        F1 = 2 * ((PPV * TPV) / (PPV + TPV))  # f1 score
        PLR = TPV / (1 - TNV)  # LR+
        NLR = (1 - TPV) / TNV  # LR-

        bootstrapped_auroc.append(auroc_)  # AUROC
        bootstrapped_auprc.append(auprc_)  # AUPRC
        bootstrapped_sen.append(TPV)  # Sensitivity
        bootstrapped_spe.append(TNV)  # Specificity
        bootstrapped_bac.append(BAAC)  # Balanced Accuracy
        bootstrapped_f1.append(F1)  # F1 score
        bootstrapped_pre.append(PPV)  # Precision
        bootstrapped_NLR.append(NLR)  # Negative Likelihood Ratio
        bootstrapped_PLR.append(PLR)  # positive Likelihood Ratio

    sorted_auroc = np.array(bootstrapped_auroc)
    sorted_auroc.sort()
    sorted_auprc = np.array(bootstrapped_auprc)
    sorted_auprc.sort()
    sorted_sen = np.array(bootstrapped_sen)
    sorted_sen.sort()
    sorted_spe = np.array(bootstrapped_spe)
    sorted_spe.sort()
    sorted_bac = np.array(bootstrapped_bac)
    sorted_bac.sort()
    sorted_f1 = np.array(bootstrapped_f1)
    sorted_f1.sort()
    sorted_pre = np.array(bootstrapped_pre)
    sorted_pre.sort()
    sorted_NLR = np.array(bootstrapped_NLR)
    sorted_NLR.sort()
    sorted_PLR = np.array(bootstrapped_PLR)
    sorted_PLR.sort()

    auroc_lower = round(sorted_auroc[int(0.025 * len(sorted_auroc))], 4)
    auroc_upper = round(sorted_auroc[int(0.975 * len(sorted_auroc))], 4)
    auprc_lower = round(sorted_auprc[int(0.025 * len(sorted_auprc))], 4)
    auprc_upper = round(sorted_auprc[int(0.975 * len(sorted_auprc))], 4)
    sen_lower = round(sorted_sen[int(0.025 * len(sorted_sen))], 4)
    sen_upper = round(sorted_sen[int(0.975 * len(sorted_sen))], 4)
    spe_lower = round(sorted_spe[int(0.025 * len(sorted_spe))], 4)
    spe_upper = round(sorted_spe[int(0.975 * len(sorted_spe))], 4)
    bac_lower = round(sorted_bac[int(0.025 * len(sorted_bac))], 4)
    bac_upper = round(sorted_bac[int(0.975 * len(sorted_bac))], 4)
    f1_lower = round(sorted_f1[int(0.025 * len(sorted_f1))], 4)
    f1_upper = round(sorted_f1[int(0.975 * len(sorted_f1))], 4)
    pre_lower = round(sorted_pre[int(0.025 * len(sorted_pre))], 4)
    pre_upper = round(sorted_pre[int(0.975 * len(sorted_pre))], 4)
    NLR_lower = round(sorted_NLR[int(0.025 * len(sorted_NLR))], 4)
    NLR_upper = round(sorted_NLR[int(0.975 * len(sorted_NLR))], 4)
    PLR_lower = round(sorted_PLR[int(0.025 * len(sorted_PLR))], 4)
    PLR_upper = round(sorted_PLR[int(0.975 * len(sorted_PLR))], 4)

    auroc_true_ci = str(round(auroc, 4)) + " (" + str(auroc_lower) + ", " + str(auroc_upper) + ")"
    auprc_true_ci = str(round(auprc, 4)) + " (" + str(auprc_lower) + ", " + str(auprc_upper) + ")"
    sen_true_ci = str(round(sensitivity, 4)) + " (" + str(sen_lower) + ", " + str(sen_upper) + ")"
    spe_true_ci = str(round(specificity, 4)) + " (" + str(spe_lower) + ", " + str(spe_upper) + ")"
    bac_true_ci = str(round(bac, 4)) + " (" + str(bac_lower) + ", " + str(bac_upper) + ")"
    f1_true_ci = str(round(f1, 4)) + " (" + str(f1_lower) + ", " + str(f1_upper) + ")"
    pre_true_ci = str(round(precision, 4)) + " (" + str(pre_lower) + ", " + str(pre_upper) + ")"
    NLR_true_ci = str(round(nlr, 4)) + " (" + str(NLR_lower) + ", " + str(NLR_upper) + ")"
    PLR_true_ci = str(round(plr, 4)) + " (" + str(PLR_lower) + ", " + str(PLR_upper) + ")"
    #
    col_n = ['thresholds', 'sensitivity', 'specificity', 'precision', 'bacc', 'f1', 'PLR', 'NLR', 'AUROC',
             'AUPRC']

    final = {"thresholds": round(t_['threshold'], 4),
                "sensitivity": sen_true_ci, "specificity": spe_true_ci,
                "precision": pre_true_ci, "bacc": bac_true_ci,
                "f1": f1_true_ci, "PLR": PLR_true_ci, "NLR": NLR_true_ci,
                "AUROC": auroc_true_ci, "AUPRC": auprc_true_ci}
    final = pd.DataFrame(final, index=[0])
    #final1 = pd.DataFrame(final)
    final = final.reindex(columns=col_n)


    total_item = {"thresholds": round(t_['threshold'], 4),
                "sensitivity": sorted_sen, "specificity": sorted_spe,
                "precision": sorted_pre, "bacc": sorted_bac,
                "f1": sorted_f1, "PLR": sorted_PLR, "NLR": sorted_NLR,
                "AUROC": sorted_auroc, "AUPRC": sorted_auprc}
    total_pd = pd.DataFrame.from_dict(total_item, orient='columns')

    print(total_pd)

    final2 = pd.DataFrame.append(final, total_pd)
    final2.to_csv("fl_1_bootstrapping.csv", mode="w")

    print("==> bootstrapping end")

    t_test_result = stats.ttest_1samp(sorted_auroc, 0.999)

    print("t-test : ", t_test_result)

def validation():
    model = build_nn_model()
    #model.load_weights("C:/Project/FL_Client/result/model/20200118-152832-2879.h5")
    #model.load_weights("C:/Project/FL_Client/result/model/20200201-170659-200-0.9702.h5")
    #model.load_weights("C:/Project/FL_Client/result/model/20200316-190933-500-0.8901.h5")   # FL4-500-200
    #model.load_weights("../FL4/model/20200318-212844-2651-0.8905-0.8905.h5")   # FL4-3000-600
    model.load_weights("../result/model/20200118-085651-496.h5")   # FL3-500
    #model.load_weights("result/model/20200118-152832-2879.h5")
    #model.fit()

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

    print("test acc : ", test_acc, " / test loss : ", test_loss)

    result = model.predict(test_images)

    auroc = metrics.roc_auc_score(test_labels, result, multi_class='ovr')
    print("auroc ovr : ", auroc)
    auroc = metrics.roc_auc_score(test_labels, result, multi_class='ovo')
    print("auroc ovo : ", auroc)

    result = np.argmax(result, axis=1)
    cm = confusion_matrix(test_labels, result)
    print("cm : \n", cm)
    acc = accuracy_score(test_labels, result)
    print("acc : {}".format(acc))
    f1 = f1_score(test_labels, result, average=None)
    f2 = f1_score(test_labels, result, average='micro')
    f3 = f1_score(test_labels, result, average='macro')
    f4 = f1_score(test_labels, result, average='weighted')
    print("f1 None : {}".format(f1))
    print("f2 micro : {}".format(f2))
    print("f3 macro : {}".format(f3))
    print("f4 weighted : {}".format(f4))

def show_confusion_matrix():

    cm_mnist_skewed = np.array([[967, 0, 0, 1, 0, 2, 8, 1, 1, 0],
               [0, 1107, 4, 2, 0, 1, 4, 2, 15, 0],
               [14, 12, 901, 10, 9, 2, 19, 18, 42, 5],
               [6, 2, 30, 875, 0, 33, 4, 24, 26, 10],
               [4, 1, 12, 0, 888, 0, 11, 3, 12, 51],
               [17, 8, 8, 41, 11, 724, 24, 13, 37, 9],
               [11, 3, 6, 0, 7, 6, 921, 1, 3, 0],
               [3, 10, 31, 2, 3, 0, 0, 949, 3, 27],
               [12, 13, 13, 17, 9, 26, 18, 15, 841, 10] ,
               [10, 8, 2, 7, 39, 6, 1, 47, 10, 879]])


    cm_mnist_unbalance = np.array([[965,	0,	0,	1,	0,	1,	10,	1,	2,	0],
                                   [0,	1118,	2,	2,	0,	0,	4,	2,	7,	0],
                                   [15,	33,	870,	16,	13,	1,	25,	19,	40,	0],
                                   [5,	5,	24,	895,	0,	32,	5,	26,	16,	2],
                                   [6,	5,	10,	1,	914,	1,	13,	5,	6,	21],
                                   [19,	14,	7,	51,	14,	705,	28,	10,	36,	8],
                                   [14,	3,	6,	0,	6,	5,	919,	3,	2,	0],
                                   [3,	17,	26,	2,	4,	0,	0,	963,	2,	11],
                                   [15,	24,	19,	27,	10,	29,	18,	21,	803,	8],
                                   [18,	13,	7,	11,	85,	7,	0,	106,	13,	749]])


    cm_mnist = np.array([[969, 0, 1, 0, 0, 2, 2, 2, 2, 2],
                          [0, 1121, 3, 1, 0, 1, 3, 2, 4, 0],
                          [4, 2, 1012, 2, 2, 0, 2, 4, 4, 0],
                          [0, 0, 2, 992, 0, 4, 0, 5, 4, 3],
                          [3, 0, 3, 0, 952, 0, 3, 2, 2, 12],
                          [3, 0, 0, 7, 1, 868, 6, 1, 4, 2],
                          [5, 3, 2, 1, 5, 3, 939, 0, 0, 0],
                          [0, 2, 8, 4, 0, 0, 0, 1007, 2, 5],
                          [6, 0, 3, 5, 6, 5, 4, 4, 938, 3],
                          [3, 2, 0, 6, 13, 2, 1, 7, 6, 969]])

    cm_mnist_fl = np.array([[967, 0, 1, 1, 0, 3, 4, 1, 1, 2],
                           [0, 1122, 2, 1, 0, 1, 5, 1, 3, 0],
                           [5, 2, 998, 6, 3, 1, 4, 8, 5, 0],
                           [0, 0, 6, 983, 0, 5, 1, 7, 5, 3],
                           [2, 0, 3, 1, 950, 0, 5, 2, 2, 17],
                           [5, 1, 0, 13, 1, 853, 9, 1, 6, 3],
                           [5, 3, 1, 1, 7, 5, 936, 0, 0, 0],
                           [1, 9, 10, 4, 3, 1, 0, 991, 0, 9],
                           [3, 2, 5, 7, 5, 7, 3, 4, 938, 0],
                           [4, 5, 1, 7, 17, 2, 1, 5, 3, 964]])


    mimic_benchmark = np.array([[2786, 76],
                                [253, 121]])

    mimic_fl_1000 = np.array([[2819, 43],
                                [288, 86]])

    mimic_fl_3000 = np.array([[2789,73], [256, 118]])

    plot_confusion_matrix(cm_mnist_unbalance, normalize=False)

def plot_confusion_matrix(cm,
                          target_names=[],
                          title='',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()


    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     )


    #plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.xlabel('Predicted label')
    plt.show()

if __name__ == "__main__":
    #validation()
    #show_confusion_matrix()
    bootstrapping()