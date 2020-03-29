import tensorflow as tf
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
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
    validation()
    #show_confusion_matrix()