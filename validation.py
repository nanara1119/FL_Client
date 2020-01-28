import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

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
    model.load_weights("result/model/20200118-152832-2879.h5")

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

    print(test_acc, test_loss)

    result = model.predict(test_images)
    result = np.argmax(result, axis=1)

    cm = confusion_matrix(test_labels, result)
    print("cm : ", cm)
    acc = accuracy_score(test_labels, result)
    print("acc : {}".format(acc))
    f1 = f1_score(test_labels, result, average=None)
    f2 = f1_score(test_labels, result, average='micro')
    f3 = f1_score(test_labels, result, average='macro')
    f4 = f1_score(test_labels, result, average='weighted')
    print("f1 : {}".format(f1))
    print("f2 : {}".format(f2))
    print("f3 : {}".format(f3))
    print("f4 : {}".format(f4))

if __name__ == "__main__":
    validation()