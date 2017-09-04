from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

EPOCHS = 10
BATCH_SIZE = 128
rate = 0.001

def normalize(data):
    if (len(data.shape) != 4):
        print('Expect a 4D array to normalize, refusing to normalize')
        return None;
    return  (data - data.mean()) / data.std()

def greyscale(data):
    return np.sum(data/3, axis=3, keepdims=True)


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    wc1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean=mu, stddev=sigma))
    bc1 = tf.Variable(tf.truncated_normal([6], mean=mu, stddev=sigma))
    conv1 = tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, bc1)

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    p1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    wc2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma))
    bc2 = tf.Variable(tf.truncated_normal([16], mean=mu, stddev=sigma))
    conv2 = tf.nn.conv2d(p1, wc2, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, bc2)

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    p2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 3: Convolutional. Output = 2x2x100.
    wc3 = tf.Variable(tf.truncated_normal([4, 4, 16, 100], mean=mu, stddev=sigma))
    bc3 = tf.Variable(tf.truncated_normal([100], mean=mu, stddev=sigma))
    conv3 = tf.nn.conv2d(p2, wc3, strides=[1, 1, 1, 1], padding='VALID')
    conv3 = tf.nn.bias_add(conv3, bc3)

    # Activation.
    p3 = tf.nn.relu(conv3)

    # Flatten. Input = 2x2x100. Output = 400.
    flat = flatten(p3)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    wfc1 = tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma))
    bfc1 = tf.Variable(tf.truncated_normal([120], mean=mu, stddev=sigma))
    fc1 = tf.add(tf.matmul(flat, wfc1), bfc1)

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)

#     # Layer 4: Fully Connected. Input = 120. Output = 84.
#     wfc2 = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma))
#     bfc2 = tf.Variable(tf.truncated_normal([84], mean=mu, stddev=sigma))
#     fc2 = tf.add(tf.matmul(fc1, wfc2), bfc2)

#     # Activation.
#     fc2 = tf.nn.relu(fc2)

#     # Dropout
#     fc2 = tf.nn.dropout(fc2, keep_prob)

#     # Layer 5: Fully Connected. Input = 84. Output = 43.
#     wout = tf.Variable(tf.truncated_normal([84, 43], mean=mu, stddev=sigma))
#     bout = tf.Variable(tf.truncated_normal([43], mean=mu, stddev=sigma))
#     logits = tf.add(tf.matmul(fc2, wout), bout)

    # Layer 5: Fully Connected. Input = 120. Output = 10.
    wout = tf.Variable(tf.truncated_normal([120, 10], mean=mu, stddev=sigma))
    bout = tf.Variable(tf.truncated_normal([10], mean=mu, stddev=sigma))
    logits = tf.add(tf.matmul(fc1, wout), bout)

    return logits

(X, Y), (X_test, y_test) = cifar10.load_data()

# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.
Y = Y.reshape(-1)
y_test = y_test.reshape(-1)

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.3, random_state=42, stratify = Y)


n_train = X_train.shape[0]
n_validation = X_valid.shape[0]
n_test = X_test.shape[0]
image_shape = X_train.shape[1:]
n_classes = np.unique(y_train).size

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

X_train_grey = greyscale(X_train)
X_train_norm_grey = normalize(X_train_grey)
X_valid_grey = greyscale(X_valid)
X_test_grey = greyscale(X_test)

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 10)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_loss = 0
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_loss += (loss * len(batch_x))
        total_accuracy += (accuracy * len(batch_x))
    return (total_loss / num_examples, total_accuracy / num_examples)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()

    for i in range(EPOCHS):
        shuffled_x, shuffled_y = shuffle(X_train_norm_grey, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = shuffled_x[offset:end], shuffled_y[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        validation_loss, validation_accuracy = evaluate(X_valid_grey, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Loss = {:.3f}".format(validation_loss))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    _, test_accuracy = evaluate(X_test_grey, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
