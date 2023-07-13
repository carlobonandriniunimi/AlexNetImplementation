import tensorflow as tf
from tensorflow import keras

from utils import *

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load train and test images-labels
(train_images, train_labels), (test_images, test_labels) = \
    keras.datasets.cifar10.load_data()
# Split test data into validation and test data
n_train = 100
n_test = 50
train_images, train_labels = train_images[:n_train], train_labels[:n_train]
validation_images, validation_labels = (
    test_images[:n_test], test_labels[:n_test])
test_images, test_labels = (
    test_images[n_test:n_test*2], test_labels[n_test:n_test*2])

# Transform data into Tensorflow dataset
train_df = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_df = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_df = tf.data.Dataset.from_tensor_slices(
    (validation_images, validation_labels))

# Visualize 5 images
# plt.figure(figsize=(20, 20))
# for i, (image, label) in enumerate(train_df.take(5)):
#     ax = plt.subplot(5, 5, i+1)
#     plt.imshow(image)
#     plt.title(CLASS_NAMES[label.numpy()[0]])
#     plt.axis('off')
# plt.show()

# Check size of datasets
train_size = train_df.cardinality().numpy()
test_size = test_df.cardinality().numpy()
validation_size = validation_df.cardinality().numpy()
print("Training data size:", train_size)
print("Testing data size:", test_size)
print("validation data size:", validation_size)

# Process, shuffle and batch data
train_df = (train_df
            .map(process_images)
            .shuffle(buffer_size=train_size)
            .batch(batch_size=32, drop_remainder=True))
test_df = (test_df
           .map(process_images)
           .shuffle(buffer_size=test_size)
           .batch(batch_size=32, drop_remainder=True))
validation_df = (validation_df
                 .map(process_images)
                 .shuffle(buffer_size=validation_size)
                 .batch(batch_size=32, drop_remainder=True))

# Model definition
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
                        activation='relu', input_shape=(277, 277, 3)),
    keras.layers.BatchNormalization(),  # Try LayerNormalization
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    # Padding same means that the padding will be put evenly on either side
    keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
                        activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                        activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                        activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                        activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

# Sets up tensorboard for training visualization
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# Compilation of the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              metrics=['accuracy'])
model.summary()

# Training
model.fit(train_df, epochs=10, validation_data=validation_df,
          validation_freq=1, callbacks=[tensorboard_cb])

# CMD: tensorboard --logdir logs, from this directory

# Evaluating model
model.evaluate(test_df)
