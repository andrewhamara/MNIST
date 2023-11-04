import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    )

def normalize_image(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

ds_train = ds_train.map(
    normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

predictions = model.predict(ds_test.unbatch().batch(ds_info.splits['test'].num_examples))
predicted_labels = np.argmax(predictions, axis=1)

true_labels, test_images = [], []

for image, label in tfds.as_numpy(ds_test.unbatch()):
    true_labels.append(label)
    test_images.append(image)

true_labels = np.array(true_labels)
test_images = np.array(test_images)

mismatches = np.where(predicted_labels != true_labels)[0]

with PdfPages('misclassified.pdf') as pdf:
    for idx in mismatches:
        plt.figure(figsize=(8,8))
        plt.imshow(test_images[idx].squeeze(), cmap='gray')
        plt.title(f"Actual: {true_labels[idx]}, Predicted: {predicted_labels[idx]}")
        plt.axis('off')
        pdf.savefig()
    plt.close()
