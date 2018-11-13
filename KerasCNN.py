import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train=tf.keras.utils.normalize(x_train, 1)
x_test=tf.keras.utils.normalize(x_test, 1)

model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=input_shape))
model.add(tf.keras.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.Flatten())
model.add(tf.keras.Dense(1000, activation='relu'))
model.add(tf.keras.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tbCallBack=tf.keras.callbacks.TensorBoard(log_dir="output", histogram_freq=0, write_graph=True, write_images=True)
model.fit(x_train, y_train, epochs=3, callbacks=[tbCallBack])

#
# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()