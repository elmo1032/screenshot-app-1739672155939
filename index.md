Certainly! Below is a simple example of how to train a neural network using Python with the popular deep learning library TensorFlow and Keras. This example demonstrates how to build and train a neural network on the MNIST dataset, which consists of handwritten digits.

### Prerequisites
Make sure you have TensorFlow installed. You can install it using pip:

```bash
pip install tensorflow
```

### Example Code

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0  # Normalize to [0, 1]
x_test = x_test.astype('float32') / 255.0    # Normalize to [0, 1]

# Reshape the data to fit the model (28x28 images to 784-dimensional vectors)
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the neural network model
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dropout(0.2))  # Dropout layer for regularization
model.add(layers.Dense(10, activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

### Explanation
1. **Data Loading**: The MNIST dataset is loaded, which contains 60,000 training images and 10,000 test images of handwritten digits.
2. **Preprocessing**: The images are normalized to the range [0, 1] and reshaped into vectors of size 784 (28x28).
3. **One-Hot Encoding**: The labels are converted to one-hot encoded format.
4. **Model Building**: A simple feedforward neural network is created with one hidden layer of 128 neurons and a dropout layer for regularization.
5. **Model Compilation**: The model is compiled with the Adam optimizer and categorical crossentropy loss function.
6. **Model Training**: The model is trained for 10 epochs with a batch size of 32, using 20% of the training data for validation.
7. **Model Evaluation**: The model is evaluated on the test set, and the accuracy is printed.
8. **Visualization**: Training and validation accuracy and loss are plotted for analysis.

### Running the Code
You can run this code in a Python environment that supports TensorFlow, such as Jupyter Notebook, Google Colab, or any local Python setup with TensorFlow installed.


Powered by https://www.blackbox.ai