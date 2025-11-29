import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import layers, Sequential
import matplotlib.pyplot as plt

# DATA
print("Loading MNIST training data...")
train_data = pd.read_csv('mnist_train.csv')
X_train_full = train_data.iloc[:, 1:].values  
y_train_full = train_data.iloc[:, 0].values   

print("Loading MNIST test data...")
test_data = pd.read_csv('mnist_test.csv')
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Reshape to 28x28 images
X_train_full = X_train_full.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Normalize pixel values
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")
print(f"Test set: {len(X_test)} images")

# MODEL
nn = Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPool2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
])

nn.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

print("Training MNIST model...")
H = nn.fit(X_train, y_train,
           validation_data=(X_val, y_val),
           epochs=10,
           batch_size=128,
           verbose=1)

# Evaluate on test set
test_loss, test_acc = nn.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Save model
nn.save('mnist_model.h5')
print("Model saved to mnist_model.h5")

# EVALUATE
plt.plot(H.history['accuracy'], label='train accuracy')
plt.plot(H.history['val_accuracy'], label='val accuracy')
plt.plot(H.history['loss'], label='loss')
plt.plot(H.history['val_loss'], label='val loss')
plt.legend()
plt.show()

