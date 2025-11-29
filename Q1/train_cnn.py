import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import layers, Sequential
import matplotlib.pyplot as plt

# DATA
data_list = []
label_list = []

print("Loading images...")
for i, address in enumerate(glob.glob("train\\*\\*.jpg")):
    image = cv2.imread(address)
    if image is not None:
        image = cv2.resize(image, (128, 128))  
        image = image/255
        
        data_list.append(image)
        label_list.append(address.split("\\")[1])
        
        if i%200 == 0:
            print(f'{i} images processed...')


X = np.array(data_list)
y = np.array(label_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(f"Training set: {len(X_train)} images")
print(f"Test set: {len(X_test)} images")
print(f"Classes: {le.classes_}")

# MODEL
nn = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPool2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

nn.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

print("Training CNN model...")
H = nn.fit(X_train, y_train,
           validation_data=(X_test, y_test),
           epochs=10,
           batch_size=64)

# Save model
nn.save('cnn_model.h5')
print("Model saved")

# EVALUATE
plt.plot(H.history['accuracy'], label='train accuracy')
plt.plot(H.history['val_accuracy'], label='test accuracy')
plt.plot(H.history['loss'], label='loss')
plt.plot(H.history['val_loss'], label='test loss')
plt.legend()
plt.show()

# Final test accuracy
test_loss, test_acc = nn.evaluate(X_test, y_test, verbose=0)
print(f"Final test accuracy: {test_acc:.4f}")
