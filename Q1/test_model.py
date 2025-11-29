import numpy as np
import cv2
import glob
from keras.models import load_model
import matplotlib.pyplot as plt

# Load model 
MODEL_PATH = 'cnn_model.h5'

print(f"Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Check input shape from model
input_shape = model.input_shape[1:3]  
IMG_SIZE = input_shape[0]
print(f"Model expects image size: {IMG_SIZE}x{IMG_SIZE}")

# Classes
classes = ['Cat', 'Dog']

def predict_image(image_path):
    """Predict a single image"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None, None
    
    # Preprocess same as training
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_resized = image_resized / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)
    
    # Predict
    prediction = model.predict(image_resized, verbose=0)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx] * 100
    class_name = classes[class_idx]
    
    return class_name, confidence

# Test on provided test images
print("\n" + "="*60)
print("Testing on provided test images")
print("="*60)

test_images = []
for address in glob.glob("test\\*\\*.jpg"):
    test_images.append(address)

correct = 0
total = 0
wrong_predictions = []

for img_path in test_images:
    true_label = img_path.split("\\")[1]
    predicted, confidence = predict_image(img_path)
    
    if predicted:
        total += 1
        is_correct = (predicted == true_label)
        if is_correct:
            correct += 1
        else:
            wrong_predictions.append((img_path, true_label, predicted, confidence))
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {img_path.split('\\')[-1]}: {predicted} ({confidence:.2f}%) - True: {true_label}")

if total > 0:
    accuracy = (correct / total) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    if wrong_predictions:
        print(f"\n{len(wrong_predictions)} wrong prediction(s):")
        for img_path, true_label, predicted, confidence in wrong_predictions:
            print(f"  - {img_path.split('\\')[-1]}: Predicted {predicted} ({confidence:.2f}%) but was {true_label}")


