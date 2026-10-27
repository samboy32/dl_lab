# ============================================================
# TRANSFER LEARNING USING VGG16
# ImageNet Feature Extraction and Classification
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

path = "dataset"

classes = [
    "airliner",
    "sports_car",
    "tiger"
]

X = []
y = []

for i, class_name in enumerate(classes):

    folder = os.path.join(path, class_name)

    for file in os.listdir(folder):

        file_path = os.path.join(folder, file)

        img = image.load_img(
            file_path,
            target_size=(224, 224)
        )

        img = image.img_to_array(img)

        X.append(img)
        y.append(i)

X = np.array(X)
y = np.array(y)

print("Images:", X.shape)
print("Labels:", y.shape)


# ------------------------------------------------------------
# 3. LOAD PRE-TRAINED VGG16
# ------------------------------------------------------------

vgg = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze VGG16
vgg.trainable = False


# ------------------------------------------------------------
# 4. FEATURE EXTRACTION
# ------------------------------------------------------------

# Remove original classifier
# Apply Global Average Pooling

extractor = Model(
    inputs=vgg.input,
    outputs=GlobalAveragePooling2D()(vgg.output)
)

# VGG16 preprocessing
X = preprocess_input(X)

# Extract features
features = extractor.predict(X)

print("Feature Vector Shape:", features.shape)
print("Feature Vector Dimension:", features.shape[1])


# ------------------------------------------------------------
# 5. SAVE FEATURES
# ------------------------------------------------------------

np.save("features.npy", features)
np.save("labels.npy", y)

print("Features saved successfully.")


# ------------------------------------------------------------
# 6. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    features,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ------------------------------------------------------------
# 7. CREATE NEW CLASSIFICATION LAYER
# ------------------------------------------------------------

model = Sequential([
    Dense(
        len(classes),
        activation="softmax",
        input_shape=(features.shape[1],)
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ------------------------------------------------------------
# 8. TRAIN CLASSIFIER
# ------------------------------------------------------------

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    verbose=1
)


# ------------------------------------------------------------
# 9. TEST ON UNSEEN IMAGES
# ------------------------------------------------------------

probabilities = model.predict(X_test)

# Select class with highest probability
predictions = np.argmax(
    probabilities,
    axis=1
)


# ------------------------------------------------------------
# 10. CALCULATE PERFORMANCE
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted"
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)


print("\n================ RESULTS ================")

print("Feature Vector Dimension:", features.shape[1])
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ------------------------------------------------------------
# 11. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot()

plt.title("VGG16 Transfer Learning - Confusion Matrix")
plt.show()
