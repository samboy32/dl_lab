# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Embedding,
    RepeatVector,
    Concatenate,
    Dropout
)

from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

DATASET_PATH = "images"

IMAGE_SIZE = (224, 224)

IMAGES_PER_CLASS = 20

EMBEDDING_DIM = 128
LSTM_UNITS = 256

EPOCHS = 30
BATCH_SIZE = 16

CLASS_NAMES = [
    "dog",
    "tiger",
    "elephant",
    "lion",
    "zebra"
]


# ------------------------------------------------------------
# 3. SELECT A SUBSET OF IMAGENET IMAGES
# ------------------------------------------------------------

image_paths = []
image_labels = []

for class_name in CLASS_NAMES:

    class_folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(class_folder):
        print("Folder not found:", class_folder)
        continue

    files = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    files = files[:IMAGES_PER_CLASS]

    for file_name in files:

        image_path = os.path.join(
            class_folder,
            file_name
        )

        image_paths.append(image_path)
        image_labels.append(class_name)


print("Total images:", len(image_paths))

print("\nClass distribution:")

for class_name in CLASS_NAMES:
    count = image_labels.count(class_name)
    print(class_name, ":", count)


# ------------------------------------------------------------
# 4. CONVERT IMAGE LABELS INTO TEXTUAL DESCRIPTIONS
# ------------------------------------------------------------

def label_to_caption(label):

    label = label.replace("_", " ")
    label = label.lower()

    caption = "a photo of a " + label + "."

    return caption


captions = [
    label_to_caption(label)
    for label in image_labels
]


print("\nExample captions:")

for i in range(min(10, len(captions))):

    print(
        image_labels[i],
        "->",
        captions[i]
    )


# ------------------------------------------------------------
# 5. TOKENIZE THE TEXTUAL DESCRIPTIONS
# ------------------------------------------------------------

caption_with_tokens = [
    "<START> " + caption + " <END>"
    for caption in captions
]


tokenizer = Tokenizer(
    filters="",
    lower=True,
    oov_token="<UNK>"
)

tokenizer.fit_on_texts(caption_with_tokens)


# ------------------------------------------------------------
# 6. CONSTRUCT VOCABULARY
# ------------------------------------------------------------

word_to_index = tokenizer.word_index

index_to_word = {
    index: word
    for word, index in word_to_index.items()
}

vocab_size = len(word_to_index) + 1

print("\nVocabulary:")
print(word_to_index)

print("\nVocabulary size:", vocab_size)


# ------------------------------------------------------------
# 7. CONVERT CAPTIONS INTO INTEGER TOKEN SEQUENCES
# ------------------------------------------------------------

sequences = tokenizer.texts_to_sequences(
    caption_with_tokens
)


print("\nExample token sequence:")

for i in range(min(5, len(sequences))):

    print(caption_with_tokens[i])
    print(sequences[i])
    print()


# ------------------------------------------------------------
# 8. DETERMINE MAXIMUM CAPTION LENGTH
# ------------------------------------------------------------

max_length = max(
    len(sequence)
    for sequence in sequences
)

print("Maximum caption length:", max_length)


# ------------------------------------------------------------
# 9. EXTRACT CNN FEATURES
# ------------------------------------------------------------

cnn = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

cnn.trainable = False


from tensorflow.keras.layers import GlobalAveragePooling2D

cnn_input = Input(
    shape=(224, 224, 3)
)

cnn_output = cnn(cnn_input)

feature_vector = GlobalAveragePooling2D()(
    cnn_output
)

cnn_encoder = Model(
    cnn_input,
    feature_vector
)


print(
    "\nCNN feature dimension:",
    cnn_encoder.output_shape
)


# ------------------------------------------------------------
# 10. EXTRACT IMAGE FEATURE VECTORS
# ------------------------------------------------------------

def extract_image_features(image_paths):

    features = []

    for image_path in image_paths:

        image = load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

        image = img_to_array(image)

        image = np.expand_dims(
            image,
            axis=0
        )

        image = preprocess_input(image)

        feature = cnn_encoder.predict(
            image,
            verbose=0
        )[0]

        features.append(feature)

    return np.array(features)


image_features = extract_image_features(
    image_paths
)


print(
    "\nExtracted feature matrix shape:",
    image_features.shape
)


# ------------------------------------------------------------
# 11. STORE CNN FEATURE VECTORS
# ------------------------------------------------------------

np.save(
    "image_features.npy",
    image_features
)

np.save(
    "caption_sequences.npy",
    np.array(sequences, dtype=object),
    allow_pickle=True
)

print("\nCNN features saved as image_features.npy")


# ------------------------------------------------------------
# 12. CREATE INPUT-OUTPUT SEQUENCES
# ------------------------------------------------------------

decoder_input_sequences = []
decoder_target_sequences = []


for sequence in sequences:

    input_sequence = sequence[:-1]

    target_sequence = sequence[1:]

    decoder_input_sequences.append(
        input_sequence
    )

    decoder_target_sequences.append(
        target_sequence
    )


decoder_input_sequences = pad_sequences(
    decoder_input_sequences,
    maxlen=max_length - 1,
    padding="post"
)

decoder_target_sequences = pad_sequences(
    decoder_target_sequences,
    maxlen=max_length - 1,
    padding="post"
)


print(
    "\nDecoder input shape:",
    decoder_input_sequences.shape
)

print(
    "Decoder target shape:",
    decoder_target_sequences.shape
)


# ------------------------------------------------------------
# 13. TRAIN-TEST SPLIT
# ------------------------------------------------------------

indices = np.arange(
    len(image_features)
)

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=image_labels
)


train_features = image_features[
    train_indices
]

test_features = image_features[
    test_indices
]

train_sequences = decoder_input_sequences[
    train_indices
]

test_sequences = decoder_input_sequences[
    test_indices
]

train_targets = decoder_target_sequences[
    train_indices
]

test_targets = decoder_target_sequences[
    test_indices
]


print("\nTraining images:", len(train_indices))
print("Testing images:", len(test_indices))


# ------------------------------------------------------------
# 14. BUILD CNN-LSTM MODEL
# ------------------------------------------------------------

image_input = Input(
    shape=(512,),
    name="cnn_feature"
)

caption_input = Input(
    shape=(max_length - 1,),
    name="caption_input"
)


# ------------------------------------------------------------
# IMAGE FEATURE PROCESSING
# ------------------------------------------------------------

image_dense = Dense(
    LSTM_UNITS,
    activation="relu"
)(image_input)


# ------------------------------------------------------------
# TEXT EMBEDDING
# ------------------------------------------------------------

embedding = Embedding(
    input_dim=vocab_size,
    output_dim=EMBEDDING_DIM,
    mask_zero=True
)(caption_input)


# ------------------------------------------------------------
# IMAGE FEATURE AS INITIAL LSTM STATE
# ------------------------------------------------------------

lstm_output = LSTM(
    LSTM_UNITS,
    return_sequences=True
)(
    embedding,
    initial_state=[
        image_dense,
        image_dense
    ]
)


# ------------------------------------------------------------
# WORD PREDICTION
# ------------------------------------------------------------

word_output = Dense(
    vocab_size,
    activation="softmax"
)(lstm_output)


# ------------------------------------------------------------
# COMPLETE CNN-LSTM MODEL
# ------------------------------------------------------------

caption_model = Model(
    inputs=[
        image_input,
        caption_input
    ],
    outputs=word_output
)


caption_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


caption_model.summary()


# ------------------------------------------------------------
# 15. TRAIN THE CNN-LSTM MODEL
# ------------------------------------------------------------

history = caption_model.fit(
    [
        train_features,
        train_sequences
    ],
    train_targets,
    validation_data=(
        [
            test_features,
            test_sequences
        ],
        test_targets
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


# ------------------------------------------------------------
# 16. PLOT TRAINING PERFORMANCE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN-LSTM Training Accuracy")
plt.legend()

plt.show()


# ------------------------------------------------------------
# 17. CAPTION GENERATION FUNCTION
# ------------------------------------------------------------

def generate_caption(feature):

    start_token = tokenizer.word_index["<start>"]

    end_token = tokenizer.word_index["<end>"]

    generated_sequence = [
        start_token
    ]


    for _ in range(max_length):

        current_sequence = pad_sequences(
            [generated_sequence],
            maxlen=max_length - 1,
            padding="post"
        )


        prediction = caption_model.predict(
            [
                feature.reshape(1, -1),
                current_sequence
            ],
            verbose=0
        )


        position = len(generated_sequence) - 1


        next_word_id = np.argmax(
            prediction[0, position]
        )


        if next_word_id == end_token:
            break


        if next_word_id == 0:
            break


        generated_sequence.append(
            next_word_id
        )


    words = []

    for token_id in generated_sequence:

        if token_id in (
            start_token,
            end_token,
            0
        ):
            continue

        word = index_to_word.get(
            token_id,
            ""
        )

        words.append(word)


    caption = " ".join(words)

    return caption


# ------------------------------------------------------------
# 18. GENERATE CAPTIONS FOR UNSEEN TEST IMAGES
# ------------------------------------------------------------

print("\n\nGenerated Captions")
print("============================")


for i in range(len(test_indices)):

    original_index = test_indices[i]

    feature = test_features[i]

    actual_label = image_labels[
        original_index
    ]

    expected_caption = captions[
        original_index
    ]

    generated_caption = generate_caption(
        feature
    )


    print("\nImage:",
          image_paths[original_index])

    print(
        "ImageNet Label:",
        actual_label
    )

    print(
        "Expected Caption:",
        expected_caption
    )

    print(
        "Generated Caption:",
        generated_caption
    )


# ------------------------------------------------------------
# 19. VISUALIZE IMAGE + GENERATED CAPTION
# ------------------------------------------------------------

def display_prediction(test_position):

    original_index = test_indices[
        test_position
    ]

    image_path = image_paths[
        original_index
    ]

    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    feature = test_features[
        test_position
    ]

    actual_label = image_labels[
        original_index
    ]

    expected_caption = captions[
        original_index
    ]

    generated_caption = generate_caption(
        feature
    )


    plt.figure(figsize=(7, 7))

    plt.imshow(image)

    plt.axis("off")

    plt.title(
        "ImageNet Label: " + actual_label +
        "\nExpected: " + expected_caption +
        "\nGenerated: " + generated_caption
    )

    plt.show()


display_prediction(0)


# ------------------------------------------------------------
# 20. DISPLAY MULTIPLE TEST IMAGES
# ------------------------------------------------------------

num_images_to_display = min(
    5,
    len(test_indices)
)


for i in range(num_images_to_display):

    display_prediction(i)


# ------------------------------------------------------------
# 21. SAVE TRAINED MODEL
# ------------------------------------------------------------

caption_model.save(
    "cnn_lstm_image_caption_model.keras"
)

print(
    "\nModel saved as:",
    "cnn_lstm_image_caption_model.keras"
)
