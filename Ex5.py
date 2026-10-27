# ============================================================
# EXERCISE 5
# TEXT-TO-IMAGE SYNTHESIS USING CONDITIONAL GAN
#
# Text Encoder      : Word Embedding + Dense Encoder
# Generator         : Conditional Generator
# Discriminator     : Conditional Discriminator
# Dataset           : ImageNet-style class-labelled images
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
    Embedding,
    Flatten,
    Concatenate,
    Reshape,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    BatchNormalization,
    Dropout
)

from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

DATASET_PATH = "images"

IMAGE_SIZE = 64

CHANNELS = 3

# Number of images from each class
IMAGES_PER_CLASS = 20

# Conditional GAN configuration
NOISE_DIM = 100
TEXT_EMBEDDING_DIM = 128
TEXT_ENCODER_DIM = 128

GENERATOR_FILTERS = 64
DISCRIMINATOR_FILTERS = 64

EPOCHS = 100
BATCH_SIZE = 16

LEARNING_RATE = 0.0002
BETA_1 = 0.5


# ------------------------------------------------------------
# 3. SELECTED IMAGENET-STYLE CLASSES
# ------------------------------------------------------------

CLASS_NAMES = [
    "dog",
    "tiger",
    "elephant",
    "lion",
    "zebra"
]


# ------------------------------------------------------------
# 4. LOAD IMAGE PATHS AND LABELS
# ------------------------------------------------------------

image_paths = []
image_labels = []


for class_name in CLASS_NAMES:

    class_folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.exists(class_folder):

        print(
            "Folder not found:",
            class_folder
        )

        continue


    files = [
        f
        for f in os.listdir(class_folder)
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            )
        )
    ]


    files = files[:IMAGES_PER_CLASS]


    for file_name in files:

        image_path = os.path.join(
            class_folder,
            file_name
        )

        image_paths.append(
            image_path
        )

        image_labels.append(
            class_name
        )


print(
    "\nTotal images:",
    len(image_paths)
)


print(
    "\nClass distribution:"
)


for class_name in CLASS_NAMES:

    count = image_labels.count(
        class_name
    )

    print(
        class_name,
        ":",
        count
    )


# ------------------------------------------------------------
# 5. CONVERT IMAGE LABELS INTO TEXTUAL DESCRIPTIONS
# ------------------------------------------------------------

def label_to_caption(label):

    label = label.replace(
        "_",
        " "
    )

    label = label.lower()

    caption = (
        "a photo of a "
        + label
    )

    return caption


captions = [
    label_to_caption(label)
    for label in image_labels
]


print(
    "\nExample textual descriptions:"
)


for i in range(
    min(10, len(captions))
):

    print(
        image_labels[i],
        "->",
        captions[i]
    )


# ------------------------------------------------------------
# 6. TOKENIZE TEXT
# ------------------------------------------------------------

tokenizer = Tokenizer(
    lower=True,
    oov_token="<UNK>"
)


tokenizer.fit_on_texts(
    captions
)


sequences = tokenizer.texts_to_sequences(
    captions
)


vocab_size = (
    len(tokenizer.word_index)
    + 1
)


max_length = max(
    len(sequence)
    for sequence in sequences
)


print(
    "\nVocabulary:",
    tokenizer.word_index
)


print(
    "\nVocabulary size:",
    vocab_size
)


print(
    "Maximum text length:",
    max_length
)


# ------------------------------------------------------------
# 7. PAD TEXT SEQUENCES
# ------------------------------------------------------------

text_sequences = pad_sequences(
    sequences,
    maxlen=max_length,
    padding="post"
)


print(
    "\nText sequence shape:",
    text_sequences.shape
)


# ------------------------------------------------------------
# 8. LOAD AND PREPROCESS IMAGES
# ------------------------------------------------------------

def load_images(image_paths):

    images = []


    for image_path in image_paths:

        image = load_img(
            image_path,
            target_size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )


        image = img_to_array(
            image
        )


        # Normalize pixel values
        # from [0, 255] to [-1, 1]

        image = (
            image / 127.5
        ) - 1.0


        images.append(
            image
        )


    return np.array(
        images,
        dtype=np.float32
    )


images = load_images(
    image_paths
)


print(
    "\nImage tensor shape:",
    images.shape
)


# ------------------------------------------------------------
# 9. TRAIN-TEST SPLIT
# ------------------------------------------------------------

indices = np.arange(
    len(images)
)


train_indices, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=image_labels
)


train_images = images[
    train_indices
]


test_images = images[
    test_indices
]


train_text = text_sequences[
    train_indices
]


test_text = text_sequences[
    test_indices
]


train_labels = [
    image_labels[i]
    for i in train_indices
]


test_labels = [
    image_labels[i]
    for i in test_indices
]


print(
    "\nTraining images:",
    len(train_images)
)


print(
    "Testing images:",
    len(test_images)
)


# ------------------------------------------------------------
# 10. BUILD TEXT ENCODER
# ------------------------------------------------------------

text_input = Input(
    shape=(max_length,),
    name="text_input"
)


# Word embedding layer

word_embedding = Embedding(
    input_dim=vocab_size,
    output_dim=TEXT_EMBEDDING_DIM,
    name="word_embedding"
)(
    text_input
)


# Flatten word embeddings

flattened_text = Flatten()(
    word_embedding
)


# Convert text representation
# into fixed-size text embedding

text_dense = Dense(
    TEXT_ENCODER_DIM,
    activation="relu",
    name="text_encoder_dense"
)(
    flattened_text
)


text_encoder = Model(
    text_input,
    text_dense,
    name="Text_Encoder"
)


print(
    "\nText Encoder:"
)


text_encoder.summary()


# ------------------------------------------------------------
# 11. BUILD CONDITIONAL GENERATOR
# ------------------------------------------------------------

noise_input = Input(
    shape=(NOISE_DIM,),
    name="random_noise"
)


generator_text_input = Input(
    shape=(max_length,),
    name="generator_text"
)


# Encode textual description

generator_text_embedding = text_encoder(
    generator_text_input
)


# Combine random noise and text

combined_generator_input = Concatenate(
    name="noise_text_combination"
)(
    [
        noise_input,
        generator_text_embedding
    ]
)


# ------------------------------------------------------------
# GENERATOR INITIAL DENSE LAYER
# ------------------------------------------------------------

x = Dense(
    8 * 8 * GENERATOR_FILTERS * 4,
    activation="relu"
)(
    combined_generator_input
)


x = Reshape(
    (
        8,
        8,
        GENERATOR_FILTERS * 4
    )
)(
    x
)


x = BatchNormalization()(x)


# ------------------------------------------------------------
# UPSAMPLING 1
# 8x8 -> 16x16
# ------------------------------------------------------------

x = Conv2DTranspose(
    GENERATOR_FILTERS * 2,
    kernel_size=4,
    strides=2,
    padding="same"
)(
    x
)


x = BatchNormalization()(x)

x = LeakyReLU(
    alpha=0.2
)(
    x
)


# ------------------------------------------------------------
# UPSAMPLING 2
# 16x16 -> 32x32
# ------------------------------------------------------------

x = Conv2DTranspose(
    GENERATOR_FILTERS,
    kernel_size=4,
    strides=2,
    padding="same"
)(
    x
)


x = BatchNormalization()(x)

x = LeakyReLU(
    alpha=0.2
)(
    x
)


# ------------------------------------------------------------
# UPSAMPLING 3
# 32x32 -> 64x64
# ------------------------------------------------------------

x = Conv2DTranspose(
    GENERATOR_FILTERS // 2,
    kernel_size=4,
    strides=2,
    padding="same"
)(
    x
)


x = BatchNormalization()(x)

x = LeakyReLU(
    alpha=0.2
)(
    x
)


# ------------------------------------------------------------
# OUTPUT IMAGE
# ------------------------------------------------------------

generated_image = Conv2D(
    CHANNELS,
    kernel_size=3,
    padding="same",
    activation="tanh",
    name="generated_image"
)(
    x
)


generator = Model(
    [
        noise_input,
        generator_text_input
    ],
    generated_image,
    name="Conditional_Generator"
)


print(
    "\nConditional Generator:"
)


generator.summary()


# ------------------------------------------------------------
# 12. BUILD CONDITIONAL DISCRIMINATOR
# ------------------------------------------------------------

discriminator_image_input = Input(
    shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        CHANNELS
    ),
    name="image_input"
)


discriminator_text_input = Input(
    shape=(max_length,),
    name="discriminator_text"
)


# ------------------------------------------------------------
# IMAGE FEATURE EXTRACTION
# ------------------------------------------------------------

y = Conv2D(
    DISCRIMINATOR_FILTERS,
    kernel_size=4,
    strides=2,
    padding="same"
)(
    discriminator_image_input
)


y = LeakyReLU(
    alpha=0.2
)(
    y
)


y = Dropout(
    0.3
)(
    y
)


# 64x64 -> 32x32


y = Conv2D(
    DISCRIMINATOR_FILTERS * 2,
    kernel_size=4,
    strides=2,
    padding="same"
)(
    y
)


y = BatchNormalization()(y)

y = LeakyReLU(
    alpha=0.2
)(
    y
)


y = Dropout(
    0.3
)(
    y
)


# 32x32 -> 16x16


y = Conv2D(
    DISCRIMINATOR_FILTERS * 4,
    kernel_size=4,
    strides=2,
    padding="same"
)(
    y
)


y = BatchNormalization()(y)

y = LeakyReLU(
    alpha=0.2
)(
    y
)


y = Dropout(
    0.3
)(
    y
)


# 16x16 -> 8x8


y = Conv2D(
    DISCRIMINATOR_FILTERS * 8,
    kernel_size=4,
    strides=2,
    padding="same"
)(
    y
)


y = BatchNormalization()(y)

y = LeakyReLU(
    alpha=0.2
)(
    y
)


# ------------------------------------------------------------
# FLATTEN IMAGE FEATURES
# ------------------------------------------------------------

image_features = Flatten()(
    y
)


# ------------------------------------------------------------
# ENCODE TEXT
# ------------------------------------------------------------

discriminator_text_embedding = text_encoder(
    discriminator_text_input
)


# ------------------------------------------------------------
# COMBINE IMAGE + TEXT
# ------------------------------------------------------------

combined_discriminator_input = Concatenate(
    name="image_text_combination"
)(
    [
        image_features,
        discriminator_text_embedding
    ]
)


# ------------------------------------------------------------
# DISCRIMINATOR CLASSIFICATION
# ------------------------------------------------------------

validity = Dense(
    1,
    activation="sigmoid",
    name="real_fake"
)(
    combined_discriminator_input
)


discriminator = Model(
    [
        discriminator_image_input,
        discriminator_text_input
    ],
    validity,
    name="Conditional_Discriminator"
)


print(
    "\nConditional Discriminator:"
)


discriminator.summary()


# ------------------------------------------------------------
# 13. COMPILE DISCRIMINATOR
# ------------------------------------------------------------

discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=BETA_1
)


discriminator.compile(
    optimizer=discriminator_optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 14. BUILD COMPLETE CONDITIONAL GAN
# ------------------------------------------------------------

# Freeze discriminator while training generator

discriminator.trainable = False


gan_noise_input = Input(
    shape=(NOISE_DIM,),
    name="gan_noise"
)


gan_text_input = Input(
    shape=(max_length,),
    name="gan_text"
)


# Generate image

gan_generated_image = generator(
    [
        gan_noise_input,
        gan_text_input
    ]
)


# Ask discriminator whether
# generated image matches the text

gan_validity = discriminator(
    [
        gan_generated_image,
        gan_text_input
    ]
)


conditional_gan = Model(
    [
        gan_noise_input,
        gan_text_input
    ],
    gan_validity,
    name="Conditional_GAN"
)


conditional_gan.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    ),
    loss="binary_crossentropy"
)


print(
    "\nConditional GAN:"
)


conditional_gan.summary()


# ------------------------------------------------------------
# 15. RESTORE DISCRIMINATOR TRAINABILITY
# ------------------------------------------------------------

discriminator.trainable = True


# ------------------------------------------------------------
# 16. ADVERSARIAL TRAINING FUNCTION
# ------------------------------------------------------------

def train_gan(
    train_images,
    train_text,
    epochs,
    batch_size
):

    generator_losses = []

    discriminator_losses = []

    discriminator_accuracies = []


    num_batches = (
        len(train_images)
        // batch_size
    )


    for epoch in range(epochs):

        epoch_generator_loss = []

        epoch_discriminator_loss = []

        epoch_discriminator_accuracy = []


        # Shuffle training data

        permutation = np.random.permutation(
            len(train_images)
        )


        shuffled_images = train_images[
            permutation
        ]


        shuffled_text = train_text[
            permutation
        ]


        for batch in range(num_batches):

            start = (
                batch
                * batch_size
            )

            end = start + batch_size


            real_images = shuffled_images[
                start:end
            ]


            real_text = shuffled_text[
                start:end
            ]


            current_batch_size = len(
                real_images
            )


            if current_batch_size == 0:
                continue


            # ------------------------------------------------
            # GENERATE RANDOM LATENT VECTORS
            # ------------------------------------------------

            noise = np.random.normal(
                0,
                1,
                (
                    current_batch_size,
                    NOISE_DIM
                )
            ).astype(
                np.float32
            )


            # ------------------------------------------------
            # GENERATE FAKE IMAGES
            # ------------------------------------------------

            fake_images = generator.predict(
                [
                    noise,
                    real_text
                ],
                verbose=0
            )


            # ------------------------------------------------
            # REAL IMAGE LABELS
            # ------------------------------------------------

            real_labels = np.ones(
                (
                    current_batch_size,
                    1
                ),
                dtype=np.float32
            )


            # ------------------------------------------------
            # FAKE IMAGE LABELS
            # ------------------------------------------------

            fake_labels = np.zeros(
                (
                    current_batch_size,
                    1
                ),
                dtype=np.float32
            )


            # ------------------------------------------------
            # TRAIN DISCRIMINATOR ON REAL IMAGES
            # ------------------------------------------------

            real_result = discriminator.train_on_batch(
                [
                    real_images,
                    real_text
                ],
                real_labels
            )


            # ------------------------------------------------
            # TRAIN DISCRIMINATOR ON FAKE IMAGES
            # ------------------------------------------------

            fake_result = discriminator.train_on_batch(
                [
                    fake_images,
                    real_text
                ],
                fake_labels
            )


            # Keras may return:
            # [loss, accuracy]

            real_loss = float(
                real_result[0]
                if isinstance(
                    real_result,
                    (list, tuple)
                )
                else real_result
            )


            fake_loss = float(
                fake_result[0]
                if isinstance(
                    fake_result,
                    (list, tuple)
                )
                else fake_result
            )


            if isinstance(
                real_result,
                (list, tuple)
            ):

                real_accuracy = float(
                    real_result[1]
                )

            else:

                real_accuracy = 0.0


            if isinstance(
                fake_result,
                (list, tuple)
            ):

                fake_accuracy = float(
                    fake_result[1]
                )

            else:

                fake_accuracy = 0.0


            discriminator_loss = (
                real_loss
                + fake_loss
            ) / 2


            discriminator_accuracy = (
                real_accuracy
                + fake_accuracy
            ) / 2


            # ------------------------------------------------
            # TRAIN GENERATOR
            # ------------------------------------------------

            noise = np.random.normal(
                0,
                1,
                (
                    current_batch_size,
                    NOISE_DIM
                )
            ).astype(
                np.float32
            )


            generator_result = conditional_gan.train_on_batch(
                [
                    noise,
                    real_text
                ],
                real_labels
            )


            generator_loss = float(
                generator_result[0]
                if isinstance(
                    generator_result,
                    (list, tuple)
                )
                else generator_result
            )


            epoch_discriminator_loss.append(
                discriminator_loss
            )


            epoch_discriminator_accuracy.append(
                discriminator_accuracy
            )


            epoch_generator_loss.append(
                generator_loss
            )


        # ----------------------------------------------------
        # STORE EPOCH RESULTS
        # ----------------------------------------------------

        average_generator_loss = np.mean(
            epoch_generator_loss
        )


        average_discriminator_loss = np.mean(
            epoch_discriminator_loss
        )


        average_discriminator_accuracy = np.mean(
            epoch_discriminator_accuracy
        )


        generator_losses.append(
            average_generator_loss
        )


        discriminator_losses.append(
            average_discriminator_loss
        )


        discriminator_accuracies.append(
            average_discriminator_accuracy
        )


        print(
            "Epoch",
            epoch + 1,
            "/",
            epochs,
            "- Generator Loss:",
            round(
                average_generator_loss,
                4
            ),
            "- Discriminator Loss:",
            round(
                average_discriminator_loss,
                4
            ),
            "- Discriminator Accuracy:",
            round(
                average_discriminator_accuracy,
                4
            )
        )


    return (
        generator_losses,
        discriminator_losses,
        discriminator_accuracies
    )


# ------------------------------------------------------------
# 17. TRAIN CONDITIONAL GAN
# ------------------------------------------------------------

print(
    "\nStarting Conditional GAN training..."
)


(
    generator_losses,
    discriminator_losses,
    discriminator_accuracies
) = train_gan(
    train_images,
    train_text,
    EPOCHS,
    BATCH_SIZE
)


# ------------------------------------------------------------
# 18. PLOT GENERATOR AND DISCRIMINATOR LOSSES
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    generator_losses,
    label="Generator Loss"
)


plt.plot(
    discriminator_losses,
    label="Discriminator Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Conditional GAN Training Loss"
)


plt.legend()


plt.show()


# ------------------------------------------------------------
# 19. PLOT DISCRIMINATOR ACCURACY
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    discriminator_accuracies,
    label="Discriminator Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "Conditional Discriminator Accuracy"
)


plt.legend()


plt.show()


# ------------------------------------------------------------
# 20. TEXT-TO-IMAGE GENERATION FUNCTION
# ------------------------------------------------------------

def generate_image_from_text(
    text,
    noise=None
):

    # Convert text into tokens

    sequence = tokenizer.texts_to_sequences(
        [text]
    )


    sequence = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post"
    )


    # Generate random latent vector

    if noise is None:

        noise = np.random.normal(
            0,
            1,
            (
                1,
                NOISE_DIM
            )
        ).astype(
            np.float32
        )


    # Generate image

    generated = generator.predict(
        [
            noise,
            sequence
        ],
        verbose=0
    )


    # Convert [-1, 1]
    # back to [0, 1]

    generated = (
        generated[0] + 1
    ) / 2.0


    generated = np.clip(
        generated,
        0,
        1
    )


    return generated


# ------------------------------------------------------------
# 21. GENERATE IMAGE FOR "A PHOTO OF A TIGER"
# ------------------------------------------------------------

text_description = (
    "a photo of a tiger"
)


generated_image = generate_image_from_text(
    text_description
)


plt.figure(
    figsize=(6, 6)
)


plt.imshow(
    generated_image
)


plt.axis(
    "off"
)


plt.title(
    "Text: " + text_description
)


plt.show()


# ------------------------------------------------------------
# 22. GENERATE IMAGES FOR ALL SELECTED CLASSES
# ------------------------------------------------------------

print(
    "\nGenerated Images:"
)


for class_name in CLASS_NAMES:

    text_description = (
        "a photo of a "
        + class_name
    )


    generated_image = generate_image_from_text(
        text_description
    )


    plt.figure(
        figsize=(5, 5)
    )


    plt.imshow(
        generated_image
    )


    plt.axis(
        "off"
    )


    plt.title(
        text_description
    )


    plt.show()


# ------------------------------------------------------------
# 23. COMPARE GENERATED IMAGE WITH IMAGENET EXAMPLE
# ------------------------------------------------------------

def display_comparison(
    class_name
):

    text_description = (
        "a photo of a "
        + class_name
    )


    # --------------------------------------------
    # Generate image
    # --------------------------------------------

    generated_image = generate_image_from_text(
        text_description
    )


    # --------------------------------------------
    # Find corresponding real image
    # --------------------------------------------

    real_index = None


    for i, label in enumerate(
        image_labels
    ):

        if label == class_name:

            real_index = i

            break


    if real_index is None:

        print(
            "No real image found for:",
            class_name
        )

        return


    real_image = load_img(
        image_paths[real_index],
        target_size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )


    # --------------------------------------------
    # Display comparison
    # --------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )


    # Real image

    plt.subplot(
        1,
        2,
        1
    )


    plt.imshow(
        real_image
    )


    plt.axis(
        "off"
    )


    plt.title(
        "ImageNet Example\n"
        + class_name
    )


    # Generated image

    plt.subplot(
        1,
        2,
        2
    )


    plt.imshow(
        generated_image
    )


    plt.axis(
        "off"
    )


    plt.title(
        "Generated Image\n"
        + text_description
    )


    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# 24. COMPARE SELECTED CLASSES
# ------------------------------------------------------------

for class_name in CLASS_NAMES:

    display_comparison(
        class_name
    )


# ------------------------------------------------------------
# 25. GENERATE IMAGES FOR PREVIOUSLY UNSEEN
#     TEXTUAL DESCRIPTIONS
# ------------------------------------------------------------

unseen_descriptions = [
    "a photo of a big tiger",
    "a photo of a wild elephant",
    "a photo of a fast zebra",
    "a photo of a large lion",
    "a photo of a small dog"
]


print(
    "\nPreviously unseen textual descriptions:"
)


for text_description in unseen_descriptions:

    generated_image = generate_image_from_text(
        text_description
    )


    plt.figure(
        figsize=(5, 5)
    )


    plt.imshow(
        generated_image
    )


    plt.axis(
        "off"
    )


    plt.title(
        "Unseen Text:\n"
        + text_description
    )


    plt.show()


# ------------------------------------------------------------
# 26. GENERATE MULTIPLE IMAGES FROM SAME TEXT
# ------------------------------------------------------------

text_description = (
    "a photo of a tiger"
)


plt.figure(
    figsize=(12, 3)
)


for i in range(4):

    generated_image = generate_image_from_text(
        text_description
    )


    plt.subplot(
        1,
        4,
        i + 1
    )


    plt.imshow(
        generated_image
    )


    plt.axis(
        "off"
    )


    plt.title(
        "Sample " + str(i + 1)
    )


plt.suptitle(
    'Text: "a photo of a tiger"'
)


plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 27. SAVE MODELS
# ------------------------------------------------------------

generator.save(
    "conditional_gan_generator.keras"
)


discriminator.save(
    "conditional_gan_discriminator.keras"
)


text_encoder.save(
    "text_encoder.keras"
)


print(
    "\nModels saved:"
)


print(
    "conditional_gan_generator.keras"
)


print(
    "conditional_gan_discriminator.keras"
)


print(
    "text_encoder.keras"
)


# ------------------------------------------------------------
# 28. FINAL EXPERIMENT SUMMARY
# ------------------------------------------------------------

print(
    "\n================================================"
)


print(
    "EXERCISE 5 COMPLETED"
)


print(
    "================================================"
)


print(
    "Text -> Text Encoder -> Text Embedding"
)


print(
    "Random Noise + Text Embedding"
)


print(
    "              ↓"
)


print(
    "Conditional Generator"
)


print(
    "              ↓"
)


print(
    "        Generated Image"
)


print(
    "              ↓"
)


print(
    "Conditional Discriminator"
)


print(
    "       ↑              ↑"
)


print(
    "   Real Image       Text"
)


print(
    "================================================"
)
