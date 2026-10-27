# %%
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras import layers, Model

PATH = "imagenet-mini"   # folder with deer, dog, frog, monkey, rabbit
SIZE = 32
NOISE = 100
EMBED = 32
EPOCHS = 300             # tiny dataset needs more epochs
BATCH = 8                # tiny dataset needs a smaller batch

images, texts = [], []

folders = sorted(
    f for f in os.listdir(PATH)
    if os.path.isdir(os.path.join(PATH, f))
)[:5]

print("Classes found:", folders)

for folder in folders:
    folder_path = os.path.join(PATH, folder)
    text = "a photo of a " + folder.replace("_", " ").lower()

    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                img = Image.open(os.path.join(folder_path, file))
                img = img.convert("RGB").resize((SIZE, SIZE))
                images.append(np.array(img) / 127.5 - 1)
                texts.append(text)
            except Exception:
                pass

images = np.array(images, dtype="float32")
texts = np.array(texts)

print(images.shape, texts.shape)
print("Example text:", texts[0])

# %%
vectorizer = layers.TextVectorization(
    max_tokens=1000,
    output_sequence_length=10
)

vectorizer.adapt(texts)
tokens = vectorizer(texts)

vocab = len(vectorizer.get_vocabulary())

# %%
t_in = layers.Input(shape=(10,), dtype="int32")
t = layers.Embedding(vocab, EMBED)(t_in)
t = layers.GlobalAveragePooling1D()(t)
t = layers.Dense(EMBED, activation="relu")(t)

encoder = Model(t_in, t)

# %%
z = layers.Input(shape=(NOISE,))
c = layers.Input(shape=(EMBED,))

x = layers.Concatenate()([z, c])
x = layers.Dense(8 * 8 * 128, activation="relu")(x)
x = layers.Reshape((8, 8, 128))(x)
x = layers.UpSampling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.UpSampling2D()(x)
out = layers.Conv2D(3, 3, padding="same", activation="tanh")(x)

G = Model([z, c], out)

# %%
img = layers.Input(shape=(SIZE, SIZE, 3))
c = layers.Input(shape=(EMBED,))

cond = layers.Dense(SIZE * SIZE)(c)
cond = layers.Reshape((SIZE, SIZE, 1))(cond)

x = layers.Concatenate()([img, cond])
x = layers.Conv2D(64, 4, strides=2, padding="same")(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Flatten()(x)
out = layers.Dense(1)(x)

D = Model([img, c], out)

# %%
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

g_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
d_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
t_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

data = tf.data.Dataset.from_tensor_slices(
    (images, tokens)
).shuffle(len(images)).batch(BATCH)

# %%

@tf.function
def train(real, words):
    n = tf.shape(real)[0]

    with tf.GradientTape(persistent=True) as tape:
        text = encoder(words, training=True)
        noise = tf.random.normal((n, NOISE))

        fake = G([noise, text], training=True)

        real_score = D([real, text], training=True)
        fake_score = D([fake, text], training=True)

        d_loss = (
            loss(tf.ones_like(real_score), real_score)
            + loss(tf.zeros_like(fake_score), fake_score)
        )

        g_loss = loss(tf.ones_like(fake_score), fake_score)
        t_loss = g_loss + d_loss

    gg = tape.gradient(g_loss, G.trainable_variables)
    dg = tape.gradient(d_loss, D.trainable_variables)
    tg = tape.gradient(t_loss, encoder.trainable_variables)

    del tape

    g_opt.apply_gradients([
        (grad, var)
        for grad, var in zip(gg, G.trainable_variables)
        if grad is not None
    ])

    d_opt.apply_gradients([
        (grad, var)
        for grad, var in zip(dg, D.trainable_variables)
        if grad is not None
    ])

    t_opt.apply_gradients([
        (grad, var)
        for grad, var in zip(tg, encoder.trainable_variables)
        if grad is not None
    ])

    return g_loss, d_loss


# %%
def generate(text):
    words = vectorizer([text])
    condition = encoder(words, training=False)

    noise = tf.random.normal((1, NOISE))
    image = G([noise, condition], training=False)[0]

    image = (image.numpy() + 1) / 2
    plt.imshow(np.clip(image, 0, 1))
    plt.title(text)
    plt.axis("off")
    plt.show()

generate("a photo of a tiger")


