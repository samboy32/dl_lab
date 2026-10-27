
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


# ============================================================
# 1. LOAD IMAGE
# ============================================================

IMAGE_PATH = "your_image.jpg"       # <-- Change to your image path

img = image.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)

# Add batch dimension
input_image = np.expand_dims(img_array, axis=0)

# VGG16 preprocessing
input_image = preprocess_input(input_image)


# ============================================================
# 2. LOAD PRE-TRAINED VGG16
# ============================================================

model = VGG16(
    weights="imagenet",
    include_top=True
)

model.summary()


# ============================================================
# 3. TOP-5 IMAGENET PREDICTIONS
# ============================================================

predictions = model.predict(input_image)

top_5 = decode_predictions(
    predictions,
    top=5
)[0]

print("\n")
print("=" * 60)
print("TOP-5 IMAGENET PREDICTIONS")
print("=" * 60)

for i, (class_id, class_name, probability) in enumerate(
    top_5,
    start=1
):

    print(
        f"{i}. {class_name:30s} "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# 4. DISPLAY INPUT IMAGE
# ============================================================

plt.figure(figsize=(5, 5))

plt.imshow(img)
plt.title("Input Image")
plt.axis("off")

plt.show()


# ============================================================
# 5. SELECT CONVOLUTIONAL LAYERS
# ============================================================
#
# Early       -> block1_conv2
# Intermediate-> block3_conv3
# Deep        -> block5_conv3
#
# ============================================================

selected_layers = [
    "block1_conv2",
    "block3_conv3",
    "block5_conv3"
]


# ============================================================
# 6. EXTRACT FEATURE MAPS
# ============================================================

layer_outputs = [
    model.get_layer(layer_name).output
    for layer_name in selected_layers
]

feature_model = Model(
    inputs=model.input,
    outputs=layer_outputs
)

feature_maps = feature_model.predict(input_image)


# ============================================================
# 7. VISUALIZE FEATURE MAPS
# ============================================================

for layer_name, feature_map in zip(
    selected_layers,
    feature_maps
):

    print("\n")
    print("=" * 60)
    print("Layer:", layer_name)
    print("Feature Map Shape:", feature_map.shape)

    # Shape = (batch, height, width, channels)

    height = feature_map.shape[1]
    width = feature_map.shape[2]
    channels = feature_map.shape[3]

    print(
        "Spatial Dimensions:",
        height,
        "x",
        width
    )

    print(
        "Number of Channels:",
        channels
    )

    # --------------------------------------------------------
    # Display first 16 feature maps
    # --------------------------------------------------------

    n_features = min(16, channels)

    cols = 4
    rows = int(np.ceil(n_features / cols))

    plt.figure(
        figsize=(12, 3 * rows)
    )

    for i in range(n_features):

        # Extract one feature map
        feature = feature_map[0, :, :, i]

        # Normalize feature map
        feature = feature - feature.min()

        if feature.max() != 0:
            feature = feature / feature.max()

        plt.subplot(
            rows,
            cols,
            i + 1
        )

        plt.imshow(
            feature,
            cmap="viridis"
        )

        plt.title(
            f"Channel {i + 1}"
        )

        plt.axis("off")

    plt.suptitle(
        f"{layer_name} - Feature Maps\n"
        f"Dimensions: "
        f"{height} × {width} × {channels}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 8. VISUALIZE LEARNED CONVOLUTIONAL FILTERS
# ============================================================

filter_layers = [
    "block1_conv1",
    "block3_conv1",
    "block5_conv1"
]


for layer_name in filter_layers:

    layer = model.get_layer(layer_name)

    # Get learned weights and biases
    filters, biases = layer.get_weights()

    print("\n")
    print("=" * 60)
    print("Filter Layer:", layer_name)
    print("Filter Shape:", filters.shape)

    # --------------------------------------------------------
    # Filter shape:
    #
    # (filter_height,
    #  filter_width,
    #  input_channels,
    #  output_channels)
    # --------------------------------------------------------

    n_filters = min(
        16,
        filters.shape[-1]
    )

    cols = 4
    rows = int(
        np.ceil(
            n_filters / cols
        )
    )

    plt.figure(
        figsize=(12, 3 * rows)
    )

    for i in range(n_filters):

        # Extract one filter
        filter_img = filters[:, :, :, i]

        # Normalize
        filter_img = (
            filter_img -
            filter_img.min()
        )

        if filter_img.max() != 0:

            filter_img = (
                filter_img /
                filter_img.max()
            )

        plt.subplot(
            rows,
            cols,
            i + 1
        )

        # ----------------------------------------------------
        # First convolutional layer:
        # Input has 3 RGB channels
        # ----------------------------------------------------

        if filter_img.shape[-1] == 3:

            plt.imshow(
                filter_img
            )

        # ----------------------------------------------------
        # Deeper convolutional layers:
        # Input has many feature channels.
        # Take mean across channels for visualization.
        # ----------------------------------------------------

        else:

            plt.imshow(
                np.mean(
                    filter_img,
                    axis=-1
                ),
                cmap="viridis"
            )

        plt.title(
            f"Filter {i + 1}"
        )

        plt.axis("off")

    plt.suptitle(
        f"Learned Convolutional Filters\n"
        f"{layer_name}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 9. PRINT FEATURE-MAP DIMENSIONS
# ============================================================

print("\n\n")
print("=" * 60)
print("FEATURE MAP DIMENSIONS")
print("=" * 60)

for layer_name, feature_map in zip(
    selected_layers,
    feature_maps
):

    print(
        f"{layer_name:20s} : "
        f"{feature_map.shape[1]} × "
        f"{feature_map.shape[2]} × "
        f"{feature_map.shape[3]}"
    )


# ============================================================
# 10. DEPTH-WISE COMPARISON
# ============================================================

print("\n\n")
print("=" * 60)
print("DEPTH-WISE COMPARISON")
print("=" * 60)

for layer_name, feature_map in zip(
    selected_layers,
    feature_maps
):

    height = feature_map.shape[1]
    width = feature_map.shape[2]
    channels = feature_map.shape[3]

    print(
        f"{layer_name:20s} -> "
        f"Spatial Size: "
        f"{height} × {width}, "
        f"Channels: {channels}"
    )


# ============================================================
# 11. COMBINED DEPTH-WISE FEATURE MAP VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 12)
)

for ax, layer_name, feature_map in zip(
    axes,
    selected_layers,
    feature_maps
):

    # --------------------------------------------------------
    # Average all channels to create one representative
    # activation map for this layer
    # --------------------------------------------------------

    activation = np.mean(
        feature_map[0],
        axis=-1
    )

    # Normalize
    activation = (
        activation -
        activation.min()
    )

    if activation.max() != 0:

        activation = (
            activation /
            activation.max()
        )

    ax.imshow(
        activation,
        cmap="viridis"
    )

    ax.set_title(
        f"{layer_name} | "
        f"{feature_map.shape[1]} × "
        f"{feature_map.shape[2]} × "
        f"{feature_map.shape[3]}",
        fontsize=13
    )

    ax.axis("off")


plt.suptitle(
    "Depth-wise Comparison of Feature Maps",
    fontsize=16
)

plt.tight_layout()
plt.show()


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n\n")
print("=" * 60)
print("EXERCISE 01 SUMMARY")
print("=" * 60)

print("\nSelected Layers:")
print("----------------")

for layer_name in selected_layers:
    print(layer_name)

print("\nFeature Map Dimensions:")
print("-----------------------")

for layer_name, feature_map in zip(
    selected_layers,
    feature_maps
):

    print(
        f"{layer_name} : "
        f"{feature_map.shape[1]} × "
        f"{feature_map.shape[2]} × "
        f"{feature_map.shape[3]}"
    )

print("\nInterpretation:")
print("--------------")

print(
    "Early layers capture low-level features "
    "such as edges, colors and simple textures."
)

print(
    "Intermediate layers capture more complex "
    "patterns, shapes and object parts."
)

print(
    "Deep layers capture high-level and "
    "more abstract semantic representations."
)

print(
    "\nAs network depth increases, spatial "
    "dimensions decrease while the number "
    "of feature channels increases."
)




import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


IMAGE_PATH = "your_image.jpg"

img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
input_image = preprocess_input(np.expand_dims(img_array, axis=0))


model = VGG16(weights="imagenet", include_top=True)
predictions = model.predict(input_image)
top_5 = decode_predictions(predictions, top=5)[0]

plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.show()


selected_layers = ["block1_conv2", "block3_conv3", "block5_conv3"]
layer_outputs = [model.get_layer(layer_name).output for layer_name in selected_layers]
feature_model = Model(inputs=model.input, outputs=layer_outputs)
feature_maps = feature_model.predict(input_image)


for layer_name, feature_map in zip(selected_layers, feature_maps):
    height, width, channels = feature_map.shape[1:]
    n_features = min(16, channels)
    cols = 4
    rows = int(np.ceil(n_features / cols))

    plt.figure(figsize=(12, 3 * rows))
    for i in range(n_features):
        feature = feature_map[0, :, :, i]
        feature = feature - feature.min()
        if feature.max() != 0:
            feature = feature / feature.max()

        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature, cmap="viridis")
        plt.title(f"Channel {i + 1}")
        plt.axis("off")

    plt.suptitle(
        f"{layer_name} Feature Maps\nDimensions: {height} × {width} × {channels}",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()


filter_layers = ["block1_conv1", "block3_conv1", "block5_conv1"]

for layer_name in filter_layers:
    filters, biases = model.get_layer(layer_name).get_weights()
    n_filters = min(16, filters.shape[-1])
    cols = 4
    rows = int(np.ceil(n_filters / cols))

    plt.figure(figsize=(12, 3 * rows))
    for i in range(n_filters):
        filter_img = filters[:, :, :, i]
        filter_img = filter_img - filter_img.min()
        if filter_img.max() != 0:
            filter_img = filter_img / filter_img.max()

        plt.subplot(rows, cols, i + 1)
        if filter_img.shape[-1] == 3:
            plt.imshow(filter_img)
        else:
            plt.imshow(np.mean(filter_img, axis=-1), cmap="viridis")
        plt.title(f"Filter {i + 1}")
        plt.axis("off")

    plt.suptitle(f"Learned Convolutional Filters\n{layer_name}", fontsize=14)
    plt.tight_layout()
    plt.show()


fig, axes = plt.subplots(3, 1, figsize=(12, 12))

for ax, layer_name, feature_map in zip(axes, selected_layers, feature_maps):
    activation = np.mean(feature_map[0], axis=-1)
    activation = activation - activation.min()
    if activation.max() != 0:
        activation = activation / activation.max()

    ax.imshow(activation, cmap="viridis")
    ax.set_title(
        f"{layer_name} | {feature_map.shape[1]} × {feature_map.shape[2]} × {feature_map.shape[3]}",
        fontsize=13
    )
    ax.axis("off")

plt.suptitle("Depth-wise Comparison of Feature Maps", fontsize=16)
plt.tight_layout()
plt.show()