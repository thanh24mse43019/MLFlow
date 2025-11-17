import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist10_mobilenet.h5")
# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Expand grayscale → RGB
x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, -1))
x_test = tf.image.grayscale_to_rgb(tf.expand_dims(x_test, -1))

# Resize MNIST (28×28 → 96×96 required by MobileNetV2)
x_train = tf.image.resize(x_train, (96, 96))
x_test = tf.image.resize(x_test, (96, 96))

# Preprocess for MobileNet
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# Build MobileNetV2 base (no top layers)
base_model = MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights='imagenet'  # use pretrained weights
)

base_model.trainable = False  # freeze backbone

# Classification head for MNIST-10
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=64
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")

# --- Save model ---
model.save(MODEL_PATH)
print("✅ Model saved as mnist10_mobilenet.h5")

