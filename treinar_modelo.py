import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import os

train_dir = "dataset/treino"
val_dir = "dataset/validacao"

classes = sorted(os.listdir(train_dir))
print("Classes detectadas:", classes)

datagen_treino = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

datagen_val = ImageDataGenerator(
    rescale=1./255
)
treino = datagen_treino.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical'
)

validacao = datagen_val.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical'
)

base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    treino,
    epochs=50,
    validation_data=validacao
)

model.save("modelo/modelo_mobilenet.h5")

print("Modelo salvo em modelo/modelo_mobilenet.h5")