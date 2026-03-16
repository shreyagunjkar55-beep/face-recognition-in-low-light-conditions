import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# loading data
X_train = np.load("X_train_final.npy")
X_test = np.load("X_test_final.npy")
y_train = np.load("y_train_final.npy")
y_test = np.load("y_test_final.npy")
classes = np.load("label_classes_final.npy")

num_classes = len(classes)

#hybrid model
input_layer = Input(shape=(160, 160, 3))

# 1. Pretrained MobileNetV2 branch
base_model = MobileNetV2(include_top=False, weights="imagenet", input_tensor=input_layer)
x1 = GlobalAveragePooling2D()(base_model.output)

# 2. Custom small CNN branch
x2 = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Conv2D(64, (3, 3), activation="relu", padding="same")(x2)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Flatten()(x2)

# Merge both feature sets
merged = Concatenate()([x1, x2])
merged = Dense(256, activation="relu")(merged)
merged = Dropout(0.5)(merged)
output = Dense(num_classes, activation="softmax")(merged)

# Final model
model = Model(inputs=input_layer, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# train model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=10,
                    batch_size=16)

#evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# save model
model.save("hybrid_face_model.h5")
print("Hybrid model saved as 'hybrid_face_model.h5'")

