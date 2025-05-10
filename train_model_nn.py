import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib

# Load data from CSV
data = pd.read_csv("data/hand_gesture_data.csv")
X = data.iloc[:, :-1].values  # Landmarks (all rows except last column)
y = data.iloc[:, -1].values   # Labels (last column)

# Normalize the features
X = X / X.max(axis=0)  # Normalize to between 0 and 1

# Encode labels (convert gesture strings to integers)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build a simple neural network
model = models.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer for number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the model and label encoder
model.save("gesture_model_nn.h5")
joblib.dump(label_encoder, "label_encoder.pkl")

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
