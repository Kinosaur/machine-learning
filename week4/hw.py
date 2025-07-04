import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Activation functions
def sigmoid(z):
    """Sigmoid activation function with clipping to prevent overflow"""
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """Derivative of sigmoid function"""
    return z * (1 - z)


bmi_df = pd.read_csv("/Users/kaungkhantlin/Developer/1_2025/machine-learning/bmi.csv")
print("Loaded BMI data from CSV file")

print(f"Dataset shape: {bmi_df.shape}")
print("Dataset info:")
print(bmi_df.head())
print("\nDataset statistics:")
print(bmi_df.describe())

# Data preprocessing
print("\n=== Data Preprocessing ===")

# 1) Encode Gender: Female→0, Male→1
bmi_df["Gender"] = bmi_df["Gender"].map({"Female": 0, "Male": 1})

# 2) Extract input features X and target Y
X_raw = bmi_df[["Gender", "Height", "Weight"]].values
Y_raw = bmi_df[["Index"]].values

# 3) Do not split data; use all data for training
X_train_raw = X_raw
Y_train_raw = Y_raw
X_test_raw = X_raw
Y_test_raw = Y_raw

# 4) Normalize input features using z-score normalization
X_train_mean = X_train_raw.mean(axis=0)
X_train_std = X_train_raw.std(axis=0)
X_train = (X_train_raw - X_train_mean) / X_train_std
X_test = (X_test_raw - X_train_mean) / X_train_std  # Use training stats for test data

# 5) Normalize target to [0,1] for better training
Y_train_min, Y_train_max = Y_train_raw.min(), Y_train_raw.max()
Y_train = (Y_train_raw - Y_train_min) / (Y_train_max - Y_train_min)
Y_test = (Y_test_raw - Y_train_min) / (Y_train_max - Y_train_min)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Feature normalization - Mean: {X_train_mean}, Std: {X_train_std}")
print(f"Target normalization - Min: {Y_train_min}, Max: {Y_train_max}")

# Neural Network Implementation
print("\n=== Neural Network Training ===")

# Set random seed for reproducibility
np.random.seed(42)

# Network architecture

# Network topology: 3-3-1
n_features = 3  # Gender, Height, Weight
n_hidden_units = 3  # Hidden layer units
n_outputs = 1

# Initialize weights and biases with Xavier initialization
weights_input_hidden = np.random.randn(n_features, n_hidden_units) * np.sqrt(
    2.0 / n_features
)
weights_hidden_output = np.random.randn(n_hidden_units, n_outputs) * np.sqrt(
    2.0 / n_hidden_units
)
bias_hidden = np.zeros((1, n_hidden_units))
bias_output = np.zeros((1, n_outputs))

# Training hyperparameters
learning_rate = 0.05
n_epochs = 50000
batch_size = 32
loss_history = []
val_loss_history = []

n_train_samples = X_train.shape[0]
n_batches = max(1, n_train_samples // batch_size)

print(f"Network architecture: {n_features} -> {n_hidden_units} -> {n_outputs}")
print(
    f"Training parameters: LR={learning_rate}, Epochs={n_epochs}, Batch size={batch_size}"
)

# Training loop
for epoch in range(n_epochs):
    epoch_loss = 0.0

    # Shuffle training data
    indices = np.random.permutation(n_train_samples)
    X_train_shuffled = X_train[indices]
    Y_train_shuffled = Y_train[indices]

    # Mini-batch training
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_train_samples)

        X_batch = X_train_shuffled[start_idx:end_idx]
        Y_batch = Y_train_shuffled[start_idx:end_idx]

        # Forward pass
        hidden_input = np.dot(X_batch, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)
        output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        y_pred = sigmoid(output_input)

        # Compute loss
        error = Y_batch - y_pred
        batch_loss = np.mean(error**2)
        epoch_loss += batch_loss

        # Backward pass
        delta_output = error * sigmoid_derivative(y_pred)
        delta_hidden = sigmoid_derivative(hidden_output) * np.dot(
            delta_output, weights_hidden_output.T
        )

        # Update weights and biases
        weights_hidden_output += learning_rate * np.dot(hidden_output.T, delta_output)
        bias_output += learning_rate * np.mean(delta_output, axis=0, keepdims=True)
        weights_input_hidden += learning_rate * np.dot(X_batch.T, delta_hidden)
        bias_hidden += learning_rate * np.mean(delta_hidden, axis=0, keepdims=True)

    # Calculate validation loss
    hidden_val = sigmoid(np.dot(X_test, weights_input_hidden) + bias_hidden)
    y_val_pred = sigmoid(np.dot(hidden_val, weights_hidden_output) + bias_output)
    val_loss = np.mean((Y_test - y_val_pred) ** 2)

    loss_history.append(epoch_loss / n_batches)
    val_loss_history.append(val_loss)

    # Print progress
    if (epoch + 1) % 5000 == 0:
        print(
            f"Epoch {epoch+1}/{n_epochs} - Train Loss: {epoch_loss/n_batches:.6f}, Val Loss: {val_loss:.6f}"
        )


# Custom evaluation metrics
def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """Calculate R² coefficient of determination"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


# Evaluation
print("\n=== Model Evaluation ===")

# Make predictions on test set
hidden_test = sigmoid(np.dot(X_test, weights_input_hidden) + bias_hidden)
y_test_pred_norm = sigmoid(np.dot(hidden_test, weights_hidden_output) + bias_output)

# Denormalize predictions
y_test_pred = y_test_pred_norm * (Y_train_max - Y_train_min) + Y_train_min
y_test_actual = Y_test_raw

# Calculate metrics
mse = mean_squared_error(y_test_actual, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_test_pred)
mae = mean_absolute_error(y_test_actual, y_test_pred)

print("Test Results:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Show sample predictions
print("\nSample Predictions (first 10 test samples):")
for i in range(min(10, len(y_test_actual))):
    true_val = y_test_actual[i][0]
    pred_val = y_test_pred[i][0]
    print(
        f"True: {true_val:.2f}  →  Predicted: {pred_val:.2f}  (Error: {abs(true_val - pred_val):.2f})"
    )

# Final model summary
print("\n=== Model Summary ===")
print(
    f"Architecture: {n_features} inputs -> {n_hidden_units} hidden -> {n_outputs} output"
)
print(
    f"Total parameters: {weights_input_hidden.size + weights_hidden_output.size + bias_hidden.size + bias_output.size}"
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Final test RMSE: {rmse:.4f}")
print(f"Final test R²: {r2:.4f}")


# Function to make predictions on new data
def predict_bmi_index(gender, height, weight):
    """Make BMI index prediction for new data"""
    # Encode gender
    gender_encoded = 1 if gender.lower() == "male" else 0

    # Create input array
    x_new = np.array([[gender_encoded, height, weight]])

    # Normalize using training statistics
    x_new_norm = (x_new - X_train_mean) / X_train_std

    # Forward pass
    hidden = sigmoid(np.dot(x_new_norm, weights_input_hidden) + bias_hidden)
    pred_norm = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)

    # Denormalize
    prediction = pred_norm * (Y_train_max - Y_train_min) + Y_train_min

    return prediction[0][0]


# Example prediction
print("\n=== Example Prediction ===")
example_pred = predict_bmi_index("Male", 175, 70)
print(f"Example: Male, 175cm, 70kg -> Predicted BMI Index: {example_pred:.2f}")

bmi_categories = {0: "Underweight", 1: "Normal", 2: "Overweight", 3: "Obese"}
predicted_category = bmi_categories.get(round(example_pred), "Unknown")
print(f"Predicted category: {predicted_category}")


# # Visualizations
# print("\n=== Generating Visualizations ===")

# # Plot 1: Training history
# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.plot(loss_history, label="Training Loss", alpha=0.7)
# plt.plot(val_loss_history, label="Validation Loss", alpha=0.7)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training History")
# plt.legend()
# plt.grid(True, alpha=0.3)

# # Plot 2: Predictions vs Actual
# plt.subplot(1, 3, 2)
# plt.scatter(y_test_actual, y_test_pred, alpha=0.6, color="blue")
# plt.plot(
#     [y_test_actual.min(), y_test_actual.max()],
#     [y_test_actual.min(), y_test_actual.max()],
#     "r--",
#     lw=2,
# )
# plt.xlabel("Actual BMI Index")
# plt.ylabel("Predicted BMI Index")
# plt.title(f"Predictions vs Actual (R² = {r2:.3f})")
# plt.grid(True, alpha=0.3)

# # Plot 3: Residuals
# plt.subplot(1, 3, 3)
# residuals = y_test_actual.flatten() - y_test_pred.flatten()
# plt.scatter(y_test_pred, residuals, alpha=0.6, color="green")
# plt.axhline(y=0, color="r", linestyle="--", lw=2)
# plt.xlabel("Predicted BMI Index")
# plt.ylabel("Residuals")
# plt.title("Residual Plot")
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # Feature importance analysis (simple approximation)
# print("\n=== Feature Importance Analysis ===")
# feature_names = ["Gender", "Height", "Weight"]
# feature_importance = np.abs(weights_input_hidden).mean(axis=1)
# feature_importance = feature_importance / feature_importance.sum()

# print("Approximate feature importance:")
# for name, importance in zip(feature_names, feature_importance):
#     print(f"{name}: {importance:.3f}")
