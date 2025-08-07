# %% [markdown]
# # BMI Prediction with Custom MLP (3-5-1 Topology)
#
# This notebook demonstrates predicting Body Mass Index (BMI) from Gender, Height, and Weight using a manually implemented Multi-Layer Perceptron (MLP) with a 3-5-1 architecture (3 inputs, 5 hidden neurons, 1 output neuron).
#
# - **Dataset:** First 100 rows of `bmi.csv`
# - **Preprocessing:** Gender encoded as 0.1 (Male) and 0.2 (Female); Height, Weight, BMI divided by 100
# - **MLP:** No external libraries, all learning steps coded manually
# - **Activation:** Sigmoid for all neurons
#
# The workflow follows the learning steps outlined in the lecture slides.

# %%
import math
import random
import pandas as pd

# %%
# Load first 500 rows
bmi_df = pd.read_csv("../bmi.csv").iloc[:100].copy()

# Encode Gender: Male=0.1, Female=0.2
bmi_df["Gender"] = bmi_df["Gender"].apply(
    lambda x: 0.2 if str(x).lower() == "female" else 0.1
)

# Convert Height, Weight, BMI to floating-point (divide by 100)
bmi_df["Height"] = bmi_df["Height"] / 100.0
bmi_df["Weight"] = bmi_df["Weight"] / 100.0
bmi_df["Index"] = bmi_df["Index"] / 10

# Prepare training data
X = bmi_df[["Gender", "Height", "Weight"]].values.tolist()
y = bmi_df["Index"].values.reshape(-1, 1).tolist()

print("Sample processed input:", X[0], "Target:", y[0])

# %% [markdown]
# ## MLP Architecture and Learning Steps
# - **Inputs:** Gender, Height, Weight (all float)
# - **Hidden Layer:** 5 neurons, sigmoid activation
# - **Output Layer:** 1 neuron, sigmoid activation
# - **Forward Pass:** Compute activations layer by layer
# - **Backward Pass:** Compute gradients and update weights using gradient descent
# - **Loss:** Mean squared error (MSE)
# - **Training:** Loop over data for multiple epochs


# %%
def sigmoid(x):
    x = max(min(x, 500), -500)
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)


def initialize_network():
    # 3 inputs, 5 hidden, 1 output
    w1 = [[random.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(5)]
    w2 = [[random.uniform(-0.5, 0.5) for _ in range(1)] for _ in range(5)]
    b_hidden = [random.uniform(-0.5, 0.5) for _ in range(5)]
    b_output = [random.uniform(-0.5, 0.5) for _ in range(1)]
    network = {"w1": w1, "w2": w2, "b_hidden": b_hidden, "b_output": b_output}
    return network


# %%
def forward_pass(network, inputs):
    w1, w2 = network["w1"], network["w2"]
    b_hidden, b_output = network["b_hidden"], network["b_output"]

    # Hidden layer activations
    hidden_activations = []
    for i in range(len(w1)):
        z_hidden = sum(inputs[j] * w1[i][j] for j in range(len(inputs))) + b_hidden[i]
        hidden_activations.append(sigmoid(z_hidden))

    # Output layer activation
    z_output = (
        sum(hidden_activations[j] * w2[j][0] for j in range(len(hidden_activations)))
        + b_output[0]
    )
    output_activation = sigmoid(z_output)
    return hidden_activations, output_activation


# %%
def backward_pass(network, hidden_activations, output_activation, target):
    w2 = network["w2"]
    output_error = target[0] - output_activation
    output_delta = output_error * sigmoid_derivative(output_activation)

    hidden_deltas = []
    for j in range(len(hidden_activations)):
        error = output_delta * w2[j][0]
        delta = error * sigmoid_derivative(hidden_activations[j])
        hidden_deltas.append(delta)

    return hidden_deltas, output_delta


def update_weights(
    network, inputs, hidden_activations, hidden_deltas, output_delta, learning_rate
):
    # Update output weights and bias
    for j in range(len(network["w2"])):
        network["w2"][j][0] += learning_rate * output_delta * hidden_activations[j]
    network["b_output"][0] += learning_rate * output_delta

    # Update hidden weights and biases
    for i in range(len(network["w1"])):
        for j in range(len(inputs)):
            network["w1"][i][j] += learning_rate * hidden_deltas[i] * inputs[j]
        network["b_hidden"][i] += learning_rate * hidden_deltas[i]


# %%
def train_network(network, X, y, epochs, learning_rate):
    for epoch in range(epochs):
        sum_error = 0
        for inputs, target in zip(X, y):
            hidden_activations, output_activation = forward_pass(network, inputs)
            sum_error += (target[0] - output_activation) ** 2
            hidden_deltas, output_delta = backward_pass(
                network, hidden_activations, output_activation, target
            )
            update_weights(
                network,
                inputs,
                hidden_activations,
                hidden_deltas,
                output_delta,
                learning_rate,
            )
        if epoch % 10000 == 0 or epoch == epochs - 1:
            print(
                f"Epoch={epoch}, Learning Rate={learning_rate:.3f}, Error={sum_error:.6f}"
            )


# %%
# Initialize network
network = initialize_network()
learning_rate = 0.01
epochs = 100000

print("--- Training MLP ---")
train_network(network, X, y, epochs, learning_rate)
print("--- Training Complete ---")

# %%
# Example prediction
test_gender = 0.2  # Female
test_height = 1.85
test_weight = 1.1
test_input = [test_gender, test_height, test_weight]
hidden, pred = forward_pass(network, test_input)
print(f"Predicted BMI (scaled): {pred:.4f}")
