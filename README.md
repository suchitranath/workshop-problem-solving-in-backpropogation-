# workshop-problem-solving-in-backpropogation-

# NAME : SUCHITRA NATH
# REG NO : 212223220112
# WORKSHOP
# DATE: 10/10/24
# PROBLEM SOLVING IN BACGPROPOGATION
# AIM:
 To implement a Multi-Layer Perceptron (MLP) with one hidden layer and train it using backpropagation for 2 iterations. The goal is to minimize the error between the predicted and target outputs by adjusting the weights and biases in the network, using the given network structure, inputs, and target output.
 # PROCEDURE:
 # 1. Initialize Inputs and targets:
 The input vector is given as 
 x = [1,1,0,1]

 The target output is t = 1
 
 # 2. Network Architecture:
 
The MLP consists of 4 input neurons, 2 hidden neurons, and 1 output neuron.

The initial weights and biases are provided for connections from input to hidden layer and hidden layer to output.

# 3. Define Activation Function:

Use the sigmoid activation function for the hidden and output layers:

\sigma(x) = \frac{1}{1 + e^{-x}}

\sigma'(x) = x \cdot (1 - x)

# 4. Forward Pass:

Calculate the output of each neuron using the weighted sum of inputs.

Pass the weighted sum through the sigmoid function to get the neuron’s output.

# 5. Error Calculation:

Calculate the error between the predicted output and the target using Mean Squared Error (MSE):

E = \frac{1}{2}(t - y)^2

# 6. Backpropogation:

Calculate the error gradients at the output neuron and propagate them backward to the hidden neurons.

Use the gradients to update the weights between input and hidden layers, and hidden and output layers.

# 7. Weight update rules:


The weights are updated using the following formula:

\Delta w = -\eta \cdot \delta \cdot x

# Repeat for 2 Iteration:

Repeat the forward pass, error calculation, and weight updates for 2 iterations.

# PROGRAM:

```
import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Given inputs and weights
inputs = np.array([1, 1, 0, 1])  # x0, x1, x2, x3
target = 1  # Target output
learning_rate = 0.5  # Learning rate



# Initial weights (same as in the diagram)
weights_input_to_hidden = np.array([[0.1, 0.3],  # Weights from x0 to x5, x6
                                    [0.3, 0.2],  # Weights from x1 to x5, x6
                                    [-0.2, 0.1],  # Weights from x2 to x5, x6
                                    [0.4, -0.3]], dtype=np.float64)  # Ensure float64 type

weights_hidden_to_output = np.array([-0.3, 0.4], dtype=np.float64)  # Weights from x5, x6 to x7

# Biases
bias_hidden = np.array([0.0, 0.0], dtype=np.float64)  # Biases for x5 and x6 as float
bias_output = np.float64(0.0)  # Bias for x7 as float

# Perform forward pass
def forward_pass(inputs, weights_input_to_hidden, weights_hidden_to_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(inputs, weights_input_to_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output
    output = sigmoid(output_layer_input)
    
    return hidden_layer_output, output

# Perform backpropagation and update weights
def backpropagate(inputs, hidden_layer_output, output, target, weights_hidden_to_output, weights_input_to_hidden, bias_hidden, bias_output, learning_rate):
    # Output layer error
    error_output = output - target
    delta_output = error_output * sigmoid_derivative(output)
    
    # Hidden layer error
    error_hidden = delta_output * weights_hidden_to_output
    delta_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    weights_hidden_to_output -= learning_rate * delta_output * hidden_layer_output
    bias_output -= learning_rate * delta_output
    
    weights_input_to_hidden -= learning_rate * np.outer(inputs, delta_hidden)
    bias_hidden -= learning_rate * delta_hidden
    
    return weights_input_to_hidden, weights_hidden_to_output, bias_hidden, bias_output

# Training for 2 iterations
for i in range(2):
    print(f"Iteration {i+1}:")
    
    # Forward pass
    hidden_layer_output, output = forward_pass(inputs, weights_input_to_hidden, weights_hidden_to_output, bias_hidden, bias_output)
    print(f"Output: {output}")
    
    # Calculate error
    error = 0.5 * (target - output) ** 2
    print(f"Error: {error}")
    
    # Backpropagation and weight update
    weights_input_to_hidden, weights_hidden_to_output, bias_hidden, bias_output = backpropagate(
        inputs, hidden_layer_output, output, target, weights_hidden_to_output, weights_input_to_hidden, bias_hidden, bias_output, learning_rate
    )

    print(f"Weights (input to hidden): \n{weights_input_to_hidden}")
    print(f"Weights (hidden to output): {weights_hidden_to_output}")
    print(f"Biases (hidden): {bias_hidden}")
    print(f"Bias (output): {bias_output}")
    print("-" * 50)

```

# OUTPUT:

![image](https://github.com/user-attachments/assets/7a0ceb7c-fbfe-48a5-9e93-ab7fb0e2d1d1)


# RESULT:

# Iteration 1:

Forward Pass:

Hidden neuron outputs: x5 ~ 0.69, x6 ~ 0.55 

Output neuron: x7 ~ 0.503

Error:

E = \frac{1}{2}(1 - 0.503)^2 \approx 0.123

Backpropagation:

Weight updates for input-to-hidden and hidden-to-output layers are calculated using the gradients and learning rate.

# Iteration 2:

Forward Pass:

After updating the weights from iteration 1, the output values are recalculated.

New hidden neuron outputs: Updated values based on adjusted weights.

New output neuron x7: Updated value (closer to target).


Error:

The error after the second iteration is calculated again and should be smaller than the error from the first iteration.


