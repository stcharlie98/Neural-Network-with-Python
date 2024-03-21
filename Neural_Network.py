# Simulation of 1 neuron in a Neural Network to understand "deeper" the basics of Deep Learning.

import math # Declaring initial variables for the simulation.

input = -1 # Setting the initial input.
output_desired = 1 # Desired output for learning.

input_weight = 0.5 # Initial weight assigned to the input.
learning_rate = 0.1 # Learning rate for adjusting weights.

# Activation function to determine neuron output.
def activation(sum): 
    if sum >= 0:
        return 1
    else:
        return 0

print("Input: ", input, "\nOutput desired: ", output_desired)

error = math.inf  # Any number multiplied by zero is zero, this short block of code doesn't return 1 (line 14) for this case, but it returns 0 instead.
virtual_neuron = 1 # Virtual neuron that always has as result 1 for learning purposes.
virtual_neuron_weight = 0.5 # Initial weight assigned to the virtual neuron.

counter_iterations = 0 # Counter to track the number of iterations.

# Loop to train the neuron until the error is 0.
while not error == 0:
    counter_iterations += 1 # Incrementing the iteration counter.
    print("\nIteration ", counter_iterations, ":", sep='') # Showing the number of iterations.

    # Calculating the sum of inputs multiplied by their respective weights.
    sum = (input * input_weight) + (virtual_neuron * virtual_neuron_weight)

    output = activation(sum) # Applying activation function to get the neuron output.

    print("Output: ", output) # Displaying the neuron's output.

    error = output_desired - output # Calculating the error.

    print("Error: ", error) # Displaying the current error.

    # Updating weights using the error and learning rate.
    if not error == 0:
        input_weight = input_weight + (learning_rate * input * error)
        virtual_neuron_weight = virtual_neuron_weight + (learning_rate * virtual_neuron * error) # Adjusting weights to prevent divergence.

print("\nThe neural network has learned successfully!!!") # Message of success!
