# Detailed simple Neural Network conception in Python
# iDream3r

# Required libraries importation
import math
import random


# Class creation to manage the Neural Network
class NeuralNetwork():
    
    def __init__(self):

        # Setting the randomness seed to keep the same random values at each run
        random.seed(1)
        # Setting the random weight of the 3 ways with a random value between -1 and 1
        self.weights=[random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]
    
    # Neural Network's input and output calculation
    def think(self,neuron_inputs):
        
        # Calling the function the calculate the inputs sum (preactivation function)
        sum_of_weighted_inputs=self.__sum_of_weighted_inputs(neuron_inputs)
        # Calling the function to calculate the output (activation funtion)
        neuron_output=self.__sigmoid(sum_of_weighted_inputs)
        # Returning the calculated output
        return(neuron_output)
    
    # Adjusting the weights of the Neural Network to reduce the error
    # Error Cost Function
    def train(self, training_set_examples, number_of_iterations):
        for iteration in range(number_of_iterations):
            for training_set_example in training_set_examples:
                
                # Prediction of the output based on the training set example
                predicted_output = self.think(training_set_example["Input"])
                
                # Calculate the error between the predicted output and the training set output
                error_in_output = training_set_example["Output"] - predicted_output
                
                # Iteration and weights adjustment
                for index in range(len(self.weights)):
                    
                    # Association between the neuron input and its weight
                    neuron_input=training_set_example["Input"][index]
                    
                    # Calculate the value of the weight adjustment using the gradient descent
                    # It's proportional to the slope
                    adjust_weight = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)
                    
                    # Adjustement
                    self.weights[index] += adjust_weight

    # Sigmoid calculation (activation function)
    def __sigmoid(self, sum_of_weighted_inputs):
        return(1 /(1+math.exp(-sum_of_weighted_inputs))) 
    
    # Sigmoid gradient calculation
    def __sigmoid_gradient(self, neuron_output):       
        return(neuron_output * (1-neuron_output))
    
    # Sum (sigma) of the inputs : multiplication of each input by its own weight
    def __sum_of_weighted_inputs(self,neuron_inputs):
        sum_of_weighted_inputs=0
        for index, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[index] * neuron_input
        return sum_of_weighted_inputs
 
    
neural_network=NeuralNetwork()

print('Random starting weights :' +str(neural_network.weights))

# Training Set
# The Neural Network should associate a "1" in index 0 in the input list with a "1" for output 
training_set_examples=[{"Input":[0,0,1],"Output":0},
                       {"Input":[1,1,1],"Output":1},
                       {"Input":[1,0,1],"Output":1},
                       {"Input":[0,1,1],"Output":0}]

# Training the Neural Network with 10000 iterations
neural_network.train(training_set_examples, number_of_iterations=10000)

print('New weights after training :' +str(neural_network.weights))

# Making a prediction for a new set
new_situation=[1,0,0]
prediction=neural_network.think(new_situation)

print("Prediction for the new situation "+str(new_situation) + " : " +str(prediction))