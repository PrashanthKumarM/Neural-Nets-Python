import sys
import math
import operator
import time
import random

train_set = []
test_set = []
correct = 0
labels = [0,90,180,270] # Problem Specific
confusion = {y: {x: 0 for x in labels} for y in labels}

def main(train, test, layer_size, learning_rate):
	now = time.time()
	print "Loading pixels...."
	train_set = load_file(train)
	test_set = load_file(test)
	nnet(train_set, test_set, layer_size, learning_rate)
	print "Time Elapsed: "+str(time.time()-now)

# Problem Specific Methods Begin

def confusion_accuracy(actual, predicted, label, output_file):
	global correct
	confusion[actual][predicted] += 1
	if actual == predicted:
		correct += 1
	output_file.write(label+" "+str(predicted)+"\n")
	print "Predicted: " + str(predicted)
	print "Actual : " + str(actual) + "\n\n"

def print_confusion():
	print "\t".join([str(x) for x in confusion.keys()])
	print "----"*8
	for x in confusion.values():
		print "\t".join([str(y) for y in x.values()])
	print"\n\n"

def load_file(file_name):
	given = open(file_name, 'r')
	return convert_pixels([x.split(' ') for x in given])

def convert_pixels(vector):
	trains = []
	pixels = []
	global method_used
	for vec in vector:
		trains.append([vec[0], int(vec[1])])
		pic = [float(x)/255.0 for x in vec[2:]]
		pixels.append(pic)
	return [trains, pixels]

# problem specific methods end

# Neural Net begins

# Base methods
def nnet(train, test, layer_size, learning_rate):
	learning_rate = float(learning_rate)
	global input_nodes, hidden_nodes, output_nodes, input_weights, output_weights, bias, activation_output, activation_hidden

	# Initializing data structures
	input_nodes = 192
	hidden_nodes = int(layer_size)
	output_nodes = len(labels)
	activation_hidden = [1.0]*hidden_nodes
	activation_output = [1.0]*output_nodes
	input_weights = [[random.random() for j in range(hidden_nodes)] for i in range(input_nodes)]
	output_weights = [[random.random() for j in range(output_nodes)] for i in range(hidden_nodes)]
	bias=1.0

	nnet_train(train, layer_size, learning_rate)
	predict(test)

def predict(test):  
	results = open("nnet_ouput.txt", 'w')
	for x,j in enumerate(test[1]):
		target_result = int(test[0][x][1])
		res = feed_forward(j)
		print "\n"
		confusion_accuracy(target_result, res, test[0][x][0], results)
	results.close()
	print_confusion()
	print "Accuracy: " + str((correct*100/len(test[1]))) + " %"

def nnet_train(train, layer_size, learning_rate):
	print "Training data....( If you find any of the output values staying at more than 0.999 till the end of training, there is a problem in the weight randomization. Please run it again. )\n"
	for x,j in enumerate(train[1]):
		targets=[0]*4
		targets[int(train[0][x][1])/90] = 1
		feed_forward(j)
		back_propagate(targets, learning_rate, j)
	print "\n"

# NNet Logical methods

def feed_forward(inputs): 
	global input_nodes, hidden_nodes, output_nodes, input_weights, output_weights, bias, activation_output, activation_hidden

	# Obtain activations (sigmoid(sum))
	activation_hidden = update_sum(hidden_nodes, input_nodes, input_weights, inputs, bias, activation_hidden)
	activation_hidden = normalize_sigmoid(max(activation_hidden,key=float), min(activation_hidden,key=float), hidden_nodes, activation_hidden) 
	activation_ouput = update_sum(output_nodes, hidden_nodes, output_weights, activation_hidden, bias, activation_output)
	activation_output = normalize_sigmoid(max(activation_hidden,key=float), min(activation_hidden,key=float), output_nodes, activation_output)

	sys.stdout.write(str(activation_output)+" "*5+"\r")
	sys.stdout.flush()
	max_out=max(activation_output,key=float)
	return activation_output.index(max_out)*90

def back_propagate(targets,learning_rate,inputs):
	global input_nodes, hidden_nodes, output_nodes, input_weights, output_weights, bias, activation_output, activation_hidden
	output_deltas = [(targets[k]-activation_output[k])*deactivate(activation_output[k]) for k in range(output_nodes)]

	#Updating output weights
	for j in range(hidden_nodes):
		for k in range(output_nodes):
			change = output_deltas[k] * activation_hidden[j]
			output_weights[j][k] += learning_rate*change

	hidden_deltas = [sum([output_deltas[k] * output_weights[j][k] for k in range(output_nodes)]) for j in range(hidden_nodes)]

	#Updating input weights
	for i in range (input_nodes):
		for j in range (hidden_nodes):
			change = hidden_deltas[j] * inputs[i]
			input_weights[i][j] += learning_rate*change

# Neural helpers

def update_sum(to_nodes, from_nodes, weights, inputs, bias, a_sum):
	for j in range(0,to_nodes):
		a_sum[j] = sum( [inputs[i] * weights[i][j] for i in range(0,from_nodes)])+bias
	return a_sum

def normalize_sigmoid(a_max, a_min, nodes, activations):
	for i in range(0,nodes):
		activations[i]=(activations[i]-a_min)/(a_max-a_min)
		activations[i]=activate(activations[i])
	return activations

def activate(x):
	return 1/(1+math.exp(-x))

def deactivate(y):
	return y*(1-y)

# Neural net ends

if __name__ == "__main__": main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
