# Neural Nets in Python to classify image data

# Neural Networks Classifier:

About Neural Networks (Reference Wikipedia):

In machine learning and cognitive science, artificial neural networks (ANNs) are a family of models inspired by biological neural networks (the central nervous systems of animals, in particular the brain) and are used to estimate or approximate functions that can depend on a large number of inputs and are generally unknown. Artificial neural networks are generally presented as systems of interconnected "neurons" which exchange messages between each other. The connections have numeric weights that can be tuned based on experience, making neural nets adaptive to inputs and capable of learning.

Aim of building the classifier: 

The aim is to create a classiﬁer that decides the correct orientation of a given image.

Settings for Neural Network along with results and screenshots of the output we got :

Input dataset               :   Train:  36976 images
                                Test:   943 images

Hidden Layers               : Tried different numbers of hidden layers.

Learning Rate               : Tried different values for learning rate.

Number of nodes in layers   : Tried different numbers of nodes in different layers of neural net.


Given below, results we obtained with different combinations of settings for Neural Nets :

				Learning Rate 0.1 and number of hidden layers 2 :
				Accuracy : 26 %

				Learning Rate 0.1 and number of hidden layers 3 :
				Accuracy : 67 %

				Learning Rate 0.1 and number of hidden layers 4 :
				Accuracy : 66 %

				Learning Rate 0.3 and number of hidden layers 3 :
				Accuracy : 68 %

				Learning Rate 0.4 and number of hidden layers 3 :
				Accuracy : 67 %

				Learning Rate 0.4 and number of hidden layers 4 :
				Accuracy : 66 %

				Learning Rate 0.4 and number of hidden layers 5 :
				Accuracy :  66 %

				Learning Rate 0.5 and number of hidden layers 2 :
				Accuracy : 39 %

				Learning Rate 0.5 and number of hidden layers 3 :
				Accuracy : 69 %

				Learning Rate 0.5 and number of hidden layers 4 :
				Accuracy : 68 %

				Learning Rate 0.5 and number of hidden layers 5 :
				Accuracy : 69 %

				Learning Rate 0.7 and number of hidden layers 4 :
				Accuracy : 70 %

				Learning Rate 0.7 and number of hidden layers 5 :
				Accuracy : 67 %

				Learning Rate 0.7 and number of hidden layers 8 :
				Accuracy : 63 %

				Learning Rate 0.8 and number of hidden layers 5 :
				Accuracy : 65 %



# Problems faced when optimizing Neural Networks code:

 The biggest problem faced when building the ANN is non propagation of weights as the input value was too high. The input value was then normalized by dividing it by 255. Even then the values did not converge as the input to outer layer or the hidden activation value was getting too high and hence when passed inside the sigmoid function was resulting in constant errors. Then we proceeded to normalize the hidden activations too. this was done by the following formula:

sum - max / min-max

reference: http://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range

The network started to train after this. After a long and strenuous session of running tests repetitively, we have arrived on a fact that with 0.3 as learning rate and hidden layer size 3 we have accuracy constantly between 68% and 72%. We were getting higher values at 0.7 and 4 , 0.5 and 5 but these were not constant. Due to the randomized starting weights and the haphazard weight propagations, these values sometimes come down to as low as 60% but now below. So 0.3 and 3 doesn't overfit and will yield a formidable result. Sometimes due to bad propagation the outputs may not converge, these will be visible as the output neurons for those will be giving a constant 0.9999 output for a long time (till trained even). But these cases are random and sparse.

# Conclusion :

In terms of learning rate :
When we tried with very small value of alpha, then it was taking longer time to converge or complete. Whereas with too large value of alpha it was again taking a lot of time to converge and often times was not completing because the local minima was getting lost because of high learning rate.

In terms of different number of hidden layers :
If we don’t have hidden layer then it is just one to one mapping from input to output.
If we have too many hidden layers then inputs will have less impact on the output.

So, we figured out the best results that we are getting is with below combinations:
With learning rate, alpha as 0.3 and number of hidden layers as 3.

You can run the program as follows:

					$ python nnet.py train-data.txt test-data.txt < hidden layer size > < learning rate >

The final output is also written to a file called nnet.txt for reference later.


(Based on the problem statement in CS B551 Elements of Artificial Intelligence by Professor David J Crandall , Indiana University, Bloomington)
