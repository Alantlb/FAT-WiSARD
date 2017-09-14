# FAT-WiSARD
A Weighted Weightless Neural Network.

## Depencies:
Python 3.X Numpy and Tensorflow

## Docs:
### Import:
from FATWiSARD import *
### Create new Model:
model = FATWiSARD ( number_of_inputs, number_of_classes, number_of_rams, radius_initial_value = 1 )
### Train:
model.train ( inputs, classes )
### Opimize:
model.optimize ( inputs, classes, learning_rate, number_of_epochs = 1 ) <br/>
Opitional Kwargs: 
optimizer = "Adam" -> Uses Adam optimizer instead of Gradient Descent.
log = n -> Prints on the screen "Epoch: x" at a step of n.
OBS: classes can be one hot encoded or not. 
### Predict:
model.predict ( input )
