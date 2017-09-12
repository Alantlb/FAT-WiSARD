# FAT-WiSARD
A Weighted Weightless Neural Network.

## Dependencies
Numpy and Tensorflow

## Documentation ( Quick Model )
### Create Model:
  model = FAT_WiSARD(input_size, output_size)
### Add Layer:
  model.add(number_of_neurons)
### Train:
  model.train(input_batch, output_batch) Output -> One Hot Encoded
### Opimize:
  model.optimize(learning_rate, number_of_epochs)
