# FAT-WiSARD
A Weighted Weightless Neural Network.

## Dependencies
Numpy and Tensorflow

## Documentation ( Quick Model )
### Create Model:
  model = FAT_WiSARD(INPUT_SIZE,OUTPUT_SIZE/NUMBER_OF_CLASSES)
### Add Layer:
  model.add(NUMBER_OF_NEURONS)
### Train:
  model.train(INPUTS,OUTPUTS) OUTPUTS NEEDS DO BE ONE HOT ENCODED
### Opimize:
  model.optimize(LEARNING_RATE,NUMBER_OF_EPOCHS=1)
