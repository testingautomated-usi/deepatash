# DeepAtash

DeepAtash is a tool for generating focused test inputs for deep learning systems.


## DeepAtash-MNIST ##
To set up the environment and run the DeepAtash tool adapted to the handwritten digit classification case study, follow the instructions [here](../MNIST/README.md).


## DeepAtash-IMDB ##
To set up the environment and run the DeepAtash tool adapted to the movie sentiment case study, follow the instructions [here](../IMDB/README.md). 


## Experimental Data and Scripts ##
To regenerate the results and plots reported in the ISSTA paper, follow the instructions [here](../experiments/README.md) 

## Extra Use Case Scenarios ##
This section contains plausible scenarios on how DeepAtash could be extended beyond the experiments performed in the paper.

### Scenario 1: Try a new target cell ###

This scenario shows the posiibility of generating tests for _DeepAtash-IMDB_ different from the ones considered in the experimental evaluation.
As an example, you can configure _DeepAtash-IMDB_ to use the alternative target cell.
To do this, you need to place the coordinates of the cekk in _IMDB/config.py_ as follows:

```
    GOAL = (4, 11)
```

Note: you should be carefull to select a GOAL in the map boundaries, the number of cells for each feature of the map is detemined as follows:

```
    NUM_CELLS = 25
```
As an example, if you have 25 as number of cells then your target cell coordinates should be in the range of [0, 24].


### Scenario 2: Test a different DL model ###

This scenario shows the possibility of using trained models for _DeepAtash-MNIST_ different from the one considered in the experimental evaluation.
As an example, you can configure _DeepAtash-MNIST_ to use the alternative model `cnnClassifier.h5` which is already in _models_ folder.
To do this, you need to place the name of the new model in _MNIST/config.py_ as follows:

```
    MODEL = 'models/cnnClassifier.h5'
```

Moreover you can test your own model. You can train a new model by running the following command:

```
python MNIST/train_model.py <YOUR_MODEL_NAME>
```

After the new model has been trained, you can place it in _MNIST/models_ folder, then edit the configuration in _MNIST/config.py_ file as follows:

```
    MODEL = 'models/<YOUR_MODEL_NAME>.h5'
```
