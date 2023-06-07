# DeepAtash-MNIST #

Focused test generator for DL systems

## General Information ##
This folder contains the application of the DeepAtash approach to the handwritten digit classification problem.
This tool is developed in Python on top of the DEAP evolutionary computation framework. It has been tested on a machine featuring an i7 processor, 16 GB of RAM, an Nvidia GeForce 940MX GPU with 2GB of memory. These instructions are for Ubuntu 18.04 (bionic) OS and python 3.6.

## Dependencies ##

> NOTE: If you want to use DeepAtash/MNIST easily without configuring your environment from scratch, you can also see [__Getting Started__](../documentation/getting_started.md)

### Configuring Ubuntu ###
Pull an Ubuntu Docker image, run and configure it by typing in the terminal:

``` 
docker pull ubuntu:bionic
docker run -it --rm ubuntu:bionic
apt update && apt-get update
apt-get install -y software-properties-common
```

### Installing git ###
Use the following command to install git:

``` 
apt install -y git
```

### Copy the project into the docker container ###

To copy DeepAtash/MNIST inside the docker container, open another console and run:

``` 
cd <DEEP_ATASH_HOME>
docker cp DeepAtash/MNIST/ <DOCKER_ID>:/
```

Where `<DEEP_ATASH_HOME>` is the location in which you downloaded the artifact and `<DOCKER_ID>` is the ID of the ubuntu docker image just started.

You can find the id of the docker image using the following command:

```
docker ps -a

CONTAINER ID   IMAGE           COMMAND       CREATED          STATUS          PORTS     NAMES
13e590d65e60   ubuntu:bionic   "/bin/bash"   2 minutes ago   Up 2 minutes             recursing_bhabha
```

### Installing Python 3.8 ###
Install Python 3.8:

``` 
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.8
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 0
```

And check if it is correctly installed, by typing the following command:

``` 
python3 -V

Python 3.8.16
```

Check that the version of python matches `3.8.*`.

### Installing pip ###

Use the following commands to install pip and upgrade it to the latest version:

``` 
apt install -y python3-pip
python3 -m pip install --upgrade pip
```

Once the installation is complete, verify the installation by checking the pip version:

``` 
python3 -m pip --version

pip 23.1.2 from /usr/local/lib/python3.8/dist-packages/pip (python 3.8)
```
### Creating a Python virtual environment ###

Install the `venv` module in the docker container:

``` 
apt-get install python3.8-dev python3.8-venv
```

Create the python virtual environment:

```
cd /DeepAtash/MNIST
python3 -m venv .venv
```

Activate the python virtual environment and updated `pip` again (venv comes with an old version of the tool):

```
. .venv/bin/activate
pip install --upgrade pip
```

### Installing Python Binding to the Potrace library ###
Install Python Binding to the Potrace library.

``` 
apt install -y build-essential python-dev libagg-dev libpotrace-dev pkg-config
``` 

Install `pypotrace` (commit `76c76be2458eb2b56fcbd3bec79b1b4077e35d9e`):

``` 
cd /
git clone https://github.com/flupke/pypotrace.git
cd pypotrace
git checkout 76c76be2458eb2b56fcbd3bec79b1b4077e35d9e
pip install numpy
pip install .
``` 

To install PyCairo and PyGObject, we follow the instructions provided by [https://pygobject.readthedocs.io/en/latest/getting_started.html#ubuntu-getting-started](https://pygobject.readthedocs.io/en/latest/getting_started.html#ubuntu-getting-started).

``` 
apt install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0
apt install -y libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0 librsvg2-dev
``` 

### Installing Other Dependencies ###

This tool has other dependencies, including `tensorflow` and `deap`, that can be installed via `pip`:

```
cd /DeepAtash/MNIST
pip install -r requirements.txt
``` 

## Usage ##
### Input ###

* A trained model in h5 format. The default one is in the folder `models`;
* `config.py` containing the configuration of the tool selected by the user.

### Run the Tool ###

To run the approach use the following command:

```
python deepatash_mnist.py
```


### Output ###

When the run is finished, the tool produces the following outputs in the `logs` folder:

* folder containing the generated inputs (in image format).


### Fine tuning # 

To use the tests generated with DeepAtash for fine tuning the model. You need to run the tool considering three different target areas (i.e. dark, grey and white) of the feature maps. Some suggested target cell are already defined in the `config.py` for each feature combinations:

```
# these goal cells computed from 10 times of running DeepHyperion (state of the art approach):
# goal cell for white area mov-lum (11, 3)  or-lum (10, 2) move-or  (17, 10)
# goal cell for grey area mov-lum (21, 9) or-lum (19, 4) move-or (16, 11)
# goal cell for dark area mov-lum (6, 0) or-lum (4, 1) move-or (7, 5)
```

Note: make sure after each run move the output of your run to the corresponding folder based on its target area, as follows:
```
mkdir ../experiments/data/mnist/DeepAtash/target_cell_in_dark
mkdir ../experiments/data/mnist/DeepAtash/target_cell_in_grey
mkdir ../experiments/data/mnist/DeepAtash/target_cell_in_white
cp logs/{YOUR_RUN_OUTPUT} ../experiments/data/mnist/DeepAtash/target_cell_in_{YOUR_SELECTED_AREA}/
```

After having at least one run for each target area and feature combination, you can start retraining the model as follows:

```
python retrain.py
```



## Troubleshooting ##

* if pip cannot install the correct version of `opencv-python` check whether you upgraded pip correctly after you activate the virtual environment `.venv`

* If tensorflow cannot be installed successfully, try to upgrade the pip version. Tensorflow cannot be installed by old versions of pip. We recommend the pip version 20.1.1.

* If the import of cairo, potrace or other modules fails, check that the correct version is installed. The correct version is reported in the file requirements.txt. The version of a module can be checked with the following command:

```
pip3 show modulename | grep Version
```
    
To fix the problem and install a specific version, use the following command:
    
```
pip3 install 'modulename==moduleversion' --force-reinstall
```
