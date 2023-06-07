# Experimental Evaluation: Data and Scripts #

## General Information ##

This folder contains the data we obtained by conducting the experimental procedure described in the paper. We used this data to generate the tables reported in the paper.

## Dependencies ##

The scripts require `python 3.8`

### Configure Ubuntu ###
Pull an Ubuntu Docker image, run and configure it by typing in the terminal:

``` 
docker pull ubuntu:bionic
docker run -it --rm ubuntu:bionic
apt-get update && apt-get upgrade -y && apt-get clean
apt-get install -y software-properties-common
```

### Copy the project into the docker container ###

To copy DeepAtash inside the docker container, open another console and run:

``` 
docker cp <DEEP_ATASH_HOME>/ <DOCKER_ID>:/
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
cd /DeepAtash/experiments
python3 -m venv .venv
```

Activate the python virtual environment and updated `pip` again (venv comes with an old version of the tool):

```
. .venv/bin/activate
pip install --upgrade pip
```
### Installing Dependencies ###

The dependencies can be installed via `pip`:

```
pip install -r requirements.txt
``` 

## (Re)Generate the plots ##

To regenerate the plots, run the following command from the **current** folder:
```
cd data
tar -xvf mnist-data
tar -xvf imdb-data
cd ..
mkdir plots
python rq1.py
python rq2.py
python rq3.py
```

> NOTE: These commands may produce RuntimeWarnings. Do not worry about them. The commands are successful if the plots are stored.

Then, you will find the following files in `plots` folder:

* `RQ1-MNIST-dark-table.txt`, `RQ1-MNIST-grey-table.txt`, `RQ1-MNIST-white-table.txt` (Table 2: RQ1)
* `RQ1-IMDB-dark-table.txt`, `RQ1-IMDB-grey-table.txt`, `RQ1-IMDB-white-table.txt` (Table 2: RQ1)
* `RQ2-MNIST-dark-table.txt`, `RQ2-MNIST-grey-table.txt`, `RQ2-MNIST-white-table.txt` (Table 3: RQ2)
* `RQ2-IMDB-dark-table.txt`, `RQ2-IMDB-grey-table.txt`, `RQ2-IMDB-white-table.txt` (Table 3: RQ2)
* `RQ3-IMDB-table.txt`, `RQ3-MNIST-table.txt` (Table 4: RQ3)

These tables correspond to the ones reported in tables 2-4 of the (pre-print) version of the ISSTA paper.
To check the results, you can copy the files from the running docker to your system, as follows:

```
docker cp <YOUR_DOCKER_NAME>:/DeepAtash/experiments/plots  /path-to-your-Desktop/
```

