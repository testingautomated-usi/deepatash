# Getting Started #

Follow the steps below to set up DeepAtash and validate its general functionality.


## Step 1: Configure the environment  ##

Pull our pre-configured Docker image for DeepAtash-MNIST:

``` 
docker pull anonissta/deepatash-image:v1.0
```

Run it by typing in the terminal the following commands:

```
docker run -it --rm anonissta/deepatash-image:v1.0
cd deepatash/MNIST
. .venv/bin/activate
```

## Step 2: Run DeepAtash ##
Use the following commands to start a 10 mins run of DeepAtash-MNIST with ga algorithm and the "Bitmaps - Orientation" combination of features:

```
python deepatash_mnist.py
```

> NOTE: `config.py` contains the tool configuration. You should edit this file to change the configuration. For example, if you want to run <i>DeepAtash-MNIST</i> with the same configuration as in the paper, you need to set the `RUN_TIME` variable inside `config.py` as follows:
```
RUN_TIME  = 3600
```

When the run ends, on the console you should see a message like this:

```
2023-04-14 14:27:41,494 INFO elapsed_time: xxx
Game Over
```

The tool produces the following outputs in the `logs/XXX` folder (where XXX is the name of folder based on your configurations):

* `log.txt`:the log info of the whole run;
* `config.json`: file containing the configuration used for the run;
* generated inputs close and within the target area  (in image and npy formats);




## Step 3: Reproduce Experimental Results ##

In case you want to regenerate the tables in the paper without re-running all the 100h+ of experiments, we provided the data of all runs of all the tools in `experiments/data`. 

To regenerate the table values reported in the paper, run the commands we report below on the provided docker.

```
cd /deepatash/experiments
. .venv/bin/activate
cd ..
cd experiments/data
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
