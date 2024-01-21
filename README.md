# DeepTraderX (DTX) running in the Threaded-BSE

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)
![linting: pylint](https://github.com/armandcismaru/DeepTrader-on-Threaded-BSE/actions/workflows/pylint.yml/badge.svg)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

This repository was created as part of my final individual project for the MEng Computer Science 4th year dissertation. It aims to explore the dynamics of a high performing, Deep Learning based trader, trained solely on data derived from the observation of Level-2 data from the Limit Order Book (LOB) of a simulated financial exchange. The project is using the [Threaded Bristol Stock Exchange (TBSE)](https://github.com/MichaelRol/Threaded-Bristol-Stock-Exchange) which was built with the purpose of replicating the asynchronous and parallel nature of real life markets, being a multithreaded extension of the [Bristol Stock Exchange (BSE)](https://github.com/davecliff/BristolStockExchange), created by Dave Cliff.

The results of my research indicate positive results of DeepTraderX (DTX) versus the other, well established, public-domain literature. In some cases the profits are overwhelmingly higher. All the details can be found in my [thesis](fz19792_meng_dissertation.pdf). 

This work is the base of [Cismaru, A. (2024). DeepTraderX: Challenging Conventional Trading Strategies with Deep Learning in Multi-Threaded Market Simulations.](cismaru2024.pdf), due to be published in the proceedings of [ICAART 2024](https://icaart.scitevents.org) in Rome, Italy. 

Feel free to message me if you want to use this work and need any details. Please star this repository if you find its contents useful.

## Running instructions

### Instructions for running on local machine

The project is written in Python 3.4 or above, so make sure you have that installed. The project uses a number of dependencies, which can be installed by running the following command in the root directory of the project (using pip):

```console
$ pip install -r requirements.txt
```

The project is made up of 2 main components: the market simulation, TBSE and the trading agent, DTX.
The ```deep_trader``` directory in ```src``` contains the code used to train the Neural Network model. The trained models are stored in the ```models``` directory. The ```tbse``` directory in ```src``` contains the code used to run the simulation.

DTX can be tested against the other trading strategies by runnning TBSE, which can be done in three different ways. These are the three different ways to enter the trader schedule. The trader schedule is the number of each type of trader present in the market session. It should be noted that in TBSE the buyer schedule is always equal to the seller schedule, i.e. there are the same number of buyers of each type as there are sellers. So if your schedule is 5 DTR and 5 AA, that means you will have 5 DTR buyers, 5 AA buyers, 5 DTR sellers and 5 AA sellers for a total of 20 traders. There are 7 traders available in TBSE, these are ZIC, ZIP, GDX, AA, Giveaway, Shaver and our AI agent, DTR. The three ways to specify this schedule are:

#### - From the config file:

```console
$ python3 tbse.py
```
By entering no command-line arguments TBSE will use the order schedule as it exists in ```config.py```(lines 16-22).

#### - From the command-line:

```console
$ python3 tbse.py [zic],[zip],[gdx],[aa],[gvwy],[shvr],[dtx]
```
Where each trader name is replaced with the number of that trader you want in the market schedule. For example:
```console
$ python3 tbse.py 0,0,0,5,0,0,5
```
will produce a trader schedule with 5 AA buyers, 5 AA sellers, 5 DTX buyers and 5 DTX sellers. You must enter a number for each of the 7 trader types, so put 0 if you do not want a certain trader present in your market session.

#### - From a CSV file:

```console
$ python3 tbse.py markets.csv
```

Using a CSV file is the most versatile way to use TBSE as it allows multiple market sessions to be defined using different trader schedules. The file ```markets.csv``` is provided. TBSE will run each row of the CSV file as a separate market session. Each row must contain 7 comma-separated numbers in the order ZIC, ZIP, GDX, AA, Giveaway, Shaver, DTX so 0 should be used if you don't wish a trader to be present in a market session. The following example CSV file will run experiments comparing 5 vs 5 of every possible pair of traders:

```
5, 5, 0, 0, 0, 0, 0
5, 0, 5, 0, 0, 0, 0
5, 0, 0, 5, 0, 0, 0
5, 0, 0, 0, 5, 0, 0
5, 0, 0, 0, 0, 5, 0
5, 0, 0, 0, 0, 0, 5
0, 5, 5, 0, 0, 0, 0
0, 5, 0, 5, 0, 0, 0
0, 5, 0, 0, 5, 0, 0
0, 5, 0, 0, 0, 5, 0
0, 5, 0, 0, 0, 0, 5
0, 0, 5, 5, 0, 0, 0
0, 0, 5, 0, 5, 0, 0
0, 0, 5, 0, 0, 5, 0
0, 0, 5, 0, 0, 0, 5
0, 0, 0, 5, 5, 0, 0
0, 0, 0, 5, 0, 5, 0
0, 0, 0, 5, 0, 0, 5
0, 0, 0, 0, 5, 5, 0
0, 0, 0, 0, 5, 0, 5
0, 0, 0, 0, 0, 5, 5
```

### Config

Market sessions ran in TBSE can be configured by editing ```config.py```. It should be noted that lines 67 onwards are for verifying the content of the configuration file and should not be changed. These lines will alert the user if they have misconfigured TBSE.

The comments within the config file should be enough for a user to understand how to configure TBSE, any missing information should be found in the [BSE Guide](https://github.com/davecliff/BristolStockExchange/blob/master/BSEguide1.2e.pdf "BSE Guide") which describes things like the different stepmodes and timemodes available. 
To run the simulation, run the following command in the root directory of the project:

```console
$ python deep_trader_tbse/tbse.py
```

The results of the simulation will be stored in a file of the form ```00-05-00-00-00-00-05.csv``` in the ```deep_trader_tbse``` directory. The first 7 numbers in the filename are the trader schedule used in the market session. Each line in the file will contain the following information:

```
trial_id, time, trader1_type, total_profit, number_of_traders1, profit_per_trader2, trader2_type, total_profit, number_of_traders2, profit_per_trader2, best_bid, best_ask
```

### Instructions for deploying on cloud clusters

This project was designed to be run on Kubernetes cloud clusters. The following instructions are for deploying the project on the [AWS EKS](https://aws.amazon.com/eks/ "AWS EKS") cloud cluster. The project can be deployed on other cloud clusters, but the instructions will be different. The instructions assume the existence of an AWS account and a Kubernetes cluster on AWS EKS. The instructions also assume that the user has the [AWS CLI](https://aws.amazon.com/cli/ "AWS CLI") installed and configured with their AWS account credentials.

The latest version of the simulation is available as a Docker image on [Docker Hub](https://hub.docker.com/repository/docker/armandcismaru/deeptrader/general). The image can be pulled from Docker Hub using the following command:

```console
$ docker pull armandcismaru/deeptrader:deeptrader2.5
```

The image can be run locally using the following command:

```console
$ docker run armandcismaru/deeptrader:deeptrader2.5
```

The image is designed to run a series of simulations using the ```markets.csv``` file. The results are stored in a remote S3 bucket (configured at lines 931-934). To update the image with a new ```markets.csv``` file, the user must build and push a new image to their own image repository using the following command and update its tag version:

```console
$ docker build -t username/deeptrader:deeptrader2.x .
$ docker push username/deeptrader:deeptrader2.x
```

The user needs to update the ```markets-simulations.yaml``` file with the new name an version of the image.

Once the image is pushed to Docker Hub, it can be deployed on the AWS EKS cluster using the following command:

```console
$ kubectl apply -f deep_trader_tbse/markets-simulations.yaml
```

The user can check the status of the pods using the following command:

```console
$ kubectl get pods
```

This way the simulation can be spread over multiple pods and run in parallel. In reality, the parallelism comes from running multiple instances of the same code when we need a lot of market simulations. Our job was configured for 9 pods but it can be configured for more or less depending on the user's needs.

## License
The code is open-sourced via the [MIT](http://opensource.org/licenses/mit-license.php) Licence: see the LICENSE file for full text. 
