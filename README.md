# DECharge

## Introduction

Discomfort and Efficient-aware electric vehicle Charge coordination (DECharge) is a decentralized Electric Vehicle (EV) charging control framework based on collective learning of I-EPOS. It allows EVs to autonomously select charging stations for charge while minimizing travel and queuing time. 

## Setup

From builds upon Python 3.7 to 3.9

### 1. Clone this repo:
```
git clone git@github.com:TDI-Lab/DECharge.git
```

### 2. Modify parameters

- Modify the properties of algorithms in `conf/epos.properties`.

- Modify the parameters of scenarios in `main.py`.

### 3. Run the code:
```
python main.py
```

## Code structure

```
├── LICENSE
├── README.md                       <- The top-level README for developers using this project.
├── IEPOS.jar                       <- The jar file to run I-EPOS in the DECharge framework
├── conf
│   ├── epos.properties             <- The parameters of the I-EPOS approach
│   ├── log4j.properties             
│   ├── measurement.conf             
│   ├── protopeer.conf            
├── datasets
│   ├── Stations                    <- The input historical dataset of charging stations
│   ├── ChargingDemands             <- The input generated dataset of EV charging requests 
│   ├── EVdemands                   <- The generated dataset as the input of I-EPOS
├── env
│   ├── RealWorld.py                <- Create the environent of the real scenarios
└────── main.py                     <- Case study and scenario settings
```


## Documents

More details of I-EPOS can be found [here](https://github.com/epournaras/EPOS).

The historical datasets of charging stations in Paris can be found [here](https://challengedata.ens.fr/login/?next=/participants/challenges/57/).

The historical datasets of EV charging requests in South Korea can be found [here](https://doi.org/10.6084/m9.figshare.22495141.v1).


## Citation

If you use DECharge in any of your work, please cite our paper:
~~~

~~~