# Feature Responsiveness Scores: Model-Agnostic Explanations for Recourse

## Requirements
- Python 3.9+

To install requirements:

```sh
pip install -r requirements.txt
```

### CPLEX
To run experiment code, you will need to download CPLEX (v. 22.1). CPLEX is cross-platform optimization tool solver a Python API. Running the `pip install` above will download the community edition (CE) from [pip](https://pypi.org/project/cplex/). The problem size allowed on the CE version will not be enough to run the entire experiment. But the full version of CPLEX is free for students and faculty members at accredited institutions. To download CPLEX:

1. Register for [IBM OnTheHub](https://ur.us-south.cf.appdomain.cloud/a2mt/email-auth)
2. Download the *IBM ILOG CPLEX Optimization Studio* from the [software catalog](https://www-03.ibm.com/isc/esd/dswdown/searchPartNumber.wss?partNumber=CJ6BPML)
3. Install CPLEX Optimization Studio.
4. Setup the CPLEX Python API [as described here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

If you have problems with CPLEX, please check the [CPLEX user manual](http://www-01.ibm.com/support/knowledgecenter/SSSA5P/welcome) or the [CPLEX forums](https://www.ibm.com/developerworks/community/forums/html/forum?id=11111111-0000-0000-0000-000000002059).

## Directory Structure
```
├── data         # datasets
├── reach        # modified code from reachml
├── src          # our source code for generating explanations
└── scripts      # scripts that call the source code for testing and running experiments
```
[reachml](https://github.com/ustunb/reachml) refers to the source code to generate reachable sets from Kothari et al. (2024).

## Running Experiment
To run the experiments in the paper, run from the repository:
```sh
python3 scripts/experiments/run_experiment_pipeline.py --data_name DATASET_NAME
```
where `DATASET_NAME` is one of `german`, `fico`, `givemecredit`.

To generate the metrics shown in Table 3:
```sh
python3 scripts/metrics/combine_metrics.py
```
will result in a csv file named `all_metrics.csv` in a `results` directory.