# FICO dataset
## Files:
`fico_data.csv`: raw FICO dataset from [Explainable Machine Learning Challenge](https://community.fico.com/s/explainable-machine-learning-challenge)

`fico_helper.csv`: contains information about feature types including which feature is the label

`fico_raw_script.py`: this script processes the data and saves it to a new csv

## Information about the dataset:

### Dataset purpose
The dataset was created to predict repayment on Home Equity Line of Credit (HELOC) applications. It's used by lenders to determine how much credit should be granted. HELOC credit lines are loans that use people's homes as collateral. 

### Dataset creation
The anonymized verion of the HELOC dataset was created by FICO to put forth an explainable machine learning challenge. The winners received a $5,000 prize. 

### Instances
Each instance is a real credit application for HELOC credit; its an application that a single person submitted and contains information about that person. There are 10,459 instances, each consisting of 23 features. These features are either binary or discrete. The label, RiskPerformance, is a binary assessment of risk of repayment based on the 23 predictors. 1 means the person hasn't been more than 90 days overdue on their payments in the last 2 years; 0 means they have at least once. 

### Noise
There are some repeated instances; there are 9,871 unique rows. The dataset is self-contained, and has been anonymized for public use in the explainability challenge. It doesn't use any protected attributes like race and gender. 

### Collection
The data is from real loan applications. 

### Processing
We preprocessed the data by cutting down the number of features considered to 7: RiskPerformance, AverageMInFile, ExternalRiskEstimate, MaxDelqEver, MSinceMostRecentDelq, MSinceOldestTradeOpen, MSinceMostRecentTradeOpen. We made the feature AverageMInFile a thermometer encoding by dividing the values by 12 to get the number of years and creating 4 dummy variables: AvgYearsInFileGeq3, AvgYearsInFileGeq5, AvgYearsInFileGeq7, AvgYearsInFileGeq9. Clearly, the features are correlated; any time Years In File is greater than or equal to 5, it's also greater than or equal to 3. We made a new feature YearsOfAcctHistory by converting MSinceOldestTradeOpen to years and capping this value at 2 years so anybody who has 2 or more years of account history has feature value 2. We made MaxDelqEver and MSinceMostRecentDelq binary and named the new features AnyDerogatoryComment and AnyDelTradeInLastYear. The result of this preprocessing is saved in `fico_preprocessed_1.csv`. 

References
- https://community.fico.com/s/explainable-machine-learning-challenge?tabset-158d9=3
- https://pbiecek.github.io/xai_stories/story-heloc-credits.html
