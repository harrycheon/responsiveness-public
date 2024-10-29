# German dataset
## Files:
`german_processed.csv`: processed german dataset

`german_processing_script.py`: this script processes the data and saves it to a new csv

`german_raw.csv`: the procesed german dataset. The raw version is found in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

## Information about the dataset:

### Dataset purpose
This is a german credit risk dataset, created to predict somebody's credit risk. It's used to classify people as having good or bad credit risk based on predictive features. 

### Dataset creation
The dataset was donated to the UCI Machine Learning Repository and contains information on loan history, demographic information, and occupation, payment history, and whether or not somebody is a good customer. 

### Instances
Each instance is a real person with credit. There are 1,000 instances, each consisting of 20 features. The features are all categorical or integers. The label, class, is a binary indicator of whether someone is a 'good' or 'bad' customer. 1 means the person is good, 2 means they're bad. 

### Noise
There are no missing values in the dataset. The variable names aren't very descriptive of what they indicate, and thus lots of preprocessing is required. The dataset is self-contained, and is anonymous. It does have gender, age, and marital status features. 

### Collection
The data is from real people who are in the german database. 

### Processing
The data is preprocessed by renaming the features to be indicative of the values they represent.  
