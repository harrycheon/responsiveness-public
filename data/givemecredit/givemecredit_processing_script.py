import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import os

abs_path = os.path.abspath(os.getcwd())
df = pd.read_csv(abs_path + "/data/give_me_credit/give_me_credit_encoded.csv")

df['SeriousDlqin2yrs'] = df['SeriousDlqin2yrs'].replace({0: 1, 1: 0})
df = df.rename(columns={"SeriousDlqin2yrs": "NotSeriousDlqin2yrs"})

raw_file = abs_path + "/data/give_me_credit/give_me_credit_data.csv"
df.to_csv(raw_file, header = True, index = False)
print("results saved!")


