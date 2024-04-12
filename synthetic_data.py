from ctgan import CTGAN
from ctgan import load_demo
import pandas as pd

df = pd.read_csv('./train.csv')


print(df)

X = df
y = X['Education']
for row in X.index:
    # Check for Dr. in the list
    if "Dr." in X.loc[row, "Candidate"]:
        X.loc[row, "isDoctor"] = 1
    else:
        X.loc[row, "isDoctor"] = 0
    if "Adv." in X.loc[row, "Candidate"]:
        X.loc[row, "isAdvocate"] = 1
    else:
        X.loc[row, "isAdvocate"] = 0
    if "SC" in X.loc[row, "Candidate"]:
        X.loc[row, "isSC"] = 1
    else:
        X.loc[row, "isSC"] = 0
    if "ST" in X.loc[row, "Candidate"]:
        X.loc[row, "isST"] = 1
    else:
        X.loc[row, "isST"] = 0
X = X.drop('Candidate',axis=1)
X = X.drop('Constituency ∇',axis=1)
X.drop('ID',axis=1,inplace=True)
for row in X.index:
    #Remove WhiteSpaces in State in the middle of the string.
    X.loc[row, "state"] = X.loc[row, "state"].replace(" ", "")
    # print(X.loc[row, "Total Assets"][-4:])
    if(isinstance(X.loc[row,  "Total Assets"], float)):
        X.loc[row,  "Total Assets"] = "0"
    elif(X.loc[row, "Total Assets"][-4:] == "Lac+"):
        X.loc[row, "Total Assets"] = X.loc[row, "Total Assets"][:-4]
        X.loc[row, "Total Assets"] = str(int(X.loc[row, "Total Assets"]) * 100)
    elif(X.loc[row, "Total Assets"][-6:] == "Crore+"):
        X.loc[row, "Total Assets"] = X.loc[row, "Total Assets"][:-6]
        X.loc[row, "Total Assets"] = str(int(X.loc[row, "Total Assets"]) * 10000)
    elif(X.loc[row,  "Total Assets"][-5:] == "Thou+"):
        X.loc[row,  "Total Assets"] = X.loc[row,  "Total Assets"][:-5]
        X.loc[row,  "Total Assets"] = str(int(X.loc[row,  "Total Assets"]))
    elif(X.loc[row,  "Total Assets"][-5:] == "Hund+"):
        X.loc[row,  "Total Assets"] = "0"
for row in X.index:
    if(isinstance(X.loc[row, "Liabilities"], float)):
        X.loc[row, "Liabilities"] = "0"
    elif(X.loc[row, "Liabilities"][-4:] == "Lac+"):
        X.loc[row, "Liabilities"] = X.loc[row, "Liabilities"][:-4]
        X.loc[row, "Liabilities"] = str(int(X.loc[row, "Liabilities"]) * 100)
    elif(X.loc[row, "Liabilities"][-6:] == "Crore+"):
        X.loc[row, "Liabilities"] = X.loc[row, "Liabilities"][:-6]
        X.loc[row, "Liabilities"] = str(int(X.loc[row, "Liabilities"]) * 10000)
    elif(X.loc[row, "Liabilities"][-5:] == "Thou+"):
        X.loc[row, "Liabilities"] = X.loc[row, "Liabilities"][:-5]
        X.loc[row, "Liabilities"] = str(int(X.loc[row, "Liabilities"]))
    elif(X.loc[row, "Liabilities"][-5:] == "Hund+"):
        X.loc[row, "Liabilities"] = "0"
# Mapping Parties to unique values in the dataframe using map function
# Create a dictionary of the parties and their corresponding unique values
party = X['Party'].unique()
party = {value: idx for idx, value in enumerate(party)}
# Add a value for nan parties
party[""] = -1
# Mapping the values to the dataframe
X['Party'] = X['Party'].map(party)
state = X['state'].unique()
state = {value: idx for idx, value in enumerate(state)}
# Mapping the values to the dataframe
X['state'] = X['state'].map(state)
education = y.unique()
education = {value: idx for idx, value in enumerate(education)}
# Mapping the values to the dataframe
y = y.map(education)
X['Education'] = y
# real_data = load_demo()
# print(real_data)

print(party)
# Names of the columns that are discrete
discrete_columns = [
    'isDoctor',
    'isAdvocate',
    'isSC',
    'isST',
    'Party',
    'Criminal Case',
    'state',
    'Education',
    'Liabilities',
    'Total Assets'
]

ctgan = CTGAN(epochs=1000, log_frequency=True, verbose=True)
ctgan.fit(X, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(8000)
final_data = []

for row in synthetic_data.index:
    if(int(synthetic_data.loc[row, "Party"]) > len(party)-1):
        print("yes")
        continue
    if(int(synthetic_data.loc[row, "state"]) > len(state)-1):
        continue
    if(int(synthetic_data.loc[row, "Education"]) > len(education)-1):
        continue
    final_data.append(synthetic_data.loc[row])

print(len(final_data))

# Save the synthetic data
reverse_party = {value: key for key, value in party.items()}
reverse_edu = {value: key for key, value in education.items()}
reverse_state = {value: key for key, value in state.items()}

for row in synthetic_data.index:
    synthetic_data.loc[row, "Party"] = reverse_party[synthetic_data.loc[row, "Party"]]
    synthetic_data.loc[row, "state"] = reverse_state[synthetic_data.loc[row, "state"]]
    synthetic_data.loc[row, "Education"] = reverse_edu[synthetic_data.loc[row, "Education"]]

synthetic_data.to_csv('synthetic_data_5ke_8k_extracted_2.csv', index=False)