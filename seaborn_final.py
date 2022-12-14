
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("../input/trainingdata/train.csv")
test_df = pd.read_csv("../input/testingdata/test.csv")

train_df.shape

"""This is moderately sized data. heavy preprocessing is required to extract meaningful features(columns) and remove faulty/ redundant data entries(rows)."""

missing_values_count = train_df.isnull().sum()
print (missing_values_count[0:10])
total_cells = np.product(train_df.shape)
total_missing = missing_values_count.sum()
print ("% of missing data = ",(total_missing/total_cells) * 100)

catagorical_features=train_df.select_dtypes(include=['category','object']).columns
continuous_features = list(filter(lambda x: x not in catagorical_features, train_df))
catagorical_features

"""# Preprocessing

Removing rows with columns having less than 1 percent NULL values.
"""

train_df.drop(axis="rows", labels=train_df.index[train_df['card2'].isna()], inplace=True)
train_df.drop(axis="rows", labels=train_df.index[train_df['card3'].isna()], inplace=True)
train_df.drop(axis="rows", labels=train_df.index[train_df['card4'].isna()], inplace=True)
train_df.drop(axis="rows", labels=train_df.index[train_df['card5'].isna()], inplace=True)
train_df.drop(axis="rows", labels=train_df.index[train_df['card6'].isna()], inplace=True)
train_df.drop(axis="rows", labels=train_df.index[train_df['D1'].isna()], inplace=True)
train_df.drop(axis="rows", labels=train_df.index[train_df['V95'].isna()], inplace=True)
train_df.drop(axis="rows", labels=train_df.index[train_df['V279'].isna()], inplace=True)
train_df.drop(axis="rows", labels=train_df.index[train_df['V281'].isna()], inplace=True)

test_df.drop(axis="rows", labels=test_df.index[test_df['card2'].isna()], inplace=True)
test_df.drop(axis="rows", labels=test_df.index[test_df['card3'].isna()], inplace=True)
test_df.drop(axis="rows", labels=test_df.index[test_df['card4'].isna()], inplace=True)
test_df.drop(axis="rows", labels=test_df.index[test_df['card5'].isna()], inplace=True)
test_df.drop(axis="rows", labels=test_df.index[test_df['card6'].isna()], inplace=True)
test_df.drop(axis="rows", labels=test_df.index[test_df['D1'].isna()], inplace=True)
test_df.drop(axis="rows", labels=test_df.index[test_df['V95'].isna()], inplace=True)
test_df.drop(axis="rows", labels=test_df.index[test_df['V279'].isna()], inplace=True)
test_df.drop(axis="rows", labels=test_df.index[test_df['V281'].isna()], inplace=True)

"""Removing Columns with more than 70 percent NULL values."""

train_df = train_df.drop(columns =train_df[ list(train_df.loc[:,'V138':'V278']) + ['dist2','R_emaildomain','D12','D13','D14','DeviceType','DeviceInfo'] + list(train_df.loc[:,'D6':'D9']) + list(train_df.loc[:,'V322':'V339']) + list(train_df.loc[:,'id_01':'id_38'])])

test_df = test_df.drop(columns =test_df[ list(test_df.loc[:,'V138':'V278']) + ['dist2','R_emaildomain','D12','D13','D14','DeviceType','DeviceInfo'] + list(test_df.loc[:,'D6':'D9']) + list(test_df.loc[:,'V322':'V339']) + list(test_df.loc[:,'id_01':'id_38'])])

"""Now filling the remaining column NULL values accordingly."""

train_df['dist1'] = train_df['dist1'].fillna(train_df['dist1'].median())
test_df['dist1'] = test_df['dist1'].fillna(test_df['dist1'].median())

train_df['addr1'].fillna(train_df['addr1'].mode().iloc[0], inplace=True)
test_df['addr1'].fillna(test_df['addr1'].mode().iloc[0], inplace=True)

train_df['P_emaildomain'] = train_df['P_emaildomain'].fillna('NA')
test_df['P_emaildomain'] = test_df['P_emaildomain'].fillna('NA')

train_df.loc[:,'D1':'D15'] = train_df.loc[:,'D1':'D15'].fillna(train_df.loc[:,'D1':'D15'].mean())
test_df.loc[:,'D1':'D15'] = test_df.loc[:,'D1':'D15'].fillna(test_df.loc[:,'D1':'D15'].mean())

train_df.loc[:,'V1':'V94'] = train_df.loc[:,'V1':'V94'].fillna(train_df.loc[:,'V1':'V94'].mean())
test_df.loc[:,'V1':'V321'] = test_df.loc[:,'V1':'V321'].fillna(test_df.loc[:,'V1':'V321'].mean())

"""replacing categorical values with numerical values in 'M' features"""

train_df.loc[:,'M1':'M9'] = train_df.loc[:,'M1':'M9'].replace({'T': 0 ,'F': 1})
train_df['M4'] = train_df['M4'].replace({'M0': 0 ,'M1': 1 , 'M2': 2})

test_df.loc[:,'M1':'M9'] = test_df.loc[:,'M1':'M9'].replace({'T': 0 ,'F': 1})
test_df['M4'] = test_df['M4'].replace({'M0': 0 ,'M1': 1 , 'M2': 2})

"""Label Encoding:"""

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

train_df['ProductCD']=label_encoder.fit_transform(train_df['ProductCD'])
train_df['card4']=label_encoder.fit_transform(train_df['card4'])
train_df['card6']=label_encoder.fit_transform(train_df['card6'])
train_df['P_emaildomain']=label_encoder.fit_transform(train_df['P_emaildomain'])

test_df['ProductCD']=label_encoder.fit_transform(test_df['ProductCD'])
test_df['card4']=label_encoder.fit_transform(test_df['card4'])
test_df['card6']=label_encoder.fit_transform(test_df['card6'])
test_df['P_emaildomain']=label_encoder.fit_transform(test_df['P_emaildomain'])

train_df = train_df.drop(columns = ['TransactionID'])
test_df = test_df.drop(columns = ['TransactionID'])

train_df.drop(axis="rows", labels=train_df.index[train_df.duplicated()], inplace=True)
train_df.duplicated().sum()

"""Filling the null in 'M' values randomly with values in same ratio as original."""

train_df['M1'].value_counts(normalize=True)

train_df['M1'] = train_df['M1'].fillna(pd.Series(np.random.choice([0,1], p=[0.99, 0.01], size=len(train_df))))
test_df['M1'] = test_df['M1'].fillna(pd.Series(np.random.choice([0,1], p=[0.99, 0.01], size=len(test_df))))

train_df['M2'].value_counts(normalize=True)

train_df['M2'] = train_df['M2'].fillna(pd.Series(np.random.choice([0,1], p=[0.89, 0.11], size=len(train_df))))
test_df['M2'] = test_df['M2'].fillna(pd.Series(np.random.choice([0,1], p=[0.89, 0.11], size=len(test_df))))

train_df['M3'].value_counts(normalize=True)

train_df['M3'] = train_df['M3'].fillna(pd.Series(np.random.choice([0,1], p=[0.78, 0.22], size=len(train_df))))
test_df['M3'] = test_df['M3'].fillna(pd.Series(np.random.choice([0,1], p=[0.78, 0.22], size=len(test_df))))

train_df['M4'].value_counts(normalize=True)

train_df['M4'] = train_df['M4'].fillna(pd.Series(np.random.choice([0,2,1], p=[0.63, 0.20, 0.17], size=len(train_df))))
test_df['M4'] = test_df['M4'].fillna(pd.Series(np.random.choice([0,2,1], p=[0.63, 0.20, 0.17], size=len(test_df))))

train_df['M5'].value_counts(normalize=True)

train_df['M5'] = train_df['M5'].fillna(pd.Series(np.random.choice([0,1], p=[0.44, 0.56], size=len(train_df))))
test_df['M5'] = test_df['M5'].fillna(pd.Series(np.random.choice([0,1], p=[0.44, 0.56], size=len(test_df))))

train_df['M6'].value_counts(normalize=True)

train_df['M6'] = train_df['M6'].fillna(pd.Series(np.random.choice([0,1], p=[0.45, 0.55], size=len(train_df))))
test_df['M6'] = test_df['M6'].fillna(pd.Series(np.random.choice([0,1], p=[0.45, 0.55], size=len(test_df))))

train_df['M7'].value_counts(normalize=True)

train_df['M7'] = train_df['M7'].fillna(pd.Series(np.random.choice([0,1], p=[0.13, 0.87], size=len(train_df))))
test_df['M7'] = test_df['M7'].fillna(pd.Series(np.random.choice([0,1], p=[0.13, 0.87], size=len(test_df))))

train_df['M8'].value_counts(normalize=True)

train_df['M8'] = train_df['M8'].fillna(pd.Series(np.random.choice([0,1], p=[0.36, 0.64], size=len(train_df))))
test_df['M8'] = test_df['M8'].fillna(pd.Series(np.random.choice([0,1], p=[0.36, 0.64], size=len(test_df))))

train_df['M9'].value_counts(normalize=True)

train_df['M9'] = train_df['M9'].fillna(pd.Series(np.random.choice([0,1], p=[0.84, 0.16], size=len(train_df))))
test_df['M9'] = test_df['M9'].fillna(pd.Series(np.random.choice([0,1], p=[0.84, 0.16], size=len(test_df))))

train_df.loc[:,'M1':'M9'] = train_df.loc[:,'M1':'M9'].fillna(train_df.mode().iloc[0])
test_df.loc[:,'M1':'M9'] = test_df.loc[:,'M1':'M9'].fillna(test_df.mode().iloc[0])

test_df.loc[:,'card2':'card5'] = test_df.loc[:,'card2':'card5'].fillna(test_df.loc[:,'card2':'card5'].mean())

"""Removing columns with variance less than 1 percent"""

threshold = 0.1

train_df=train_df.drop(train_df.std()[train_df.std() < threshold].index.values, axis = 1)

test_df=test_df.drop(test_df.std()[test_df.std() < threshold].index.values, axis = 1)

"""Removing columns with correlation > 0.90

as they are somewhat linearly dependent so they will not bring any new information to the data.
"""

corr_matrix = train_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
train_df.drop(to_drop, axis=1, inplace=True)
test_df.drop(to_drop, axis=1, inplace=True)

train_df.shape

X= train_df.drop(['isFraud'],axis=1)
Y=train_df['isFraud']

train_col = X.columns
test_col = test_df.columns

"""Normalization of data"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
test_df = scaler.transform(test_df)

X = pd.DataFrame(X, columns=train_col)
test_df = pd.DataFrame(test_df, columns=test_col)

fraud = train_df[train_df['isFraud']==1]
normal = train_df[train_df['isFraud']==0]

print(fraud.shape,normal.shape)

"""Removing the data imbalance with Under-sampling"""

from imblearn.under_sampling import RandomUnderSampler

under = RandomUnderSampler()
X_res,y_res=under.fit_resample(X,Y)

from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))

"""# Fitting the model: Applying Xgboost

XGBoost has outperformed every other classification paradigm during our project.
"""

from xgboost import XGBClassifier

# fit model no training data
# these are tuned hyperparameters
model = XGBClassifier( colsample_bytree = 0.4,learning_rate = 0.1,max_depth = 20,min_child_weight = 1,gamma = 0.3)
model.fit(X_res, y_res)

y_pred_xg = model.predict(test_df)

y_xg = pd.DataFrame(y_pred_xg,columns=['isFraud'])

id = []
for i in range(147635):
    id.append(i)
Id = pd.DataFrame(id, columns = ['Id'])

frames = [Id, y_xg]
xg_df = pd.concat(frames, axis=1)

xg_df.to_csv('xg_final.csv',index=False)