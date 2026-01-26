<H3>THIRUMALAI K</H3>
<H3>212224240176</H3>
<H3>EX. NO.1</H3>
<H3>26/1/2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#Load dataset
data = pd.read_csv("Churn_Modelling.csv")
print("First 5 rows:\n", data.head())

#Explore dataset
print("\nDataset Info:\n")
print(data.info())

print("\nMissing Values:\n")
print(data.isnull().sum())

print("\nStatistical Summary:\n")
print(data.describe())

#Drop irrelevant columns
# RowNumber, CustomerId, and Surname don't help prediction
data = data.drop(['RowNumber','CustomerId','Surname'], axis=1)

#Encode categorical variables (Geography, Gender)
label = LabelEncoder()
data['Geography'] = label.fit_transform(data['Geography'])
data['Gender'] = label.fit_transform(data['Gender'])

print("\nAfter Encoding:\n", data.head())

#Separate features and target
X = data.drop('Exited', axis=1).values   # features
y = data['Exited'].values                # target

#Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("\nNormalized Features (first 5 rows):\n", X_scaled[:5])

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

```


## OUTPUT:
<img width="1042" height="522" alt="image" src="https://github.com/user-attachments/assets/05206fdd-0804-451f-bcde-5b8f5ea70b7b" />
<img width="470" height="591" alt="image" src="https://github.com/user-attachments/assets/ea30865a-4e50-4b7e-b5f6-1448408ffd9e" />
<img width="287" height="425" alt="image" src="https://github.com/user-attachments/assets/040ce13f-ff70-4704-82b4-1352a8d11fed" />
<img width="1033" height="720" alt="image" src="https://github.com/user-attachments/assets/fd04e0c5-c22b-4bb7-b0ae-13ba875857bb" />
<img width="937" height="703" alt="image" src="https://github.com/user-attachments/assets/55452c42-243f-4ff3-bf14-9230218902c7" />







## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


