<H3>THIRUMALAI K</H3>
<H3>212224240176</H3>
<H3>EX. NO.1</H3>
<H3>27/1/2026</H3>
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
#import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


#Read the dataset from drive
df = pd.read_csv("Churn_Modelling.csv")   
print(df.head())


# Finding Missing Values
print(df.isnull().sum())


#Handling Missing values
df.fillna(df.mean(numeric_only=True), inplace=True)
print(df.isnull().sum())


#Check for Duplicates
print("Duplicates:", df.duplicated().sum())


#Detect Outliers (simple method using describe)
print(df.describe())


#Normalize the dataset (only numeric columns)
num_df = df.select_dtypes(include='number')
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(num_df), columns=num_df.columns)
print(df_norm.head())


#split the dataset into input and output
X = df_norm.iloc[:, :-1].values
Y = df_norm.iloc[:, -1].values


#splitting the data for training & Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#Print the training data and testing data
print("Training Data:\n", X_train[:5])
print("Testing Data:\n", X_test[:5])


```


## OUTPUT:
<img width="822" height="292" alt="image" src="https://github.com/user-attachments/assets/5b84346a-f50f-42db-86f9-2b0f0262c078" />
<img width="476" height="386" alt="image" src="https://github.com/user-attachments/assets/7988f22e-c9fe-4717-86b9-28feb494092a" /><img width="331" height="327" alt="image" src="https://github.com/user-attachments/assets/a917609f-7feb-4c72-8078-50c312604d42" />
<img width="782" height="353" alt="image" src="https://github.com/user-attachments/assets/fa8a1bea-a9e0-4f58-8a5a-0289902ae696" />
<img width="623" height="302" alt="image" src="https://github.com/user-attachments/assets/d1946c03-b475-40de-a68f-3acebdbce2a8" />
<img width="681" height="212" alt="image" src="https://github.com/user-attachments/assets/f460251a-5a4f-44a8-9e7a-3bc4d6bf3763" />
<img width="662" height="216" alt="image" src="https://github.com/user-attachments/assets/ae2f0707-d792-47a5-9e0d-8a63d16fb12b" />









## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


