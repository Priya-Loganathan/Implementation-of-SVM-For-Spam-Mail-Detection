# Implementation-of-SVM-For-Spam-Mail-Detection

## Aim:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the necessary packages.

2.Read the given csv file and display the few contents of the data.

3.Assign the features for x and y respectively.

4.Split the x and y sets into train and test sets.

5.Convert the Alphabetical data to numeric using CountVectorizer.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: DELLI PRIYA L
RegisterNumber: 212222230029
*/
```
```
import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
```
```
import pandas as pd
data= pd.read_csv("/content/spam.csv",encoding='Windows-1252')
```
```
data.head()
```
```
data.info()
```
```
x=data["v1"].values
```
```
y=data["v2"].values
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```
```
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

### Result output
![image](https://github.com/Priya-Loganathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121166075/aaa8e951-a92a-40f4-8e7e-a8d9dddaa9c1)
### data.head()
![image](https://github.com/Priya-Loganathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121166075/30bda476-48ec-4f39-96fb-347e938e0365)
### data.info()
![image](https://github.com/Priya-Loganathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121166075/1fa10c39-8276-4d93-8452-6cb0049d862b)
### data.isnull().sum()
![image](https://github.com/Priya-Loganathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121166075/a619d3e5-5c55-4334-981d-81df1a51d3fc)
### Y_prediction value
![image](https://github.com/Priya-Loganathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121166075/9a4393cf-f679-4d9e-b798-bb3900a4440c)
### Accuracy value
![image](https://github.com/Priya-Loganathan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121166075/7df9b110-0068-449a-9c9f-28e24816adaf)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
