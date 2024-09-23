# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries.

2. Upload the dataset and check for any null values using .isnull() function.

3. Import LabelEncoder and encode the dataset.

4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5. Predict the values of arrays.

6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7. Predict the values of array.

8. Apply to new unknown values.

## Program:
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
```
Developed by: HARSHITHA V
RegisterNumber: 212223230074
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv(r"C:\Users\admin\Downloads\Salary.csv")

print(data.head())
print(data.info())
print("\nMissing values in each column:\n", data.isnull().sum())

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

print("\nEncoded data:\n", data.head())

X = data[["Position", "Level"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

predicted_salary = dt.predict([[5, 6]])
print(f"\nPredicted Salary for Position 5 and Level 6: {predicted_salary[0]:.2f}")

plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=X.columns, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/2ab3c067-4760-43cb-ac9d-88394c13fdfe)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
