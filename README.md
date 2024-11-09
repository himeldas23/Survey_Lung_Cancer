# Lung Cancer Prediction Model

This project applies machine learning techniques to predict the likelihood of lung cancer based on survey data. The model aims to assist in early cancer detection by analyzing key factors related to lung cancer risk.


## Overview
Lung cancer is a significant health issue globally, and early detection is crucial for effective treatment. This model leverages data science methods to predict lung cancer likelihood with an accuracy of 90.32% using the survey_lung_cancer dataset.

To run this notebook, ensure you have the following libraries installed:

- `pandas` - For data loading and manipulation
- `numpy` - For numerical operations
- `scikit-learn` - For machine learning tasks
## Key Features
Accuracy: Achieves an accuracy of 90.32% on the survey lung cancer dataset.
Technologies Used: Python, Scikit-Learn, Pandas, and other data science libraries.

## Dataset
The dataset includes a variety of features related to lung cancer risk factors:

- GENDER: Gender of the individual (encoded from string to - -  numerical values)
- AGE: Age of the individual
- SMOKING: Smoking habits
- YELLOW_FINGERS: Presence of yellow fingers (a potential smoking-related sign)
- ANXIETY: Anxiety levels
- PEER_PRESSURE: Influence of peer pressure
- CHRONIC DISEASE: Presence of chronic diseases
- FATIGUE: Level of fatigue
- ALLERGY: History of allergies
- WHEEZING: Wheezing symptoms
- ALCOHOL CONSUMING: Alcohol consumption habits
- COUGHING: Frequency of coughing
- SHORTNESS OF BREATH: Experience of shortness of breath
- SWALLOWING DIFFICULTY: Difficulty in swallowing
- CHEST PAIN: Experience of chest pain
- LUNG_CANCER: Target variable indicating the presence of lung cancer



## Analysis Steps
**Data Preparation:**
- Load the dataset using Pandas.
- Check dataset shape and structure.
```bash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
```
```bash
df=pd.read_csv('/content/survey lung cancer.csv')
df
df.shape
```
| GENDER | AGE | SMOKING | YELLOW_FINGERS | ANXIETY | PEER_PRESSURE | CHRONIC_DISEASE | FATIGUE | ALLERGY | WHEEZING | ALCOHOL_CONSUMING | COUGHING | SHORTNESS_OF_BREATH | SWALLOWING_DIFFICULTY | CHEST_PAIN | LUNG_CANCER |
|--------|-----|---------|----------------|---------|---------------|-----------------|---------|---------|----------|-------------------|----------|---------------------|-----------------------|------------|-------------|
| M      | 69  | 1       | 2              | 2       | 1             | 1               | 2       | 1       | 2        | 2                 | 2        | 2                   | 2                     | 2          | YES         |
| M      | 74  | 2       | 1              | 1       | 1             | 2               | 2       | 2       | 1        | 1                 | 1        | 2                   | 2                     | 2          | YES         |
| F      | 59  | 1       | 1              | 1       | 2             | 1               | 2       | 1       | 2        | 1                 | 2        | 2                   | 1                     | 2          | NO          |
| M      | 63  | 2       | 2              | 2       | 1             | 1               | 1       | 1       | 1        | 2                 | 1        | 1                   | 2                     | 2          | NO          |
| F      | 63  | 1       | 2              | 1       | 1             | 1               | 1       | 1       | 2        | 1                 | 2        | 2                   | 1                     | 1          | NO          |
| ...    | ... | ...     | ...            | ...     | ...           | ...             | ...     | ...     | ...      | ...               | ...      | ...                 | ...                   | ...        | ...         |
| F      | 56  | 1       | 1              | 1       | 2             | 2               | 2       | 1       | 1        | 2                 | 2        | 2                   | 2                     | 1          | YES         |
| M      | 70  | 2       | 1              | 1       | 1             | 1               | 2       | 2       | 2        | 2                 | 2        | 2                   | 1                     | 2          | YES         |
| M      | 58  | 2       | 1              | 1       | 1             | 1               | 1       | 2       | 2        | 2                 | 2        | 1                   | 1                     | 2          | YES         |
| M      | 67  | 2       | 1              | 2       | 1             | 1               | 2       | 2       | 1        | 2                 | 2        | 2                   | 1                     | 2          | YES         |
| M      | 62  | 1       | 1              | 1       | 2             | 1               | 2       | 2       | 2        | 2                 | 1        | 1                   | 2                     | 1          | YES         |
309 rows × 16 columns


**Split X and Y data**

```bash
X=df.iloc[:,0:15]
Y=df.iloc[:,-1]
```
**Target variable**

![Target variable](Images/target%20variable.png)

**Preprocessing:**
- Label encode the Gender column.
- Check for and handle any missing values.


```bash
# Convert categorical Gender column to numerical using Label Encoding
label_encoder = LabelEncoder()
X['GENDER'] = label_encoder.fit_transform(X['GENDER'])
```
```bash
# Check for any remaining non-numeric values or NaN
print(X.isnull().sum())  # Check for missing values
print(X.dtypes)  # Check data types again
```
**Modeling:**
- Split the data into training and test sets.
- Train a logistic regression model on the training data.
```bash
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
```
```bash
model = LogisticRegression()
model.fit(X_train, Y_train)
```
**Make predictions**
```bash
Y_pred = model.predict(X_test)
```
**Evaluate the model**
```bash
print("Accuracy:", accuracy_score(Y_test, Y_pred))

```


## How to Use
1. Clone this repository:
```bash
https://github.com/himeldas23/Survey_Lung_Cancer.git
```
2. Navigate to the project directory:
```bash
cd Survey_Lung_Cancer
```
3. Install Dependencies:
```bash
pip install -r requirements.txt
```
4. Run the Notebook or Script:\
Open the survey_lung_cancer.ipynb notebook and run each cell to load the dataset and execute the analysis steps.
## Future Improvements
- Experiment with additional algorithms to potentially improve accuracy.
- Explore new features or external datasets to expand predictive power.
- Integrate visualizations to better understand feature importance.
## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.

