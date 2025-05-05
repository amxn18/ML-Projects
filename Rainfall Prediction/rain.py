import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('Rainfall Prediction/Rainfall.csv')
df.columns = df.columns.str.strip()  # removes leading/trailing spaces
# print(df.isnull().sum())
df['winddirection'] = df['winddirection'].fillna(df['winddirection'].mode()[0])
df['windspeed'] = df['windspeed'].fillna(df['windspeed'].median())
# print(df.isnull().sum())

# print(df['rainfall'].value_counts()) 
encoder = LabelEncoder()
df['rainfall'] = encoder.fit_transform(df['rainfall'])
# print(df['rainfall'].value_counts())

yes = df[df['rainfall'] == 1]
no = df[df['rainfall'] == 0]
noResampled = resample(no,
                      replace=True,  
                      n_samples=len(yes),  
                      random_state=123)  

balancedDf = pd.concat([yes, noResampled])
# print(balancedDf['rainfall'].value_counts())

# print(balancedDf.head())    
x = balancedDf.drop(['rainfall', 'day'], axis=1)
# print(x.head())
y = balancedDf['rainfall']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify= y)
scalar = StandardScaler()
x_trainScaled = scalar.fit_transform(x_train)
x_testScaled = scalar.transform(x_test)

model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, x_trainScaled, y_train, cv=5, scoring='accuracy')
# print("Cross-validation scores:", scores)
print("Average CV accuracy:", scores.mean())

model.fit(x_trainScaled, y_train)
YPredictionTraining = model.predict(x_trainScaled)
YPredictionTesting = model.predict(x_testScaled)
print("Training accuracy:", accuracy_score(y_train, YPredictionTraining))           
print("Testing accuracy:", accuracy_score(y_test, YPredictionTesting))
# print("Confusion Matrix:\n", confusion_matrix(y_test, YPredictionTesting))
# print("Classification Report:\n", classification_report(y_test, YPredictionTesting))


# Sample input data (replace values as needed)
input_data = {
    'pressure': 1010,       # Replace with appropriate value
    'maxtemp': 30,          # Replace with appropriate value
    'temparature': 30,      # Replace with appropriate value
    'mintemp': 18,          # Replace with appropriate value
    'dewpoint': 15,         # Replace with appropriate value
    'humidity': 70,         # Replace with appropriate value
    'cloud': 40,            # Replace with appropriate value
    'sunshine': 5,          # Replace with appropriate value
    'winddirection': 180,   # Replace with appropriate value
    'windspeed': 12         # Replace with appropriate value
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Scale the input using the same StandardScaler used for training
input_scaled = scalar.transform(input_df)

# Make prediction using the trained model
prediction = model.predict(input_scaled)

# Display result
if prediction[0] == 1:
    print("Prediction: Rainfall")
else:
    print("Prediction: No Rainfall")

# Add Values To The Input DataFrame