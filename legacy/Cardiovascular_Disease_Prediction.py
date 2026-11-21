import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
data=pd.read_csv('/content/heart.csv')
data.info()
new_var = print(data.describe())
new_var
print(data.isnull().sum())
print(data['target'].value_counts())
sns.countplot(x='target', data=data)
plt.title('Distribution of Target Variable')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()
data.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()
sns.boxplot(x='target', y='age', data=data)
plt.title('Age vs. Heart Disease')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()
sns.countplot(x='sex', hue='target', data=data)
plt.title('Sex vs. Heart Disease')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Count')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()
categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal', 'sex', 'exang', 'fbs']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
print("Categorical features one-hot encoded.")
print(data.head())
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

scaler = StandardScaler()

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("Numerical features scaled.")
print(X_train.head())
print(X_test.head())
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

#base models
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
adaboost_model = AdaBoostClassifier(n_estimators=100, random_state=42)


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression # Using Logistic Regression as the meta-model here

# Define base models (can be the same or different from Voting)
estimators = [
    ('dt', dt_model),
    ('gb', gb_model),
    ('ada', adaboost_model)
]

# Initialize the Stacking Classifier
# final_estimator is the meta-model
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# Train the Stacking Classifier
stacking_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_stacking = stacking_model.predict(X_test)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f"Stacking Classifier Accuracy: {accuracy_stacking:.4f}")
print(classification_report(y_test, y_pred_stacking))

# Plot Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_stacking), annot=True, fmt='d', cmap='Blues')
plt.title('Stacking Classifier Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
