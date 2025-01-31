import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')
# Path to the ZIP file
zip_file_path = r"C:\Users\khans\OneDrive\Desktop\Spring 2024\AI\archive (9).zip"


# Function to read and print the contents of the ZIP file
def print_zip_contents(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Print the list of files in the ZIP file
        print("Files in ZIP file:")
        for file_name in zip_ref.namelist():
            print(file_name)
        # Extract and print the content of each file in the ZIP file (optional)
        print("\nContents of the file:")
        for file_name in zip_ref.namelist():
            with zip_ref.open(file_name) as file:
                print(file_name + ":")
                print(file.read().decode('utf-8'))  # Assumes text files, adjust as needed


# Call the function with the specified ZIP file path
print_zip_contents(zip_file_path)

#Random Forest
df = pd.read_csv(r"C:\Users\khans\OneDrive\Desktop\Spring 2024\AI\archive (9).zip")
print(df)

df.info()
df.head()
df.tail()
df.nunique()

# Assuming df is your DataFrame
X = df.iloc[:, 1:-1].values  # Features
y = df.iloc[:, -1].values  # Target variable

#Check for and handle categorical variables
label_encoder = LabelEncoder()
x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical = df.select_dtypes(exclude=['object']).values
x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values

# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

# Fit the regressor with x and y data
regressor.fit(x, y)

# Access the OOB Score
oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

# Making predictions on the same data or new data
predictions = regressor.predict(x)

# Evaluating the model
mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y, predictions)
print(f'R-squared: {r2}')

np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Feature values
y = 3 * X.squeeze() + np.random.randn(100) * 3  # Target values (with some noise)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X, y)

X_grid = np.linspace(0, 10, 1000)[:, np.newaxis]

y_pred = rf_regressor.predict(X_grid)

plt.scatter(X, y, color='blue', label='Real Data')

plt.plot(X_grid, y_pred, color='green', label='Predicted Values')

plt.xlabel('Feature Values')
plt.ylabel('Performance')
plt.title('RandomForest Regression Results')
plt.legend()
plt.grid(True)
plt.show()

# Assuming regressor is your trained Random Forest model
# Pick one tree from the forest, e.g., the first tree (index 0)
tree_to_plot = regressor.estimators_[0]

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree_to_plot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest")
plt.show()

# Load your dataset (replace 'X' and 'y' with your features and target variable)
# X, y = load_data()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the GBM model
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
gbm.fit(X_train, y_train)

# Make predictions
y_pred = gbm.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Convert the target variable to binary (1 for diabetes, 0 for no diabetes)
y_binary = (y > np.median(y)).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the decision boundary with accuracy information
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 0], hue=y_test, palette={0: 'blue', 1: 'red'}, marker='o')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression Decision Boundary\nAccuracy: {:.2f}%".format(
    accuracy * 100))
plt.show()

# Plot ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('Features')
plt.ylabel('Performance')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy: {:.2f}%'.format(
    accuracy * 100))
plt.legend(loc="lower right")
plt.show()

# importing scikit learn with make_blobs
from sklearn.datasets import make_blobs

# creating datasets X containing n_samples
# Y containing two classes
X, Y = make_blobs(n_samples=500, centers=2,
                  random_state=0, cluster_std=0.40)

import matplotlib.pyplot as plt

# plotting scatters
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring');
plt.show()

# creating linspace between -1 to 3.5
xfit = np.linspace(-1, 3.5)

# plotting scatter
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')

# plot a line between the different sets of data
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5);
plt.show()

# KNN algorithm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the data
file_path = r'C:\Users\khans\OneDrive\Desktop\Spring 2024\AI\archive (9).zip'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Step 2: Preprocess the data
# Select the relevant features and target variable
# Let's assume we are predicting whether a team made it to the Round of 16 (binary classification)

# Features
features = ['Year', 'G', 'W', 'L', 'W-L%', 'SRS', 'SOS', 'Conf. W', 'Conf. L', 'Home W',
            'Home L', 'Away W', 'Away L', 'Team Points', 'Opp Points', 'FG%', '3P%', 'FT%',
            'Home win rate', 'Away win rate', 'Conference win rate', 'Point diff %', 'AdjEM',
            'AdjO', 'AdjD', 'AdjT', 'Luck', 'SOS AdjEM', 'OppO', 'OppD', 'NCSOS AdjEM', 'Seed']

# Target variable
target = 'Made Round of 16'

# Extract features and target from the dataset
X = data[features]
y = data[target]

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # You can choose the number of neighbors
knn.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = knn.predict(X_test)

# Step 7: Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))

# Optional: Tune the number of neighbors (k) using cross-validation
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': range(1, 21)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print('Best number of neighbors:', grid_search.best_params_)
print('Best cross-validation score:', grid_search.best_score_)