import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv('titanic-dataset.csv')

# Data preprocessing
df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].median())  # Use fillna with a new DataFrame
df.dropna(inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Split data into features and target variable
X = df.drop(['PassengerId', 'Survived', 'Name'], axis=1)
y = df['Survived']

# Define train_test_split function
def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    mask = np.random.rand(len(X)) < (1 - test_size)
    X_train, X_test = X[mask], X[~mask]
    y_train, y_test = y[mask], y[~mask]
    return X_train, X_test, y_train, y_test

# Define DecisionTreeClassifier class
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, random_state=None):
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)
    
    def build_tree(self, X, y, depth):
        if len(y) == 0:
            return None
        
        if self.max_depth is not None and depth >= self.max_depth:
            return majority_class(y)
        
        if len(np.unique(y)) == 1:
            return y[0]
        else:
            best_split = find_best_split(X, y, random_state=self.random_state)
            if best_split is None:
                return majority_class(y)
            else:
                left_indices = X[:, best_split['feature']] <= best_split['threshold']
                right_indices = ~left_indices
                left_tree = self.build_tree(X[left_indices], y[left_indices], depth+1)
                right_tree = self.build_tree(X[right_indices], y[right_indices], depth+1)
                return {'feature': best_split['feature'], 'threshold': best_split['threshold'],
                        'left': left_tree, 'right': right_tree}
    
    def predict(self, X):
        predictions = []
        for x in X:
            node = self.tree
            while isinstance(node, dict):
                if x[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node)
        return np.array(predictions)

# Define functions for finding best split and majority class
def find_best_split(X, y, random_state=None):
    np.random.seed(random_state)
    n_features = X.shape[1]
    best_split = None
    best_gini = 1.0
    
    for feature in range(n_features):
        unique_values = np.unique(X[:, feature])
        for value in unique_values:
            left_indices = X[:, feature] <= value
            right_indices = ~left_indices
            gini = gini_impurity(y[left_indices], y[right_indices])
            if gini < best_gini:
                best_gini = gini
                best_split = {'feature': feature, 'threshold': value, 'gini': gini}
    
    return best_split if best_split is not None and best_gini < 1.0 else None

def gini_impurity(y_left, y_right):
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    if n_left == 0 or n_right == 0:
        return 0.0
    p_left = np.sum(y_left == majority_class(y_left)) / n_left
    p_right = np.sum(y_right == majority_class(y_right)) / n_right
    return p_left * (1 - p_left) + p_right * (1 - p_right)

def majority_class(y):
    if len(y) == 0:
        return None
    unique, counts = np.unique(y, return_counts=True)
    return unique[np.argmax(counts)]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Build a Decision Tree Classifier model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Evaluate the model
accuracy = np.mean(y_test == y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate confusion matrix
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Generate detailed prediction table
test_passenger_ids = df.iloc[pd.Series(y_test).index]['PassengerId'].values
test_names = df.iloc[pd.Series(y_test).index]['Name'].values
test_ages = df.iloc[pd.Series(y_test).index]['Age'].values
test_sex = df.iloc[pd.Series(y_test).index]['Sex_male'].values
test_embarked_Q = df.iloc[pd.Series(y_test).index]['Embarked_Q'].values
test_embarked_S = df.iloc[pd.Series(y_test).index]['Embarked_S'].values
prediction_table = pd.DataFrame({'PassengerId': test_passenger_ids, 'Name': test_names, 'Age': test_ages,
                                 'Sex_male': test_sex, 'Embarked_Q': test_embarked_Q,
                                 'Embarked_S': test_embarked_S, 'Predicted Survived': y_pred,
                                 'Actual Survived': y_test})
print(prediction_table)