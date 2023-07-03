from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
file1 = pd.read_csv('testmo.csv', encoding='utf-8')
file2 = pd.read_csv('testnham.csv', encoding='utf-8')

X = pd.concat([file1, file2])
y = pd.concat([pd.Series([0]*len(file1)), pd.Series([1]*len(file2))]) # 0 la co tinh tao, 1 la buon ngu
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = svm.SVC(kernel="linear")
model.fit(X_train.values, y_train.values)

filename = 'test.h5'
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

train_score = model.score(X_train, y_train)
score = model.score(X_test, y_test)
print("Train score: ", train_score)
print("Test score: ",score)
print(type(X_test))
print(model.predict(X_test))
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score
#
# # Define number of folds
# k = 5
#
# # Initialize KFold object
# kf = KFold(n_splits=k)
#
# # Initialize list to store accuracy scores
# accuracy_list = []
#
# # Loop through each fold
# for train_index, test_index in kf.split(X):
#     # Split data into training and testing sets
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     # Initialize model
#
#     # Fit the model on the training set
#     model.fit(X_train, y_train)
#
#     # Make predictions on the testing set
#     y_pred = model.predict(X_test)
#
#     # Calculate accuracy score
#     accuracy = accuracy_score(y_test, y_pred)
#
#     # Append accuracy score to the list
#     accuracy_list.append(accuracy)
#
# # Calculate average accuracy score
# avg_accuracy = np.mean(accuracy_list)
#
#("Average Accuracy Score:", avg_accuracy)