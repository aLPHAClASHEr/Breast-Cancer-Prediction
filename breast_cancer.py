import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


# Data collection and Processing

# Loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
#print(breast_cancer_dataset)

# Loading the data to a pandas dataframe
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
# Printing the first 5 rows of the dataframe
#print(data_frame.head(3))
#print(data_frame.shape)

# Adding the 'target' column to the dataframe
data_frame['label'] = breast_cancer_dataset.target
print(data_frame.head(12))

#print(data_frame.shape)
#print(data_frame.info())  # Also checks for numm values in the dataset
#print(data_frame.describe())

#print(data_frame['label'].value_counts())

# This command brings the mean values for all the columns for the 2 values present in the 'label' column
#print(data_frame.groupby('label').mean())

# Separating the features and targets
X = data_frame.drop(columns = 'label', axis = 1)  # The .drop means all the columns except 'label' are stored in X variable
Y = data_frame['label']
#print(X)

# Splitting data into training and testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Model training
model = LogisticRegression(max_iter=1846)
model.fit(X_train, Y_train)

# Model Evaluation using accuracy check
X_train_prediction = model.predict(X_train)
training_data_score = accuracy_score(Y_train, X_train_prediction)
print('Training Data Prediction Accuracy:', training_data_score*100)

X_test_prediction = model.predict(X_test)
testing_data_score = accuracy_score(Y_test, X_test_prediction)
print('Testing Data Prediction Accuracy:', testing_data_score*100)


# Building model predictor algorithm

#input_data = input('Enter the Data: ')
#input_data = tuple(float(a) for a in input_data.split(","))
#print(input_data)

#new_array = np.asarray(input_data)
#input_data_reshaped = new_array.reshape(1, -1)

#prediction = model.predict(input_data_reshaped)
#print(prediction)

#if prediction[0] == 0:
    #print('The Breast Cancer is Malignant')
#else:
    #print('The Breast Cancer is Benign')
    
    
    
# Saving the model to a pickle file
filename = 'trained_model.pkl'
pickle.dump(model, open(filename, 'wb'))

# Loading the saved model
loaded_model = pickle.load(open(filename, 'rb'))
X_test_predict = loaded_model.predict(X_test)
testing_data = accuracy_score(Y_test, X_test_predict)
print('Testing Data Prediction Accuracy:', testing_data*100)



#input_data = input('Enter the Data: ')
#input_data = tuple(float(a) for a in input_data.split(","))
#print(input_data)

#new_array = np.asarray(input_data)
#input_data_reshaped = new_array.reshape(1, -1)

#prediction = loaded_model.predict(input_data_reshaped)
#print(prediction)

#if prediction[0] == 0:
 #   print('The Breast Cancer is Malignant')
#else:
 #   print('The Breast Cancer is Benign')
