import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_df = pd.read_csv('Train_data.csv')
test_df = pd.read_csv('Test_data.csv')

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :]

counts = X_train['service'].value_counts()

print(X_test[X_test['service'] == 'tftp_u'].index.values)



X_test.drop([12163], inplace = True, axis = 0)


from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()


X_train.iloc[:, 1] = le1.fit_transform(X_train.iloc[:, 1])
X_test.iloc[:, 1] = le1.transform(X_test.iloc[:, 1])

X_train.iloc[:, 2] = le2.fit_transform(X_train.iloc[:, 2])
X_test.iloc[:, 2] = le2.transform(X_test.iloc[:, 2])

X_train.iloc[:, 3] = le3.fit_transform(X_train.iloc[:, 3])
X_test.iloc[:, 3] = le3.transform(X_test.iloc[:, 3])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1, 2, 3])], remainder='passthrough')

X_train = columnTransformer.fit_transform(X_train)
X_test = columnTransformer.transform(X_test)

le_y = LabelEncoder()

y_train = le_y.fit_transform(y_train)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)


from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()


classifier.add(Dense(input_dim = 118, activation = 'relu', output_dim = 128))

classifier.add(Dense(output_dim = 256, activation = 'relu'))

classifier.add(Dense(output_dim = 256, activation = 'relu'))

classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()


history = classifier.fit(X_train, y_train, validation_data = (X_val, y_val),batch_size = 25, epochs = 20)


y_pred = classifier.predict(X_test)


for i in range(len(y_pred)):
    if y_pred[i] < 0.5:
        y_pred[i] = 0
    
    else:
        y_pred[i] = 1
    
