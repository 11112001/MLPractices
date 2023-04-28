import pandas as pd
#algorithim models to be used to find the best accurate one
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
#used to split datasets
from sklearn.model_selection import train_test_split
#used to find which model has scored higher
from sklearn.metrics import accuracy_score


#import dataset
grade_data = pd.read_excel('Data.xlsx')

#split dataet into training set and test set
X = grade_data.drop(columns=['SNAMES ','Total Marks','Marks /20','Grading '])
Y = grade_data['Grading ']
X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size = 0.2)

#creating models
decision_model = DecisionTreeClassifier()
logistic_model = LogisticRegression(solver='lbfgs',max_iter=10000)
svm_model = svm.SVC(kernel='linear')
rf_model = RandomForestClassifier(n_estimators=100)

#train models
decision_model.fit(X_train,Y_train)
logistic_model.fit(X_train,Y_train)
svm_model.fit(X_train,Y_train)
rf_model.fit(X_train,Y_train)

#predict the model
decision_prediction = decision_model.predict(x_test)
logistic_prediction = logistic_model.predict(x_test)
svm_prediction = svm_model.predict(x_test)
rf_prediction = rf_model.predict(x_test)


#calcute.predictmodel accurancy
decision_score = accuracy_score(y_test,decision_prediction)
logistic_score = accuracy_score(y_test,logistic_prediction)
svm_score = accuracy_score(y_test,svm_prediction)
rf_score = accuracy_score(y_test,rf_prediction)

#display accurancy
print('Decision = ',decision_score*100,'%' )
print('Logistics = ',logistic_score*100,'%')
print('svm = ', svm_score*100,'%')
print('rf = ',rf_score*100,'%')