import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
#import data from excel file
grade_data = pd.read_excel('Data.xlsx')
#learning with model
X = grade_data.drop(columns=['SNAMES ','Total Marks','Marks /20','Grading '])
Y = grade_data['Grading ']
model = LogisticRegression(solver='lbfgs',max_iter=10000)
model.fit(X.values,Y)

#create persisting model
joblib.dump(model,'Grading.joblib')

