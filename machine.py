import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#import data from excel file
grade_data = pd.read_excel('Data.xlsx')
#learning with model
X = grade_data.drop(columns=['SNAMES ','Total Marks','Marks /20','Grading '])
Y = grade_data['Grading ']
model = DecisionTreeClassifier()
model.fit(X.values,Y)
#predict with the model
prediction = model.predict([[6,6,18,30]])
print(prediction)

