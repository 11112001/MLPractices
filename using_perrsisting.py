import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

quiz = int(input('Enter quiz marks: '))
assign = int(input('Enterassignment marks: '))
mid = int(input('Enter mid marks: '))
final = int(input('Enter final marks: '))

#predict from created model
model = joblib.load('Grading.joblib')
prediction = model.predict([[quiz,assign,mid,final]])
print("The grade is : ", prediction)