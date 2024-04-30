# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:40:15 2024

@author: Samson
"""

# Importing the libraries
import pandas as pd
import pickle

dataset = pd.read_csv('HR_Recruitment_Data.csv')


X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('HR_Recruitment.pkl','wb'))

# Loading model to compare the results
HR_Recruitment = pickle.load(open('HR_Recruitment.pkl','rb'))
