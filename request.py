# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 07:09:05 2024

@author: Hp
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'experience':2, 'assessment_score':9, 'interview_score':6})

print(r.json())