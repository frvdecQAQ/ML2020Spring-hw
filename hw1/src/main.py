# -*- coding:utf-8 -*- 
import numpy as np
import pandas as pd

df = pd.read_csv('../hw1/data/train.csv', encoding='big5')
print(df.head())

df = pd.read_csv('../hw1/data/test.csv', encoding='big5')
print(df.head())