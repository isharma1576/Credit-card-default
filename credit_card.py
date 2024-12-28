import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cc_db = pd.read_csv(r'/home/ishan-sharma/work/projects/Credit-card-default/Credit Card Client dataset/UCI_Credit_Card.csv')
# print(cc_db)

cc_db.rename(columns={
    'ID':'id',
    'LIMIT_BAL':'limit_bal',
    'SEX':'sex',
    'EDUCATION':'education',
    'MARRIAGE':'marriage',
    'AGE':'age',
    'PAY_0':'pay_1',
    'PAY_2':'pay_2',
    'PAY_3':'pay_3',
    'PAY_4':'pay_4',
    'PAY_5':'pay_5',
    'PAY_6':'pay_6',
    'BILL_AMT1':'bill_1',
    'BILL_AMT2':'bill_2',
    'BILL_AMT3':'bill_3',
    'BILL_AMT4':'bill_4',
    'BILL_AMT5':'bill_5',
    'BILL_AMT6':'bill_6',
    'PAY_AMT1':'pay_amt1',
    'PAY_AMT2':'pay_amt2',
    'PAY_AMT3':'pay_amt3',
    'PAY_AMT4':'pay_amt4',
    'PAY_AMT5':'pay_amt5',
    'PAY_AMT6':'pay_amt6',
    'default.payment.next.month': 'default.payment.next.month'},
    inplace = True)
print('credit card columns :', cc_db.columns)

# print('limit balance:', cc_db['limit_bal'])

fig = plt.figure(figsize =(10, 7))
# plt.boxplot(cc_db['limit_bal'])
# plt.show()

# cleaning balance limit
# print('balance limit percentile:',cc_db['limit_bal'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]))

limit_median = cc_db['limit_bal'].median()
limit_temp = cc_db['limit_bal'] > 500000

cc_db['new_limit'] = cc_db['limit_bal']
cc_db['new_limit'][limit_temp] = limit_median

# print('new balance limit percentile:', cc_db['new_limit'].quantile([0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])

# print("working upto here-------------------")