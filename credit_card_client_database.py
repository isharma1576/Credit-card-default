# importing all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cc_db = pd.read_csv('/home/ishan-sharma/work/projects/Credit-card-default/Credit Card Client dataset/UCI_Credit_Card.csv')

# credit_card.shape

# credit_card.info()

# credit_card.head(10)

# Outlier in Balance Limit

plt.boxplot(credit_card['LIMIT_BAL'])

Limit_percentile = credit_card['LIMIT_BAL'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])
Limit_percentile

# cleaning Balance Limit

limit_median = credit_card['LIMIT_BAL'].median()

limit_temp = credit_card['LIMIT_BAL'] > 500000

credit_card['new_limit'] = credit_card['LIMIT_BAL']
credit_card['new_limit'][limit_temp] = limit_median

Limit_percentile_1 = credit_card['new_limit'].quantile([0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])
Limit_percentile_1

# Outier Detection In Age

plt.boxplot(credit_card["AGE"])

print(credit_card['AGE'].value_counts())
sns.countplot(y='AGE',  data=credit_card)

age_median = credit_card['AGE'].median()
age_median

age_temp = credit_card['AGE'] > 70

credit_card['New_age'] = credit_card['AGE']
credit_card['New_age'][age_temp] = age_median

credit_card['New_age'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])

# Outlier in Pay_0(after 6 months of non payments consumer is considered to be defaulter)

print(credit_card['PAY_0'].value_counts())
sns.countplot(y='PAY_0', data=credit_card)

pay_0_crosstab_target = pd.crosstab(credit_card['PAY_0'], credit_card['default.payment.next.month'])
pay_0_crosstab_target

pay_0_crosstab_target_percent = pay_0_crosstab_target.apply(lambda x: x/x.sum(), axis=1)
round(pay_0_crosstab_target_percent, 2)

credit_card['Pay_0_new'] = credit_card['PAY_0']
credit_card['Pay_0_new'][credit_card['PAY_0'] > 6] = 4
credit_card['Pay_0_new'].value_counts()

# Outlier in Pay_2

print(credit_card['PAY_2'].value_counts())
sns.countplot(y='PAY_2', data=credit_card)

pay_2_crosstab_target = pd.crosstab(credit_card['PAY_2'], credit_card['default.payment.next.month'])
pay_2_crosstab_target_percent = pay_2_crosstab_target.apply(lambda x: x/x.sum(), axis = 1)
round(pay_2_crosstab_target_percent, 2)

credit_card['Pay_2_new'] = credit_card['PAY_2']
credit_card['Pay_2_new'][credit_card['PAY_2'] > 6] = 2
credit_card['Pay_2_new'].value_counts()

# Outlier in Pay_3

print(credit_card['PAY_3'].value_counts())
sns.countplot(y='PAY_3', data=credit_card)

pay_3_crosstab_target = pd.crosstab(credit_card['PAY_3'], credit_card['default.payment.next.month'])
pay_3_crosstab_target_percent = pay_3_crosstab_target.apply(lambda x: x/x.sum(), axis = 1)
round(pay_3_crosstab_target_percent, 2)

credit_card['Pay_3_new'] = credit_card['PAY_3']
credit_card['Pay_3_new'][credit_card['PAY_3'] > 6] = 6
credit_card['Pay_3_new'].value_counts()

# Outlier in Pay_4

print(credit_card['PAY_4'].value_counts())
sns.countplot(y='PAY_4', data=credit_card)

pay_4_crosstab_target = pd.crosstab(credit_card['PAY_4'], credit_card['default.payment.next.month'])
pay_4_crosstab_target_percent = pay_4_crosstab_target.apply(lambda x: x/x.sum(), axis = 1)
round(pay_4_crosstab_target_percent, 2)

credit_card['pay_4_new'] = credit_card['PAY_4']
credit_card['pay_4_new'][credit_card['PAY_4'] > 6] = 4
credit_card['pay_4_new'].value_counts()

# Outlier in Pay_5

print(credit_card['PAY_5'].value_counts())
sns.countplot(y='PAY_5', data=credit_card)

pay_5_crosstab_target = pd.crosstab(credit_card['PAY_5'], credit_card['default.payment.next.month'])
pay_5_crosstab_target_percent = pay_5_crosstab_target.apply(lambda x: x/x.sum(), axis = 1)
round(pay_5_crosstab_target_percent, 2)

credit_card['pay_5_new'] = credit_card['PAY_5']
credit_card['pay_5_new'][credit_card['PAY_5'] > 6] = 6
credit_card['pay_5_new'].value_counts()

# Outlier in Pay_6

print(credit_card['PAY_6'].value_counts())
sns.countplot(y='PAY_6', data=credit_card)

pay_6_crosstab_target = pd.crosstab(credit_card['PAY_6'], credit_card['default.payment.next.month'])
pay_6_crosstab_target_percent = pay_6_crosstab_target.apply(lambda x: x/x.sum(), axis = 1)
round(pay_6_crosstab_target_percent, 2)

credit_card['pay_6_new'] = credit_card['PAY_6']
credit_card['pay_6_new'][credit_card['PAY_6'] > 6] = 6
credit_card['pay_6_new'].value_counts()

# Outlier in Bill_amount 1

credit_card['BILL_AMT1'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]) #99% good data

bill_amt1_median = credit_card['BILL_AMT1'].median()
bill_amt1_median

bill_amt1_temp = credit_card['BILL_AMT1'] >  350110.68

credit_card['New_bill_1'] = credit_card['BILL_AMT1']
credit_card['New_bill_1'][bill_amt1_temp] = bill_amt1_median

# Outlier in Bill_amount 2

credit_card['BILL_AMT2'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])

bill_amt2_median = credit_card['BILL_AMT2'].median()

bill_amt2_temp = credit_card['BILL_AMT2'] > 337495.28

credit_card['New_bill_2'] = credit_card['BILL_AMT2']
credit_card['New_bill_2'][bill_amt2_temp] = bill_amt2_median

# Outlier in Bill_amount 3

credit_card['BILL_AMT3'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])  #99% good data

bill_amt3_median = credit_card['BILL_AMT3'].median()

bill_amt3_temp = credit_card['BILL_AMT3'] > 325030.39

credit_card['New_bill_3'] = credit_card['BILL_AMT3']
credit_card['New_bill_3'][bill_amt3_temp] = bill_amt3_median

# Outlier in Bill_amount 4

credit_card['BILL_AMT4'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])  #99% good data

bill_amt4_median = credit_card['BILL_AMT4'].median()

bill_amt4_temp = credit_card['BILL_AMT4'] > 304997.27

credit_card['New_bill_4'] = credit_card['BILL_AMT4']
credit_card['New_bill_4'][bill_amt4_temp] = bill_amt4_median

# Outlier in Bill_amount 5

credit_card['BILL_AMT5'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])  #99% good data

bill_amt5_median = credit_card['BILL_AMT5'].median()

bill_amt5_temp = credit_card['BILL_AMT5'] > 285868.33

credit_card['New_bill_5'] = credit_card['BILL_AMT5']
credit_card['New_bill_5'][bill_amt5_temp] = bill_amt5_median

# Outlier in Bill_amount 6

credit_card['BILL_AMT6'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])  #99% good data

bill_amt6_median = credit_card['BILL_AMT6'].median()

bill_amt6_temp = credit_card['BILL_AMT6'] > 279505.06

credit_card['New_bill_6'] = credit_card['BILL_AMT6']
credit_card['New_bill_6'][bill_amt6_temp] = bill_amt6_median

# Outlier in Pay_amount 1

credit_card['PAY_AMT1'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])  #99% good data

pay_amt1_median = credit_card['PAY_AMT1'].median()

pay_amt1_temp = credit_card['PAY_AMT1'] > 66522.18

credit_card['New_pay_amt1'] = credit_card['PAY_AMT1']
credit_card['New_pay_amt1'][pay_amt1_temp] = pay_amt1_median

# Outlier in Pay_amount 2

credit_card['PAY_AMT2'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])  #99% good data

pay_amt2_median = credit_card['PAY_AMT2'].median()

pay_amt2_temp = credit_card['PAY_AMT2'] > 76651.02

credit_card['New_pay_amt2'] = credit_card['PAY_AMT2']
credit_card['New_pay_amt2'][pay_amt2_temp] = pay_amt2_median

# Outlier in Pay_amount 3

credit_card['PAY_AMT3'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])  #99% good data

pay_amt3_median = credit_card['PAY_AMT3'].median()

pay_amt3_temp = credit_card['PAY_AMT3'] > 70000.00

credit_card['New_pay_amt3'] = credit_card['PAY_AMT3']
credit_card['New_pay_amt3'][pay_amt3_temp] = pay_amt3_median

# Outlier in Pay_amount 4

credit_card['PAY_AMT4'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])  #99% good data

pay_amt4_median = credit_card['PAY_AMT4'].median()

pay_amt4_temp = credit_card['PAY_AMT4'] > 67054.44

credit_card['New_pay_amt4'] = credit_card['PAY_AMT4']
credit_card['New_pay_amt4'][pay_amt4_temp] =  pay_amt4_median

# Outlier in Pay_amount 5

credit_card['PAY_AMT5'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])  #99% good data

pay_amt5_median = credit_card['PAY_AMT5'].median()

pay_amt5_temp = credit_card['PAY_AMT5'] > 65607.56

credit_card['New_pay_amt5'] = credit_card['PAY_AMT5']
credit_card['New_pay_amt5'][pay_amt5_temp] =  pay_amt5_median

# Outlier in Pay_amount 6

credit_card['PAY_AMT6'].quantile([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1])  #99% good data

pay_amt6_median = credit_card['PAY_AMT6'].median()

pay_amt6_temp = credit_card['PAY_AMT6'] >  82619.05

credit_card['New_pay_amt6'] = credit_card['PAY_AMT6']
credit_card['New_pay_amt6'][pay_amt6_temp] =  pay_amt6_median

credit_card.info()

credit_card_cleaned = credit_card[['ID', 'new_limit', 'SEX', 'EDUCATION', 'MARRIAGE', 'New_age', 'Pay_0_new', 'Pay_2_new', 'Pay_3_new', 'pay_4_new', 'pay_5_new', 'pay_6_new', 'New_bill_1', 'New_bill_2', 'New_bill_3', 'New_bill_4', 'New_bill_5', 'New_bill_6', 'New_pay_amt1', 'New_pay_amt2', 'New_pay_amt3', 'New_pay_amt4', 'New_pay_amt5', 'New_pay_amt6', 'default.payment.next.month']]
credit_card_cleaned

credit_card_cleaned.info()

credit_card_cleaned.rename(columns = {'new_limit' : 'Balance_Limit', 'New_age' : 'Age', 'Pay_0_new' : 'Pay 0', 'Pay_2_new' : 'Pay 2', 'Pay_3_new' : 'Pay 3', 'pay_4_new' : 'Pay 4', 'pay_5_new' : 'Pay 5', 'pay_6_new' : 'Pay 6',
                                      'New_bill_1' : 'Bill 1', 'New_bill_2' : 'Bill 2', 'New_bill_3' : 'Bill 3', 'New_bill_4' : 'Bill 4', 'New_bill_5' : 'Bill 5', 'New_bill_6' : 'Bill 6',
                                      'New_pay_amt1' : 'Pay amt 1', 'New_pay_amt2' : 'Pay amt 2', 'New_pay_amt3' : 'Pay amt 3', 'New_pay_amt4' : 'Pay amt 4', 'New_pay_amt5' : 'Pay amt 5', 'New_pay_amt6' : 'Pay amt 6'}, inplace = True)

# Modelling credit card after cleaning the data

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(credit_card_cleaned[['Balance Limit']+['Age']+['Pay 0']+['Pay 2']+['Pay 3']+['Pay 4']+['Pay 5']+['Pay 6']+['Bill 1']+['Bill 2']+['Bill 3']+['Bill 4']+['Bill 5']+['Bill 6']+['Pay amt 1']+['Pay amt 2']+['Pay amt 3']+['Pay amt 4']+['Pay amt 5']+['Pay amt 6']], credit_card_cleaned['default.payment.next.month'])

import numpy as np
from sklearn.metrics import confusion_matrix

predict1 = logistic.predict(credit_card_cleaned[['Balance Limit']+['Age']+['Pay 0']+['Pay 2']+['Pay 3']+['Pay 4']+['Pay 5']+['Pay 6']+['Bill 1']+['Bill 2']+['Bill 3']+['Bill 4']+['Bill 5']+['Bill 6']+['Pay amt 1']+['Pay amt 2']+['Pay amt 3']+['Pay amt 4']+['Pay amt 5']+['Pay amt 6']])
predict1

cm1 = confusion_matrix(credit_card_cleaned['default.payment.next.month'], predict1)
print(cm1)

print("col sums", sum(cm1))
total=sum(sum(cm1))
print("Total", total)

accuracy = (cm1[0,0]+cm1[1,1])/total
print("Accuracy", accuracy)

import statsmodels.formula.api as sm

def vif_cal(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]]
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)

vif_cal(input_data=credit_card_cleaned, dependent_col="default.payment.next.month")

vif_cal(input_data=credit_card_cleaned.drop('Bill 2', axis = 1), dependent_col="default.payment.next.month")

import statsmodels.api as sm
m1=sm.Logit(credit_card_cleaned['default.payment.next.month'],credit_card_cleaned[['SEX']+['EDUCATION']+['MARRIAGE']+['Balance Limit']+['Age']+['Pay 0']+['Pay 2']+['Pay 3']+['Pay 4']+['Pay 5']+['Pay 6']+['Bill 1']+['Bill 2']+['Bill 3']+['Bill 4']+['Bill 5']+['Bill 6']+['Pay amt 1']+['Pay amt 2']+['Pay amt 3']+['Pay amt 4']+['Pay amt 5']+['Pay amt 6']])
m1.fit()
print(m1.fit().summary())

import statsmodels.api as sm
m2=sm.Logit(credit_card_cleaned['default.payment.next.month'],credit_card_cleaned[['SEX']+['EDUCATION']+['MARRIAGE']+['Balance Limit']+['Age']+['Pay 0']+['Pay 2']+['Pay 3']+['Pay 5']+['Bill 1']+['Pay amt 1']+['Pay amt 2']+['Pay amt 3']+['Pay amt 4']+['Pay amt 5']+['Pay amt 6']])
m2.fit()
print(m2.fit().summary())

logistic2 = LogisticRegression()
logistic2.fit(credit_card_cleaned[['SEX']+['EDUCATION']+['MARRIAGE']+['Balance Limit']+['Age']+['Pay 0']+['Pay 2']+['Pay 3']+['Pay 5']+['Bill 1']+['Pay amt 1']+['Pay amt 2']+['Pay amt 3']+['Pay amt 4']+['Pay amt 5']+['Pay amt 6']], credit_card_cleaned['default.payment.next.month'])

predict2 = logistic2.predict(credit_card_cleaned[['SEX']+['EDUCATION']+['MARRIAGE']+['Balance Limit']+['Age']+['Pay 0']+['Pay 2']+['Pay 3']+['Pay 5']+['Bill 1']+['Pay amt 1']+['Pay amt 2']+['Pay amt 3']+['Pay amt 4']+['Pay amt 5']+['Pay amt 6']])
predict2

cm2 = confusion_matrix(credit_card_cleaned['default.payment.next.month'], predict2)
print(cm2)

print("col sums", sum(cm2))
total=sum(sum(cm2))
print("Total", total)

accuracy2 = (cm2[0,0]+cm2[1,1])/total
print("Accuracy", accuracy2)