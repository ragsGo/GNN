import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
df = pd.read_excel('new_results.xlsx', sheet_name='best_wheat')

plt.plot(df.actual)
plt.plot(df.predict, 'o')
plt.show()
true_value = df.actual

predicted_value = df.predict
r2score = r2_score(true_value, predicted_value)
print('r2score--', r2score)


plt.figure(figsize=(10,10))
plt.scatter(true_value, predicted_value, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(predicted_value), max(true_value))
p2 = min(min(predicted_value), min(true_value))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
Calculate error from between target and predictions
(Based on merged data frame of test data and predictions)
"""
df['error'] = df['actual'] - df['predict']

# #Set plot size
# plt.subplots(figsize=(10,5))
# #Set X-Axis range
# plt.xlim(-20, 20)
# plt.title('Model Error Distribution')
# plt.ylabel('No. of Predictions')
# plt.xlabel('Error')
# plt.hist(df['error'], bins=np.linspace(-20, 20, num=41, dtype=int));
# plt.show()


sns.regplot(x='actual', y="predict", data=df)
plt.show()
sns.regplot(x='predict', y='error', data=df, scatter=True, color='red')
plt.show()

# import statsmodels.api as sm
# import scipy.stats as stats
# sm.qqplot(df.predict,line='45',fit=True,dist=stats.norm)
# sm.qqplot(df.actual,line='45',fit=True,dist=stats.norm)
# sm.qqplot(df.error,line='45',fit=True,dist=stats.norm)
# plt.show()