#%%
import pandas as pd
import numpy as np

np.random.seed(1)

data = {'A':['a','b','c','d','e','f','g','h','i','j'],
        'B':np.random.exponential(0.5, 10)}
df = pd.DataFrame(data)
df
import matplotlib.pyplot as plt

plt.hist(df['B'])
plt.show()
