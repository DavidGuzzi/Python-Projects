#%%
import pandas as pd
import numpy as np

np.random.seed(1)

data = {'A':['a','b','c','d','e','f','g','h','i','j'],
        'B':np.random.exponential(0.5, 10)*100}
df = pd.DataFrame(data)
df_procesado = df.copy()
import matplotlib.pyplot as plt

# plt.hist(df['B'])
# plt.show()

Q1 = df['B'].quantile(0.25)
Q3 = df['B'].quantile(0.75)
IQR = Q3 - Q1
threshold = 1.5*IQR
limite_inferior = Q1 - threshold
limite_superior = Q3 + threshold
atipicos_inferiores = df['B'] < limite_inferior
atipicos_superiores = df['B'] > limite_superior
ext_inf_a_005 = atipicos_inferiores.sum()
ext_sup_a_095 = atipicos_superiores.sum()
df_procesado.loc[atipicos_inferiores, 'B'] = df['B'].quantile(0.05)
df_procesado.loc[atipicos_superiores, 'B'] = df['B'].quantile(0.95)

# plt.hist(df_procesado['B'])
# plt.show()


#%%
# df_procesado['B'].skew()
import pandas as pd
import numpy as np

data = {'A':['a','b','c','d','e','f','g','h','i','j'],
        'B':[1,1,1,1,2,2,2,2,2,3]}
df = pd.DataFrame(data)
# set(df['B'])
df_np = np.array(df)
df_np

#%%

import numpy as np
a=np.random.randn(4,3)
b=np.random.randn(4,1)

c=[]
for i in range(3):
        for j in range(4):
                print(a[i][j]+b[j])
#%%
import numpy as np
x = np.random.rand(3, 2)

y = np.sum(x, axis=0, keepdims=True)

y.shape