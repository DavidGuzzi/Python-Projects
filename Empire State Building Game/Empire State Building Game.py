#numpy and matplotlib imported
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


#seed set
np.random.seed(7)

#initialize all_walks and and the number of similitudes is selected
all_walks = []
z = 500

#simulate random walk 500 times
for i in range(z) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001:
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)


# In[10]:


#convert all_walks to NumPy array: np_aw
np_aw = np.array(all_walks)

#transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.xlabel('Iteration')
plt.ylabel('Random walks')
plt.show()


# In[11]:


#select last row from np_aw_t: ends
ends = np_aw_t[-1,:]

#plot histogram of ends, display plot
plt.hist(ends)
plt.xlabel('Interval')
plt.ylabel('Ends point frequency')
plt.show()


# In[12]:


#count the number of integers in 'ends' that are greater than or equal to 60
count = 0
for i in ends[ends >= 60]:
    count = count + 1


# In[13]:


#what's the estimated chance that you'll reach at least 60 steps high if you play this Empire State Building game?
print('The chance that this end point is greater than or equal to 60 is '+ str((count/z)*100) + '%')
