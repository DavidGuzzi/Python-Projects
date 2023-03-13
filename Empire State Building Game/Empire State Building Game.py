#!/usr/bin/env python
# coding: utf-8

# ## **Empire State Building Game** ##

# Empire State Building Game is a game where players take turns throwing a die 100 times and climbing up the stairs of the Empire State Building based on the results. Each time a player rolls the die, if the result is 1 or 2, the player goes back one step. If the die roll is 3, 4, or 5, the player goes forward one step. If the die roll is 6, the player re-rolls the die and moves forward the number of steps indicated by the new roll. However, there are some restrictions in the game. The player cannot fall below step 0 and there is a 0.1% chance that the player falls down the stairs and starts again from step 0. The goal of the game is to reach the top of the building, represented by step number 60.
# 
# **A model will be created that attempts to replicate the game. The goal will be to find the probability of reaching step number 60.**

# In[8]:


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

