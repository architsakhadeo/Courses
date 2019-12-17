import pickle
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy import stats

dir = 'runs_data_cartpole1_50runs_new/testing/baseline/'
files = os.listdir(dir)
all_returns_baseline = []
for file in files:
    f = open(dir+file, 'rb')
    all_returns_baseline.append(pickle.load(f))
    f.close()

dir_ppo = 'runs_data_cartpole1_50runs_new/testing/baseline_discounted/'
files_ppo = os.listdir(dir_ppo)
all_returns_baseline_discounted = []
for file_ppo in files_ppo:
    f_ppo = open(dir_ppo+file_ppo, 'rb')
    all_returns_baseline_discounted.append(pickle.load(f_ppo))
    f_ppo.close()

dir_wob = 'runs_data_cartpole1_50runs_new/testing/nobaseline/'
files_wob = os.listdir(dir_wob)
all_returns_nobaseline = []
for file_wob in files_wob:
    f_wob = open(dir_wob+file_wob, 'rb')
    all_returns_nobaseline.append(pickle.load(f_wob))
    f_wob.close()

dir_ppo1 = 'runs_data_cartpole1_50runs_new/testing/nobaseline_discounted/'
files_ppo1 = os.listdir(dir_ppo1)
all_returns_nobaseline_discounted = []
for file_ppo1 in files_ppo1:
    f_ppo1 = open(dir_ppo1+file_ppo1, 'rb')
    all_returns_nobaseline_discounted.append(pickle.load(f_ppo1))
    f_ppo1.close()



dir = 'runs_data_cartpole1_50runs_new/training/baseline/'
files = os.listdir(dir)
all_returns_baseline_train = []
for file in files:
    f = open(dir+file, 'rb')
    all_returns_baseline_train.append(pickle.load(f))
    f.close()

dir_ppo = 'runs_data_cartpole1_50runs_new/training/baseline_discounted/'
files_ppo = os.listdir(dir_ppo)
all_returns_baseline_discounted_train = []
for file_ppo in files_ppo:
    f_ppo = open(dir_ppo+file_ppo, 'rb')
    all_returns_baseline_discounted_train.append(pickle.load(f_ppo))
    f_ppo.close()

dir_wob = 'runs_data_cartpole1_50runs_new/training/nobaseline/'
files_wob = os.listdir(dir_wob)
all_returns_nobaseline_train = []
for file_wob in files_wob:
    f_wob = open(dir_wob+file_wob, 'rb')
    all_returns_nobaseline_train.append(pickle.load(f_wob))
    f_wob.close()

dir_ppo1 = 'runs_data_cartpole1_50runs_new/training/nobaseline_discounted/'
files_ppo1 = os.listdir(dir_ppo1)
all_returns_nobaseline_discounted_train = []
for file_ppo1 in files_ppo1:
    f_ppo1 = open(dir_ppo1+file_ppo1, 'rb')
    all_returns_nobaseline_discounted_train.append(pickle.load(f_ppo1))
    f_ppo1.close()





all_returns_baseline = np.mean(np.array(all_returns_baseline),1)
all_returns_baseline_discounted = np.mean(np.array(all_returns_baseline_discounted),1)
all_returns_nobaseline = np.mean(np.array(all_returns_nobaseline),1)
all_returns_nobaseline_discounted = np.mean(np.array(all_returns_nobaseline_discounted),1)

all_returns_baseline_train = np.mean(np.array(all_returns_baseline_train),1)
all_returns_baseline_discounted_train = np.mean(np.array(all_returns_baseline_discounted_train),1)
all_returns_nobaseline_train = np.mean(np.array(all_returns_nobaseline_train),1)
all_returns_nobaseline_discounted_train = np.mean(np.array(all_returns_nobaseline_discounted_train),1)

print()
print('Test')
print()

count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline)):
    if all_returns_baseline[i] > all_returns_baseline_discounted[i]:
        count1 += 1
    elif all_returns_baseline[i] < all_returns_baseline_discounted[i]:
        count2 += 1
    else:
        count3 += 1
print('Baseline, no discounting is better than Baseline, discounting ', str(count1), ' out of ', str(len(all_returns_baseline)))
print('Baseline, discounting is better than Baseline, no discounting ', str(count2), ' out of ', str(len(all_returns_baseline)))
print('Baseline, no discounting is equal to Baseline, discounting ', str(count3), ' out of ', str(len(all_returns_baseline)))

count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline)):
    if all_returns_baseline[i] > all_returns_nobaseline[i]:
        count1 += 1
    elif all_returns_baseline[i] < all_returns_nobaseline[i]:
        count2 += 1
    else:
        count3 += 1
print('Baseline, no discounting is better than No Baseline, no discounting ', str(count1), ' out of ', str(len(all_returns_baseline)))
print('No Baseline, no discounting is better than Baseline, no discounting ', str(count2), ' out of ', str(len(all_returns_baseline)))
print('Baseline, no discounting is better than No Baseline, no discounting ', str(count3), ' out of ', str(len(all_returns_baseline)))

count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline)):
    if all_returns_baseline[i] > all_returns_nobaseline_discounted[i]:
        count1 += 1
    elif all_returns_baseline[i] < all_returns_nobaseline_discounted[i]:
        count2 += 1
    else:
        count3 += 1
print('Baseline, no discounting is better than No Baseline, discounting ', str(count1), ' out of ', str(len(all_returns_baseline)))
print('No Baseline, discounting is better than Baseline, no discounting ', str(count2), ' out of ', str(len(all_returns_baseline)))
print('Baseline, no discounting is better than No Baseline, discounting ', str(count3), ' out of ', str(len(all_returns_baseline)))

count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline)):
    if all_returns_baseline_discounted[i] > all_returns_nobaseline[i]:
        count1 += 1
    elif all_returns_baseline_discounted[i] < all_returns_nobaseline[i]:
        count2 += 1
    else:
        count3 += 1
print('Baseline, discounting is better than No Baseline, no discounting ', str(count1), ' out of ', str(len(all_returns_baseline)))
print('No Baseline, no discounting is better than Baseline, discounting ', str(count2), ' out of ', str(len(all_returns_baseline)))
print('Baseline, discounting is better than No Baseline, no discounting ', str(count3), ' out of ', str(len(all_returns_baseline)))

count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline)):
    if all_returns_baseline_discounted[i] > all_returns_nobaseline_discounted[i]:
        count1 += 1
    elif all_returns_baseline_discounted[i] < all_returns_nobaseline_discounted[i]:
        count2 += 1
    else:
        count3 += 1
print('Baseline, discounting is better than No Baseline, discounting ', str(count1), ' out of ', str(len(all_returns_baseline)))
print('No Baseline, discounting is better than Baseline, discounting ', str(count2), ' out of ', str(len(all_returns_baseline)))
print('Baseline, discounting is better than No Baseline, discounting ', str(count3), ' out of ', str(len(all_returns_baseline)))

count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline)):
    if all_returns_nobaseline[i] > all_returns_nobaseline_discounted[i]:
        count1 += 1
    elif all_returns_baseline[i] < all_returns_nobaseline_discounted[i]:
        count2 += 1
    else:
        count3 += 1
print('No Baseline, no discounting is better than No Baseline, discounting ', str(count1), ' out of ', str(len(all_returns_baseline)))
print('No Baseline, discounting is better than No Baseline, no discounting ', str(count2), ' out of ', str(len(all_returns_baseline)))
print('No Baseline, no discounting is better than No Baseline, discounting ', str(count3), ' out of ', str(len(all_returns_baseline)))

print()
print('Train')
print()
##############

count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline_train)):
    if all_returns_baseline_train[i] > all_returns_baseline_discounted_train[i]:
        count1 += 1
    elif all_returns_baseline_train[i] < all_returns_baseline_discounted_train[i]:
        count2 += 1
    else:
        count3 += 1
print('Baseline, no discounting is better than Baseline, discounting ', str(count1), ' out of ', str(len(all_returns_baseline_train)))
print('Baseline, discounting is better than Baseline, no discounting ', str(count2), ' out of ', str(len(all_returns_baseline_train)))
print('Baseline, no discounting is better than Baseline, discounting ', str(count3), ' out of ', str(len(all_returns_baseline_train)))

count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline_train)):
    if all_returns_baseline_train[i] > all_returns_nobaseline_train[i]:
        count1 += 1
    elif all_returns_baseline_train[i] < all_returns_nobaseline_train[i]:
        count2 += 1
    else:
        count3 += 1
print('Baseline, no discounting is better than No Baseline, no discounting ', str(count1), ' out of ', str(len(all_returns_baseline_train)))
print('No Baseline, no discounting is better than Baseline, no discounting ', str(count2), ' out of ', str(len(all_returns_baseline_train)))
print('Baseline, no discounting is better than No Baseline, no discounting ', str(count3), ' out of ', str(len(all_returns_baseline_train)))

count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline_train)):
    if all_returns_baseline_train[i] > all_returns_nobaseline_discounted_train[i]:
        count1 += 1
    elif all_returns_baseline_train[i] < all_returns_nobaseline_discounted_train[i]:
        count2 += 1
    else:
        count3 += 1
print('Baseline, no discounting is better than No Baseline, discounting ', str(count1), ' out of ', str(len(all_returns_baseline_train)))
print('No Baseline, discounting is better than Baseline, no discounting ', str(count2), ' out of ', str(len(all_returns_baseline_train)))
print('Baseline, no discounting is better than No Baseline, discounting ', str(count3), ' out of ', str(len(all_returns_baseline_train)))


count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline_train)):
    if all_returns_baseline_discounted_train[i] > all_returns_nobaseline_train[i]:
        count1 += 1
    elif all_returns_baseline_discounted_train[i] < all_returns_nobaseline_train[i]:
        count2 += 1
    else:
        count3 += 1
print('Baseline, discounting is better than No Baseline, no discounting ', str(count1), ' out of ', str(len(all_returns_baseline_train)))
print('No Baseline, no discounting is better than Baseline, discounting ', str(count2), ' out of ', str(len(all_returns_baseline_train)))
print('Baseline, discounting is better than No Baseline, no discounting ', str(count3), ' out of ', str(len(all_returns_baseline_train)))


count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline_train)):
    if all_returns_baseline_discounted_train[i] > all_returns_nobaseline_discounted_train[i]:
        count1 += 1
    elif all_returns_baseline_discounted_train[i] < all_returns_nobaseline_discounted_train[i]:
        count2 += 1
    else:
        count3 += 1
print('Baseline, discounting is better than No Baseline, discounting ', str(count1), ' out of ', str(len(all_returns_baseline_train)))
print('No Baseline, discounting is better than Baseline, discounting ', str(count2), ' out of ', str(len(all_returns_baseline_train)))
print('Baseline, discounting is better than No Baseline, discounting ', str(count3), ' out of ', str(len(all_returns_baseline_train)))


count1 = 0
count2 = 0
count3 = 0
for i in range(len(all_returns_baseline_train)):
    if all_returns_nobaseline_train[i] > all_returns_nobaseline_discounted_train[i]:
        count1 += 1
    elif all_returns_baseline_train[i] < all_returns_nobaseline_discounted_train[i]:
        count2 += 1
    else:
        count3 += 1
print('No Baseline, no discounting is better than No Baseline, discounting ', str(count1), ' out of ', str(len(all_returns_baseline_train)))
print('No Baseline, discounting is better than No Baseline, no discounting ', str(count2), ' out of ', str(len(all_returns_baseline_train)))
print('No Baseline, no discounting is better than No Baseline, discounting ', str(count3), ' out of ', str(len(all_returns_baseline_train)))


