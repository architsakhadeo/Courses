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




# Welch T test for different variances, averaged across 50 runs for 100 episodes, 100 outputs, axis=0

print(len(np.mean(np.array(all_returns_baseline),0)))

print('T-test for (Baseline, no discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline),0), np.mean(np.array(all_returns_baseline),0), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline),0), np.mean(np.array(all_returns_baseline_discounted),0), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline),0), np.mean(np.array(all_returns_nobaseline),0), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline),0), np.mean(np.array(all_returns_nobaseline_discounted),0), axis=0, equal_var=False))

print('T-test for (Baseline, discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted),0), np.mean(np.array(all_returns_baseline),0), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted),0), np.mean(np.array(all_returns_baseline_discounted),0), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted),0), np.mean(np.array(all_returns_nobaseline),0), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted),0), np.mean(np.array(all_returns_nobaseline_discounted),0), axis=0, equal_var=False))

print('T-test for (No Baseline, no discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline),0), np.mean(np.array(all_returns_baseline),0), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline),0), np.mean(np.array(all_returns_baseline_discounted),0), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline),0), np.mean(np.array(all_returns_nobaseline),0), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline),0), np.mean(np.array(all_returns_nobaseline_discounted),0), axis=0, equal_var=False))

print('T-test for (No Baseline, discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted),0), np.mean(np.array(all_returns_baseline),0), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted),0), np.mean(np.array(all_returns_baseline_discounted),0), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted),0), np.mean(np.array(all_returns_nobaseline),0), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted),0), np.mean(np.array(all_returns_nobaseline_discounted),0), axis=0, equal_var=False))

print()
print()
print(len(np.mean(np.array(all_returns_baseline),1)))
# Welch T test for different variances, averaged across 100 episodes for 50 runs, 50 outputs, axis=1

print('T-test for (Baseline, no discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline),1), np.mean(np.array(all_returns_baseline),1), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline),1), np.mean(np.array(all_returns_baseline_discounted),1), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline),1), np.mean(np.array(all_returns_nobaseline),1), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline),1), np.mean(np.array(all_returns_nobaseline_discounted),1), axis=0, equal_var=False))

print('T-test for (Baseline, discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted),1), np.mean(np.array(all_returns_baseline),1), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted),1), np.mean(np.array(all_returns_baseline_discounted),1), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted),1), np.mean(np.array(all_returns_nobaseline),1), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted),1), np.mean(np.array(all_returns_nobaseline_discounted),1), axis=0, equal_var=False))

print('T-test for (No Baseline, no discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline),1), np.mean(np.array(all_returns_baseline),1), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline),1), np.mean(np.array(all_returns_baseline_discounted),1), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline),1), np.mean(np.array(all_returns_nobaseline),1), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline),1), np.mean(np.array(all_returns_nobaseline_discounted),1), axis=0, equal_var=False))

print('T-test for (No Baseline, discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted),1), np.mean(np.array(all_returns_baseline),1), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted),1), np.mean(np.array(all_returns_baseline_discounted),1), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted),1), np.mean(np.array(all_returns_nobaseline),1), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted),1), np.mean(np.array(all_returns_nobaseline_discounted),1), axis=0, equal_var=False))
print()
print()

###################################################


print(len(np.mean(np.array(all_returns_baseline_train),0)))

print('T-test for (Baseline, no discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_train),0), np.mean(np.array(all_returns_baseline),0), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_train),0), np.mean(np.array(all_returns_baseline_discounted),0), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_train),0), np.mean(np.array(all_returns_nobaseline),0), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_train),0), np.mean(np.array(all_returns_nobaseline_discounted),0), axis=0, equal_var=False))

print('T-test for (Baseline, discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted_train),0), np.mean(np.array(all_returns_baseline),0), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted_train),0), np.mean(np.array(all_returns_baseline_discounted),0), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted_train),0), np.mean(np.array(all_returns_nobaseline),0), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted_train),0), np.mean(np.array(all_returns_nobaseline_discounted),0), axis=0, equal_var=False))

print('T-test for (No Baseline, no discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_train),0), np.mean(np.array(all_returns_baseline),0), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_train),0), np.mean(np.array(all_returns_baseline_discounted),0), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_train),0), np.mean(np.array(all_returns_nobaseline),0), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_train),0), np.mean(np.array(all_returns_nobaseline_discounted),0), axis=0, equal_var=False))

print('T-test for (No Baseline, discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted_train),0), np.mean(np.array(all_returns_baseline),0), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted_train),0), np.mean(np.array(all_returns_baseline_discounted),0), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted_train),0), np.mean(np.array(all_returns_nobaseline),0), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted_train),0), np.mean(np.array(all_returns_nobaseline_discounted),0), axis=0, equal_var=False))


print()
print()
print(len(np.mean(np.array(all_returns_baseline_train),1)))
# Welch T test for different variances, averaged across 100 episodes for 50 runs, 50 outputs, axis=1


print('T-test for (Baseline, no discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_train),1), np.mean(np.array(all_returns_baseline_train),1), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_train),1), np.mean(np.array(all_returns_baseline_discounted_train),1), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_train),1), np.mean(np.array(all_returns_nobaseline_train),1), axis=0, equal_var=False))
print('T-test for (Baseline, no discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_train),1), np.mean(np.array(all_returns_nobaseline_discounted_train),1), axis=0, equal_var=False))

print('T-test for (Baseline, discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted_train),1), np.mean(np.array(all_returns_baseline_train),1), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted_train),1), np.mean(np.array(all_returns_baseline_discounted_train),1), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted_train),1), np.mean(np.array(all_returns_nobaseline_train),1), axis=0, equal_var=False))
print('T-test for (Baseline, discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_baseline_discounted_train),1), np.mean(np.array(all_returns_nobaseline_discounted_train),1), axis=0, equal_var=False))

print('T-test for (No Baseline, no discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_train),1), np.mean(np.array(all_returns_baseline_train),1), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_train),1), np.mean(np.array(all_returns_baseline_discounted_train),1), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_train),1), np.mean(np.array(all_returns_nobaseline_train),1), axis=0, equal_var=False))
print('T-test for (No Baseline, no discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_train),1), np.mean(np.array(all_returns_nobaseline_discounted_train),1), axis=0, equal_var=False))

print('T-test for (No Baseline, discounting) and (Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted_train),1), np.mean(np.array(all_returns_baseline_train),1), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted_train),1), np.mean(np.array(all_returns_baseline_discounted_train),1), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (No Baseline, no discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted_train),1), np.mean(np.array(all_returns_nobaseline_train),1), axis=0, equal_var=False))
print('T-test for (No Baseline, discounting) and (No Baseline, discounting) : ', stats.ttest_ind(np.mean(np.array(all_returns_nobaseline_discounted_train),1), np.mean(np.array(all_returns_nobaseline_discounted_train),1), axis=0, equal_var=False))
print()
print()
