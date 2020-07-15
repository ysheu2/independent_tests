# Independence Test: Distance Correlation (Dcorr)
# Dcorr is a measure of dependence (coef=0 --> independent) between 2 paired matrices (can have unequal dimensions)
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('classic')
import matplotlib.ticker as ticker
import seaborn as sns; sns.set(style="white")
import pandas as pd
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.benchmarks import simulations as sims
from mgcpy.hypothesis_tests.transforms import k_sample_transform

np.random.seed(12345678)

#  For two matrices, x must have the same dimension, y can have unequal dimensions
df_x = pd.read_excel('/Users/yishin/Desktop/control_HRF.xlsx') # x is time(secs), y is num of subjects
df_y = pd.read_excel('/Users/yishin/Desktop/patient_HRF.xlsx')
numpy_x = df_x.values
numpy_y = df_y.values

# checking dimensions of the matrices
U = df_x.values
V = df_y.values
print("The shape of U is:", U.shape)
print("The shape of V is:", V.shape)

# some graphs of the raw data
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(U)
ax2.plot(V)

ax1.set_xlabel('time in secs')
ax1.set_ylabel('height')

ax1.set_title('controls')
ax2.set_title('patients')

plt.savefig('rawdata_figure.png', dpi=300)

# Perform dcorr test. (h0=indep, h1=dependent)
dcorr = DCorr(which_test='biased')
dcorr_statistic, independence_test_metadata = dcorr.test_statistic(numpy_x, numpy_y)
p_value, _ = dcorr.p_value(numpy_x, numpy_y)

print("Dcorr test statistic:", dcorr_statistic)
print("P Value:", p_value)

