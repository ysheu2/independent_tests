#  Multiscale Graph Correlation (MGC) independence test.  MGCPY package(Vogelstein).
#  Can be used to test independence from two high dimensional datasets.
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('classic')
import matplotlib.ticker as ticker
import seaborn as sns; sns.set(style='white')
import pandas as pd
from mgcpy.independence_tests.mgc import MGC
from mgcpy.benchmarks import simulations as sims
np.random.seed(12345678)

# x must have the same dimension for two paired matrices, y can have different dimensions
df_x = pd.read_excel('/Users/yishin/Desktop/control_HRF.xlsx')  # x is time(sec), y is num of samples
df_y = pd.read_excel('/Users/yishin/Desktop/patient_HRF.xlsx')
numpy_x = df_x.values
numpy_y = df_y.values

#  The test statistic is between (-1, 1) due to normalization, p value is calculated using a permutation test.
mgc = MGC()
mgc_statistic, independence_test_metadata = mgc.test_statistic(numpy_x,numpy_y)
p_value, _ = mgc.p_value(numpy_x,numpy_y)
print("MGC test stats:", mgc_statistic)
print("P value:",p_value)
print("Optimal Scale:", independence_test_metadata["optimal_scale"])

# local correlation map
local_corr = independence_test_metadata["local_correlation_matrix"]

# define two rows for subplots
fig, (ax, cax) = plt.subplots(ncols=2, figsize=(9.45, 7.5),  gridspec_kw={"width_ratios":[1, 0.05]})

# draw heatmap
fig.suptitle("Local Correlation Map", fontsize=17)
ax = sns.heatmap(local_corr, cmap="YlGnBu", ax=ax, cbar=False)

# colorbar
fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical")
ax.invert_yaxis()

# optimal scale
optimal_scale = independence_test_metadata["optimal_scale"]
ax.scatter(optimal_scale[0], optimal_scale[1], marker='X', s=200, color='red')

# other formatting
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel('#Neighbors for X', fontsize=15)
ax.set_ylabel('#Neighbors for Y', fontsize=15)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)
cax.xaxis.set_tick_params(labelsize=15)
cax.yaxis.set_tick_params(labelsize=15)
fig.suptitle('cMGC = ' + str(mgc_statistic) + ', pMGC = ' + str(p_value), fontsize=20)

#plt.show()
plt.savefig('correlation_heat_map.png', dpi=300)