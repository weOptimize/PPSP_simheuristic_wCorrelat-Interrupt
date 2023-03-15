#!/home/pinoystat/Documents/python/mymachine/bin/python

#* get execution time 
import time

start_time = time.time()

#get budgetting confidence policy
budgetting_confidence_policies = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#array to store all budgeted durations linked to the budgetting confidence policy
budgeteddurations = []
stdevs = []
#array to store all found solutions
solutions = []
#array to store all results of the monte carlo simulation
mcs_results = []
mcs_results_nointerrupt = []

#*****

import numpy as np
import pandas as pd
import random
import seaborn as sns
from pandas_ods_reader import read_ods
from operator import itemgetter
import matplotlib.pyplot as plt 
from scipy import stats as st
from copulas.multivariate import GaussianMultivariate
from scipy.stats import rv_continuous, rv_histogram, norm, uniform, multivariate_normal, beta


from fitter import Fitter, get_common_distributions, get_distributions


#import created scripts:
from task_rnd_triang_NO_interrupts_stdev_new_R2 import *
from task_rnd_triang_with_interrupts_stdev_new_R2 import *
from survival_value_extractor import *


#I define the number of candidates to be considered
nrcandidates = 10

#defining a global array that stores all portfolios generated (and another one for the ones that entail a solution)
tested_portfolios = []
solution_portfolios = []

#defining the correlation matrix to be used in the monte carlo simulation (and as check when the correlations are expected to be 0)
correlation_matrix = []






#defining the function that, for each budgetting confidence policy, computes the budgeted duration
#of each project and the standard deviation of the budgeted duration (and the related budgeted cost)
#initialize an array of budgeted durations that is nrcandidates x len(budgetting_confidence_policies)
budgetedcosts = np.zeros((nrcandidates, len(budgetting_confidence_policies)))
#initialize an array of standard deviations that is sized as far as nrcandidates
stdevs = np.zeros((nrcandidates, 1))
for i in range(nrcandidates):
    iterations=10000
    #open ten different ODS files and store the results in a list after computing the CPM and MCS
    filename = "RND_Schedules/data_wb" + str(i+1) + ".ods"
    #print(filename)
    mydata = read_ods(filename, "Sheet1")
    #open ten different ODS files and store the results in a list after computing the CPM and MCS
    filename = "RND_Schedules/riskreg_" + str(i+1) + ".ods"
    #print(filename)
    myriskreg = read_ods(filename, "riskreg")
    #compute MonteCarlo Simulation and store the results in an array called "sim_durations"
    sim_costs_nointerrupt = MCS_CPM_RRn(mydata, myriskreg, iterations)
    sim_costs = MCS_CPM_RR(mydata, myriskreg, iterations)
    #multiply each value in sim_durations by 5000 to get the results in Euros
    #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the iterations
    mcs_results_nointerrupt.append(sim_costs_nointerrupt)
    mcs_results.append(sim_costs)
    for j in range(len(budgetting_confidence_policies)):
        budgetting_confidence_policy = budgetting_confidence_policies[j]
        #print(budgetting_confidence_policy)
        #extract the survival value from the array sim_duration that corresponds to the budgetting confidence policy
        survival_value = survival_value_extractor(sim_costs, budgetting_confidence_policy, iterations)
        #store the first survival value in an array where the columns correspond to the budgetting confidence policies and the rows correspond to the projects
        budgetedcosts[i][j]=survival_value
    #I perform a sumproduct to the array of budgeted durations to get the total budgeted cost (each unit in the array costs 5000 euros) now x1 because I did it before
    #totalbudget=sum(budgetedcosts)*1
    #I multiply each value in the array of budgeted durations by 5000 to get the total budgeted cost per project (each unit in the array costs 5000 euros) keeping the same type of array
    bdgtperproject_matrix=budgetedcosts*1



# copy the array with all MCS results
df0 = pd.DataFrame(data=mcs_results).T
df0.rename(columns={0:"P01", 1:"P02", 2:"P03", 3:"P04", 4:"P05", 5:"P06", 6:"P07", 7:"P08", 8:"P09", 9:"P10"}, inplace=True)
df0_nointerrupt = pd.DataFrame(data=mcs_results_nointerrupt).T
df0_nointerrupt.rename(columns={0:"P01", 1:"P02", 2:"P03", 3:"P04", 4:"P05", 5:"P06", 6:"P07", 7:"P08", 8:"P09", 9:"P10"}, inplace=True)#*** execution time
print("Execution time after SIMULATION step: %s milli-seconds" %((time.time() - start_time)* 1000))



#choose a portfolio that includes the ten projects and sets all to one
chosen_portfolio = np.ones(nrcandidates)
#multiply dataframe 0 by the chosen portfolio to reflect the effect of the projects that are chosen
pf_df = df0 * chosen_portfolio
pf_df_nointerrupt = df0_nointerrupt * chosen_portfolio
#sum the rows of the new dataframe to calculate the total cost of the portfolio
pf_cost = pf_df.sum(axis=1)
pf_cost_nointerrupt = pf_df_nointerrupt.sum(axis=1)

# title of the plot
# ax.set_title('Monte Carlo Simulation of a candidate project')
# ax.hist(mcs_results[0], bins=200, color='grey', label='Histogram')
# ax.hist(mcs_results[0], bins=200, color='grey', label='Histogram')
# title of the x axis
# ax.set_xlabel('Cost in k€')
# Create a twin Axes object that shares the x-axis of the original Axes object
# ax2 = ax.twinx()
# Plot the histogram of the monte carlo simulation of the first project in the form of a cumulative distribution function
# ax2.hist(mcs_results[0], bins=200, color='black', cumulative=True, histtype='step', density=True, label='Cumulative Distribution')
# Plot the histogram of the monte carlo simulation mcs_results_nointerrupt in the form of a cumulative distribution function at the same plot
# ax2.hist(mcs_results_nointerrupt[0], bins=200, color='red', cumulative=True, histtype='step', density=True, label='Cumulative Distribution without interruptions')
# Set the y-axis of the twin Axes object to be visible
# ax2.yaxis.set_visible(True)
#set maximum value of the y axis of the twin Axes object to 1
# ax2.set_ylim(0, 1)
# add grid to the plot following the y axis of the twin Axes object
# ax2.grid(axis='y')
# add grid to the plot following the x axis of the original Axes object
# ax.grid(axis='x')
# Add legend
# ax.legend(loc='center left')
# ax2.legend(loc='upper left')


# Plot the histograms
fig, ax = plt.subplots()
ax.hist(pf_cost, bins=200, alpha=0.5, density=True, color='grey', label='Aggregated costs of the ten candidate projects with interruptions')
ax.hist(pf_cost_nointerrupt, bins=200, alpha=0.8, density=True, color='black', label='Aggregated costs of the ten candidate projects without interruptions')

# Calculate the parameters of the normal distribution fits
mu1, std1 = norm.fit(pf_cost)
mu2, std2 = norm.fit(pf_cost_nointerrupt)

# Plot the normal distribution fits
x = np.linspace(pf_cost.min(), pf_cost_nointerrupt.max(), 100)
# Plot the normal distribution fit without interruption in grey colour
ax.plot(x, norm.pdf(x, mu1, std1), 'grey', label='Normal fit with interruptions')
# plot mu1 and mu2
ax.axvline(mu1, color='grey', linestyle='dashed', linewidth=1)
ax.axvline(mu2, color='black', linestyle='dashed', linewidth=1)
# plot a line that comes from mu1 and mu2 and align it vertically to the top
ax.plot([mu1, mu2], [0.009, 0.009], 'k-', lw=1)
# plot the difference between mu1 and mu2 and align vertically to the top
ax.text((mu1+mu2)/2, 0.009, str(round(mu1-mu2, 2))+' k€', horizontalalignment='center', verticalalignment='top')




# Plot the normal distribution fit without interruption in black colour
ax.plot(x, norm.pdf(x, mu2, std2), 'k-', label='Normal fit without interruptions')


# Add a legend and title
ax.legend()
ax.set_title('Histogram and Normal Distribution Fit')

plt.show()
#*** execution time
print("Execution time: %s milli-seconds" %((time.time() - start_time)* 1000))


  



