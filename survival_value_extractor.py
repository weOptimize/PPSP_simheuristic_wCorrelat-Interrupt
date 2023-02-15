#This file includes the function that returns the survival value for a given budgetting confidence policy.
#It is called from the main file
import math
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt


#define the function that returns the survival value for a given budgetting confidence policy
def survival_value_extractor(sim_costs, budgetting_confidence_policy, iterations):
	# plt.show()
	#plt.hist(sim_durations, bins = iterations) 
	#plt.title ("Histogram of CPM durations WITH interruptions")
	#plt.xlabel("Duration (days)")
	#plt.ylabel("Frequency")
	#plt.show() #ACTIVAR PARA VER EL HISTOGRAMA
	# plotting the survival function
    #calculate the cumulative sum of the values of the histogram
	valuesplus, base = np.histogram(sim_costs, bins=iterations) #it returns as many values as specified in bins valuesplus are frequencies, base the x-axis limits for the bins 
	cumulativeplus = np.cumsum(valuesplus)
	survivalvalues = 100*(len(sim_costs)-cumulativeplus)/len(sim_costs)
	#return index of item from survivalvalues that is closest to "1-budgetting_confidence_policy" typ.20%
	index = (np.abs(survivalvalues-100*(1-budgetting_confidence_policy))).argmin()
	#return value at base (which is indeed the durations that correspond to survival level) that matches the index
	budgetedduration = np.round(base[index],2)
	return budgetedduration
    
	#print(valuesplus)
	# plt.plot(base[:-1], len(durations)-cumulative, c='green')
	#plt.plot(base[:-1], 100-survivalvalues, c='green')	 #ACTIVAR PARA VER EL HISTOGRAMA
	#plt.title ("Survival function of CPM durations WITH interruptions")
	#set vertical tick label every 10 points
	#plt.yticks(np.arange(0, 101, 10))
	#plt.xlabel("Duration (days)")
	#plt.ylabel("Fulfilment confidence (%)")
	#plt.grid()		 #ACTIVAR PARA VER EL HISTOGRAMA
	#plt.show() #ACTIVAR PARA VER EL HISTOGRAMA
	#print(base)
	#print(cumulativeplus)
	#print(survivalvalues)
	#print(index)
    