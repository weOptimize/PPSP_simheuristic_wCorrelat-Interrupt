#this is the task module
import math
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt 

rng = np.random.default_rng()

#print stars 
def stars(number):
	for i in range(number):
		print("*", end = "")
	print("")

#error messages
def errorCodeMsg():
	print("Error in input file : CODE ")
	quit()
	
def errorPredMsg():
	print("Error in input file : PREDECESSORS ")
	quit()

def errorAlphaMsg():
	print("Error in input file : ALPHA ")
	quit()	

def errorBetaMsg():
	print("Error in input file : BETA ")
	quit()	

def errorDaysMsg():
	print("Error in input file : DAYS ")
	quit()

# Scans if the code in predecessors and successors are
# in the list of task codes:
def getTaskCode(mydata, code):
	x = 0
	flag = 0
	for i in mydata['CODE']:
		if(i == code):
			flag = 1
			break
		
		x+=1
		
	if(flag == 1):
		return x
	else:
		errorCodeMsg()
		
		
# Critical Path Method Forward Pass Function
# EF -> Earliest Finish
# ES -> Earliest Start

def forwardPass(mydata):
	# ntask -> number of tasks
	ntask = mydata.shape[0]
	ES = np.zeros(ntask, dtype = np.float64)
	EF = np.zeros(ntask, dtype = np.float64)
	temp = [] #hold temporary codes
	
	# for each task:
	for i in range(ntask):
		if(mydata['PREDECESSORS'][i] == None):
			ES[i] = 0
			try:
				EF[i] = ES[i] + mydata['SIMDAYS'][i]
			except:
				errorDaysMsg()
				
		else:
			for j in mydata['PREDECESSORS'][i]:
				index = getTaskCode(mydata,j)
				if(index == i):
					errorPredMsg()
				else:
					temp.append(EF[index])
				
			ES[i] = round(max(temp),2)
			try:
				EF[i] = ES[i] + mydata['SIMDAYS'][i]
			except:
				errorDaysMsg()
		
			
		#reset temp
		temp = []
		
	
	#Update dataFrame:
	mydata['ES'] = ES
	mydata['EF'] = EF
	
	return mydata
	
	
# Critical Path Method Backward Pass function
# LS -> Latest Start
# LF -> Latest Finish

def backwardPass(mydata):
	ntask = mydata.shape[0]
	temp = []
	LS = np.zeros(ntask, dtype = np.float64)
	LF = np.zeros(ntask, dtype = np.float64)
	SUCCESSORS = np.empty(ntask, dtype = object)
	
	#create successor column:
	
	for i in range(ntask-1, -1,-1):
		if(mydata['PREDECESSORS'][i] != None):
			for j in mydata['PREDECESSORS'][i]:
				index = getTaskCode(mydata,j)
				if(SUCCESSORS[index] != None):
					SUCCESSORS[index] += mydata['CODE'][i]
				else:
					SUCCESSORS[index] = mydata['CODE'][i]
				
	#incorporate the column to the data frame:
				
	mydata["SUCCESSORS"] = SUCCESSORS
	
	#compute for  EF and LS:
	
	for i in range(ntask-1, -1, -1):
		if(mydata['SUCCESSORS'][i] == None):
			LF[i] = np.max(mydata['EF'])
			LS[i] = (LF[i] - mydata['SIMDAYS'][i])
		else:
			for j in mydata['SUCCESSORS'][i]:
				index = getTaskCode(mydata,j)
				temp.append(LS[index])
			
			LF[i] = round(min(temp),2)
			LS[i] = LF[i] - mydata['SIMDAYS'][i]
			
			#reset temp list:
			temp = [] 
			
	
	#incorporate LF and LS to data frame :
	
	mydata['LS'] = LS
	mydata['LF'] = LF
	
	return mydata
	
#compute for SLACK and CRITICAL state 	
def slack(mydata):
	ntask = mydata.shape[0]
	
	SLACK = np.zeros(shape = ntask, dtype = np.float64)
	CRITICAL = np.empty(shape = ntask,dtype = object)
	
	for i in range(ntask):
		SLACK[i] = round((mydata['LS'][i] - mydata['ES'][i]),2)
		if(SLACK[i] == 0):
			CRITICAL[i] = "YES"
		else:
			CRITICAL[i] = "NO"
			
	#incorporate SLACK and CRITICAL to data frame 
	
	mydata['SLACK'] = SLACK
	mydata['CRITICAL'] = CRITICAL
	
	
	#re arrange columns in dataframe:
	mydata = mydata.reindex(columns = ['DESCR', 'CODE','PREDECESSORS','SUCCESSORS','DAYS','SIMDAYS','ES','EF','LS','LF','SLACK','CRITICAL'])
	
	return mydata
	
#simulate arrival of customers in a queue
def simulatearrivals(average, periods):
	arrivals = []
	rng = np.random.default_rng()
	sa = rng.poisson(average, periods)
	#print(sa)
	#count, bins, ignored = plt.hist(sa, average*3, density=True)
	#plt.show()
	return(sa)

def computeCPM(mydata):
	#create simdays column:
	ntask = mydata.shape[0]
	SIMDAYS = []
	for i in range(ntask):
		#simduration.append(rnd.weibullvariate((mydata['ALPHA'][i]),(mydata['BETA'][i])))
		SIMDAYS.append(round(rnd.triangular((mydata['LOW'][i]),(mydata['HIGH'][i]),(mydata['MODE'][i])),2))
	#incorporate the column of simulated durations to the data frame:
				
	mydata['SIMDAYS'] = SIMDAYS
	#print(SIMDAYS)
	mydata = forwardPass(mydata)
	mydata = backwardPass(mydata)
	mydata = slack(mydata)
	return mydata

def computeRR(myriskreg):
	# nrisk -> number of tasks
	nrisk = myriskreg.shape[0]
	#initialize pxi array with size of nrisk
	pxi = np.zeros(nrisk, dtype = np.float64)
	for i in range(nrisk):
		pxi[i] = round((rng.binomial(1,myriskreg['Probability'][i])) * rnd.triangular(myriskreg['Opt_impact'][i],myriskreg['Pess_impact'][i],myriskreg['ML_impact'][i]),2)
	#sum all values at pxi	
	total_impact_RR = sum(pxi)
	#extract baseline cost from risk register file
	baseline_cost = myriskreg['Base_Bdgt'][0]
	return total_impact_RR, baseline_cost


			
def MCS_CPM_RR(mydata, myriskreg, iterations):
	durationsplus = []
	callsperday= []
	callarray= []
	durat = 0
	duratplus = 0
	totaldays = 0
	totalcalls = 0
	projectcost = []

	for i in range(iterations):
		computeCPM(mydata)
		durat = round(np.max(mydata['EF']),2)
		totaldays = math.ceil(durat)
		callsperday = simulatearrivals(5, totaldays)
		#sum all values at totalcalls
		totalcalls = sum(callsperday)
		callarray = rng.uniform(0.02, 0.06, totalcalls)
		duratplus = round(durat + sum(callarray),2)
		#execute function to compute risk register impact and store the value in variable "total_impact_RR"
		impact_RR = computeRR(myriskreg)
		total_impact_RR = impact_RR[0]
		baseline_cost = impact_RR[1]
		costoftime = duratplus * 3 + total_impact_RR + baseline_cost
		projectcost.append(costoftime)
	#print(durationsplus) #ACTIVAR PARA VER EL RETORNO DE LA FUNCION
	return projectcost


def MCS_CPM_PF(mydata, iterations):
	durationsplus = []
	durations = []
	callsperday= []
	callarray= []
	survivalvalues= []
	durat = 0
	duratplus = 0
	totaldays = 0
	totalcalls = 0
	stdev = 0
	median = 0	

	for i in range(iterations-1):
		computeCPM(mydata)
		durat = round(np.max(mydata['EF']),2)
		totaldays = math.ceil(durat)
		callsperday = simulatearrivals(5, totaldays)
		#sum all values at totalcalls
		totalcalls = sum(callsperday)
		rng = np.random.default_rng()
		callarray = rng.uniform(0.02, 0.06, totalcalls)
		duratplus = round(durat + sum(callarray),2)
		durationsplus.append(duratplus)
		durations.append(durat)

	#print(durations) - noshow porque sólo me interesa durationsplus
	#print(durationsplus) - IMPORTANTE - este es el primero a mostrar si quieres ver las duraciones

	plt.hist(durationsplus, bins = 50) 
	plt.title ("Histogram of CPM durations WITH interruptions")
	#plt.show() #ACTIVAR PARA VER EL HISTOGRAMA
	# plotting the survival function
	values, base = np.histogram(durations, bins=2000)
	valuesplus, base = np.histogram(durationsplus, bins=500) #it returns as many values as specified in bins valuesplus are frequencies, base the x-axis limits for the bins 
	cumulativeplus = np.cumsum(valuesplus)
	survivalvalues = 100*(len(durationsplus)-cumulativeplus)/len(durationsplus)
	#return value at base (which is indeed the durations that correspond to survival level) that matches the index

	#print(duratplus) #ACTIVAR PARA VER EL BUDGETED DURATION Y EL STDEV
	return durationsplus

def printTask(mydata):
	stars(70)
	print("ES = Earliest Start; EF = Earliest Finish; LS = Latest Start, LF = Latest Finish")
	stars(70)
	print(mydata)
	stars(70)
	