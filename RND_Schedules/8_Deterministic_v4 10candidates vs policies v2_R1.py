#!/home/pinoystat/Documents/python/mymachine/bin/python

#* get execution time 
import time

start_time = time.time()

#get budgetting confidence policy
#budgetting_confidence_policies = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#array to store all budgeted durations linked to the budgetting confidence policy
budgeteddurations = []
stdevs = []
#array to store all found solutions
solutions = []

#*****

import numpy as np
import pandas as pd
from pandas_ods_reader import read_ods
from operator import itemgetter

import matplotlib.pyplot as plt 

#import created scripts:
from task_only_nominal_no_interrupts import *

#I define the number of candidates to be considered
nrcandidates = 10

#defining a global array that stores all portfolios generated
tested_portfolios = []

#definining the function that randomly generates a portfolio of projects
def generate_portfolio():
    portfolio = np.random.randint(2, size=nrcandidates)
    repetitions = 0
    while portfolio.tolist() in tested_portfolios and repetitions < 100:
        portfolio = np.random.randint(2, size=nrcandidates)
        repetitions = repetitions + 1
    tested_portfolios.append(portfolio.tolist())
    #portfolio = []
    #for i in range(nrcandidates):
    #    portfolio.append(np.random.randint(2))
    return portfolio


#defining the function that calculates the net present value of a portfolio of projects
def portfolio_npv(portfolio):
    npv_portfolio = 0
    for i in range(nrcandidates):
        if portfolio[i] == 1:
            npv_portfolio += npv(wacc, cashflows[i])
    return npv_portfolio

#defining the function that calculates the total budget of a portfolio of projects
def portfolio_totalbudget(portfolio):
    totalbudget_portfolio = 0
    for i in range(nrcandidates):
        if portfolio[i] == 1:
            totalbudget_portfolio += bdgtperproject[i]
    return totalbudget_portfolio

#defining the function that maximizes the net present value of a portfolio of projects, while respecting the budget constraint
def maximize_npv():
    no_update_iter = 0
    #generar un portafolio que no incluye ningun proyecto
    best_portfolio = [0] * nrcandidates
    best_npv = portfolio_npv(best_portfolio)
    best_budget = portfolio_totalbudget(best_portfolio)
    while no_update_iter < 1000:
        no_update_iter = no_update_iter + 1
        new_portfolio = generate_portfolio()
        new_npv = portfolio_npv(new_portfolio)
        new_budget = portfolio_totalbudget(new_portfolio)
        if new_npv > best_npv and new_budget <= maxbdgt:
            best_portfolio = new_portfolio
            best_npv = new_npv
            best_budget = new_budget
            #print("iterations: %s" %(no_update_iter))
            no_update_iter = 0
            #print("new best portfolio: %s, npv: %s, budget: %s" %(best_portfolio, best_npv, best_budget))
    return best_portfolio, best_npv, best_budget

#open ten different ODS files and store the results in a list after computing the CPM and MCS
for i in range(nrcandidates):
    filename = "RND_Schedules/data_wb" + str(i+1) + ".ods"
    print(filename)
    mydata = read_ods(filename, "Sheet1")



    #for xlsx files:
    #mydata = pd.read_excel(io = "data.xlsx",sheet_name = "Sheet1")

    #compute for the Critical Path
    #mydata = computeCPM(mydata)

    #compute MonteCarlo Simulation
    budgetedduration = MCS_CPM (mydata)
    #Print all values returned from MCS_CPM
    #print(budgetedduration)
    #append the budgeted duration to the array of budgeted durations
    budgeteddurations.append(budgetedduration)
    #printTask(mydata)

#I perform a sumproduct to the array of budgeted durations to get the total budgeted cost (each unit in the array costs 500 euros)
totalbudget=sum(budgeteddurations)*500
#I multiply each value in the array of budgeted durations by 500 to get the total budgeted cost per project (each unit in the array costs 500 euros)
bdgtperproject = [x * 500 for x in budgeteddurations]
#I define the budget constraint #was 250k
maxbdgt = 240000
#open a file named "expected_cash_flows.txt", that includes ten rows and five columns, and store the values in a list. Each row corresponds to a project, and each column corresponds to a year
cashflows = []
with open('RND_Schedules/expected_cash_flows.txt') as f:
    for line in f:
        cashflows.append([float(x) for x in line.split()])

#initialize a variable that reflects the weighted average cost of capital
wacc = 0.1
#defining the function that calculates the net present value of a project
def npv(rate, cashflows):
    return sum([cf / (1 + rate) ** i for i, cf in enumerate(cashflows)])

#defining a global array that stores all portfolios generated
solution_portfolios = []

projectselection = maximize_npv()
solutions.append(projectselection)
#separate the npv results from the solutions list
npv_results = [x[1] for x in solutions]
#separate the portfolio results from the solutions list
portfolio_results = [x[0] for x in solutions]
#separate the budgets taken from the solutions list
budgets = [x[2] for x in solutions]
# scatter plot the npv results respect to each budgetting confidence policy
print(portfolio_results)
print(npv_results)
print(budgets)
#print(budgetting_confidence_policies)
#plt.scatter(budgetting_confidence_policies, npv_results)
#zoom in the plot so that the minumum value of the x axis is 0.5 and the maximum value of the x axis is 1
plt.title("NPV vs Budgetting Confidence Policy")
plt.xlabel("Budgetting Confidence Policy")
plt.ylabel("NPV")
plt.xlim(0.45, 1)
plt.ylim(min(npv_results)-100000, max(npv_results)+100000)
plt.grid()		
plt.show()
# create a square array with the information included in portfolio_results
solution_portfolios = np.array(portfolio_results)
# plot the square array as a heatmap, where ones are black and zeros are white
plt.imshow(solution_portfolios, cmap='binary', interpolation='nearest')
plt.xlabel("Project")
#hide the ticks of the x axis but show the labels for each integer value
plt.xticks(np.arange(0, nrcandidates, 1))
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
#remove the label of the y axis
plt.yticks([])
plt.show()



#print(tested_portfolios)
#print("number of tested_portfolios:")
#print(tested_portfolios.__len__())

#*** execution time
print("Execution time: %s milli-seconds" %((time.time() - start_time)* 1000))


  



