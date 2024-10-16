import csv
import matplotlib.pyplot as plt

x_var = []
y_var = []

#Opens the contents of csv file for problem 1
with open('linear_regression_data.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        x_var.append(float(row[0]))
        y_var.append(float(row[1]))

#Calculates the mean
n = len(x_var)
x_mean = sum(x_var) / n
y_mean = sum(y_var) / n

#Calculates the slpoe and intercept for line that best fits for linear reg
numerator = 0
denominator = 0
for i in range(n):
    numerator += (x_var[i] - x_mean) * (y_var[i] - y_mean)
    denominator += (x_var[i] - x_mean) ** 2
    
slope = numerator / denominator
intercept = y_mean - slope * x_mean

#calculates the predicted y values
predictedY = []
for x in x_var:
    predictedY.append(slope * x + intercept)

#uses matplotlib to create a linear reg graph
plt.scatter(x_var, y_var, color='black')
plt.plot(x_var, predictedY, color='blue')
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.show()