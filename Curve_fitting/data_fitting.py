import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# Reading Data
data = pd.read_csv('./Curve_fitting/Data.csv')
charges = data['charges'].values
age =  data['age'].values
age = np.array(age)
charges = np.array(charges)

#find the mean of data
charges_mean = np.mean(charges)
age_mean = np.mean(age)

#finding the co-variance for the data
co_var = np.zeros((2,2))
age_norm = []
charges_norm = []
for i in range(len(age)):
    age_norm.append((age[i]-np.min(age))/(np.max(age) - np.min(age)))
    charges_norm.append((charges[i]-np.min(charges))/(np.max(charges) - np.min(charges)))

charges_mean_norm = np.mean(charges_norm)
age_mean_norm = np.mean(age_norm)
for i in range(len(age)):
    co_var[0][0] += ((age_norm[i] - age_mean_norm)**2)
    co_var[1][0] += (age_norm[i] - age_mean_norm)*(charges_norm[i] - charges_mean_norm)
    co_var[0][1] += (age_norm[i] - age_mean_norm)*(charges_norm[i] - charges_mean_norm)
    co_var[1][1] += ((charges_norm[i] - charges_mean_norm)**2)

co_var[0][0] = co_var[0][0]/len(age)
co_var[0][1] = co_var[0][1]/len(age)
co_var[1][0] = co_var[1][0]/len(age)
co_var[1][1] = co_var[1][1]/len(charges)

#Finding the eigen vectors and eigen values for the co-variance matrix
eigval,eigvec = np.linalg.eig(co_var)
eigvec = eigvec.T

for i in range(2):
    eigvec[i][0] = eigval[i]*eigvec[i][0]
    eigvec[i][1] = eigval[i]*eigvec[i][1]
eigvec = eigvec.T
plt.scatter(age,charges)
origin = [np.median(age),np.median(charges)]
eig_vec1 = eigvec[:,0]
eig_vec2 = eigvec[:,1]
eigen_plot = plt.figure(1)
plt.xlabel("age")
plt.ylabel("charges")
plt.title("Eigenvector of covariance matrix")
plt.quiver(*origin, *eig_vec1, color=['r'],scale = 0.5)
plt.quiver(*origin, *eig_vec2, color=['b'], scale = 0.5)



#Standard Least square method
X = np.array([age,np.ones(len(age))])
X = X.T
Y = np.array(charges)
B = np.linalg.inv(X.T @ X) @ X.T @ Y
charge_fit = []
for i in age:
    charge_fit.append(B[0]*i + B[1])
fit_plot = plt.figure(2)
plt.scatter(age,charges)
plt.plot(age,charge_fit,color = 'red',label="Standard Least Square")

for i in range(len(age)):
    co_var[0][0] += ((age[i] - age_mean)**2)
    co_var[1][0] += (age[i] - age_mean)*(charges[i] - charges_mean)
    co_var[0][1] += (age[i] - age_mean)*(charges[i] - charges_mean)
    co_var[1][1] += ((charges[i] - charges_mean)**2)


#Total least square method
uTu = [[co_var[0][0],co_var[0][1]],[co_var[1][0],co_var[1][1]]]
uTueig_val,uTueig_vec = np.linalg.eig(uTu)
min_idx = np.where(uTueig_val == np.min(uTueig_val))
a = uTueig_vec[0][min_idx]
b = uTueig_vec[1][min_idx]
d = a*age_mean + b*charges_mean
Charges_tls = [] 
for i in range(len(age)):
    Charges_tls.append((d - a*age[i])/b)
plt.plot(age,Charges_tls,color = 'blue',label="Total Least Square")


#RANSAC method
error = 0.2
probability = 0.95
sample = 2
N = np.log(1 - probability)/np.log(1 - (1 - error)**sample)
N = math.ceil(N)
A = np.array([age,charges])
A = A.T
rnsc_points = []
for i in range(N):
    random_pts = np.random.choice(list(range(len(data))),size=sample,replace=False)
    m = (A[random_pts[1]][1] - A[random_pts[0]][1])/(A[random_pts[1]][0] - A[random_pts[0]][0])
    c = A[random_pts[0]][1] - m*A[random_pts[0]][0]
    point = 0
    x1 = A[random_pts[0]][0]
    y1 = A[random_pts[0]][1]
    x2 = A[random_pts[1]][0]
    y2 = A[random_pts[1]][1]
    for j in range(len(A)):
        if j in random_pts:
            continue
        d = abs((x2-x1)*(y1-A[j][1])-(x1-A[j][0])*(y2-y1))/math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if d < 5:
            point = point + 1
    rnsc_points.append([random_pts[0],random_pts[1],point])
rnsc_points = np.array(rnsc_points)
sol = np.median(np.where(rnsc_points[:,2] == np.max(rnsc_points[:,2])))
x1 = A[rnsc_points[int(sol)][0]][0]
y1 = A[rnsc_points[int(sol)][0]][1]
x2 = A[rnsc_points[int(sol)][1]][0]
y2 = A[rnsc_points[int(sol)][1]][1]
charges_rnsc = []
for i in age:
    charges_rnsc.append(((y2-y1)/(x2-x1))*(i-x1) + y1)

plt.plot(age,charges_rnsc,color = 'yellow',label="RANSAC")
plt.legend(loc = "lower right")
plt.xlabel('age')
plt.ylabel('charges')
plt.title("Various curve fitting method")
plt.show()