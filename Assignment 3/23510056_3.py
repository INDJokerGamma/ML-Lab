import matplotlib.pyplot as plt
import numpy as np



np.random.seed(0)
n =1000;
data = np.random.normal(10, 8, n)

data = np.exp(data/5) 
#e^x => data^5 (5 is random value), not much change but push the larger values more apart and closess the smaller values
data

extra_outliner = np.array([100,150,200])
data = np.concatenate((data, extra_outliner))
print(data)

plt.boxplot(data)
plt.show()

# Z-Scaling
mean = np.mean(data)
sd = np.std(data)

z  = (data - mean) / sd
print(z)
plt.boxplot(z)
plt.show()


#Robust Scaling
median = np.median(data)
q1 = np.percentile(data, 25) #1st quartile
q2 = np.percentile(data, 75) #2nd quartile
iqr = q2-q1

robust = (data - median) / iqr
print(robust)
plt.boxplot(robust)
plt.show()

# Comapring Z-score and Robust Normalization
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.boxplot(data)
plt.title("Original Data")

plt.subplot(1, 3, 2)
plt.boxplot(z)
plt.title("Z-scaled Data")

plt.subplot(1, 3, 3)
plt.boxplot(robust)
plt.title("Robust-scaled Data")

plt.show()

# Getting the scales allignend
z_norm = ( z - np.min(z) ) / ( np.max(z) - np.min(z) )
robust_norm = ( robust - np.min(robust) ) / ( np.max(robust) - np.min(robust) )

plt.figure(figsize=(12,5))
plt.suptitle("Comparing Oiginal ,Z-score and Robust Normalization")

plt.subplot(1, 3, 1)
plt.boxplot(data)
plt.title("Original Data")

plt.subplot(1, 3, 2)
plt.boxplot(z_norm)
plt.title("Z-score")

plt.subplot(1, 3, 3)
plt.boxplot(robust_norm)
plt.title("Robust ")

plt.show()