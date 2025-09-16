import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
num_samples = 1000

x = np.random.normal(5, 3, num_samples)
y = 0.3*x + np.random.normal(2, 1, num_samples)

plt.scatter(x, y)
plt.show()


# Euclidean Distance
data = np.vstack((x, y)).T
mean_x = np.mean(x)
mean_y = np.mean(y)
euclideian_dist = np.sqrt((x-mean_x)**2 + (y - mean_y)**2)
threshold_euclidean = np.percentile(euclideian_dist, 95)
outliers_euclidean = data[euclideian_dist > threshold_euclidean]
plt.title('Euclidean Distance Outlier Detection')
plt.scatter(x, y, c=euclideian_dist)
plt.scatter(outliers_euclidean[:, 0], outliers_euclidean[:, 1], color='red')
plt.show()


# Mahalanobis Distance
mean_vector = np.array([mean_x, mean_y])
S = np.cov(data, rowvar=False)
S_inverse =  np.linalg.inv(S)

mahal_dist = []
for i in data:
    diff = i - mean_vector
    distance = np.sqrt(diff.T @ S_inverse @ diff)
    mahal_dist.append(distance)
mahal_dist = np.array(mahal_dist)

threshold_mahalanobis = np.percentile(mahal_dist, 95)
outliers_mahalanobis = data[mahal_dist > threshold_mahalanobis]

plt.title('Mahalanobis Distance Outlier Detection')
plt.scatter(x, y, c=mahal_dist)
plt.scatter(outliers_mahalanobis[:, 0], outliers_mahalanobis[:, 1], color='red')

plt.show()

# Common Outliers in both methods
common_outliers = [];
for i in range(num_samples):
    if euclideian_dist[i] > threshold_euclidean and mahal_dist[i] > threshold_mahalanobis:
        common_outliers.append(data[i])

common_outliers = np.array(common_outliers) 
print(common_outliers)
plt.title('Common Outliers Detected by Both Methods')
plt.scatter(x, y)

if len(common_outliers) > 0:
    plt.scatter(common_outliers[:, 0], common_outliers[:, 1], color='black')
    
plt.show()


# Observations
# ----------------------------
# 1. Euclidean distance assumes circular spread around the mean. 
#
# 2. Mahalanobis distance accounts for correlation between X and Y. The data is elongated  (elliptical).
#
# 3. Euclidean often marks points as outliers even if they lie along 
#
# 4. Mahalanobis focuses on points that deviate off the central line . 
#
# 5. Some points are common outliers under both methods, but many differ. 


def mahal_only(data, num_points=10, x_mean_close=0, y_mean_close=10):
    mean_vector = np.mean(data, axis=0)

    new_points = []
    for _ in range(num_points):
        new_x = mean_vector[0] + x_mean_close
        new_y = mean_vector[1] + y_mean_close
        new_points.append([new_x, new_y])

    return np.array(new_points)

new_outliers = mahal_only(data, num_points=20, x_mean_close=0, y_mean_close=3)
new_data = np.vstack((data, new_outliers))


plt.scatter(new_data[:,0], new_data[:,1] )
plt.scatter(outliers_mahalanobis[:, 0], outliers_mahalanobis[:, 1], color='red')
if len(new_outliers) > 0:
    plt.scatter(new_outliers[:,0], new_outliers[:,1], color='green')
plt.title("Outliers Detected by Mahalanobis but not Euclidean")
plt.show()


plt.scatter(new_data[:,0], new_data[:,1] )
plt.scatter(outliers_euclidean[:, 0], outliers_euclidean[:, 1], color='red')
if len(new_outliers) > 0:
    plt.scatter(new_outliers[:,0], new_outliers[:,1], color='green')
plt.title("Outliers Detected by Mahalanobis but not Euclidean")
plt.show()
