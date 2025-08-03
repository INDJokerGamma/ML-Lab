import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median
from scipy.stats import gmean

def min_var_change_count(k, N):
    var_change_counts = []
    
    for _ in range(k):
        arr = np.random.randint(1, 100000, size=N)
        temp_min = float('inf')
        var_change_count = 0
        i = 0
        while i < len(arr):
            if temp_min > arr[i]:
                temp_min = arr[i]
                var_change_count += 1
            i += 1
        var_change_counts.append(var_change_count)
    
    # Plot histogram of var_change_counts
    plt.figure(figsize=(8, 4))
    plt.hist(var_change_counts, bins=30, edgecolor='black')
    plt.title(f"Histogram of var_change_count (N={N})")
    plt.xlabel("var_change_count")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Calculate statistics
    mean_val = mean(var_change_counts)
    median_val = median(var_change_counts)
    geo_mean_val = gmean(var_change_counts)
    
    print(f"N = {N} -> Mean: {mean_val:.2f}, Median: {median_val}, Geometric Mean: {geo_mean_val:.2f}")
    
    return mean_val

# Test for multiple values of N
Ns = [10, 20, 50, 100, 200, 500, 1000, 2000,
      5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

mean_var_counts = []

# Run the function for each N and collect mean_var_change_count
for N in Ns:
    print(f"Processing N={N} ...")
    mean_val = min_var_change_count(k=100, N=N)
    mean_var_counts.append(mean_val)

# Plot N vs mean_var_change_count
plt.figure(figsize=(10, 6))
plt.plot(Ns, mean_var_counts, marker='o', linestyle='-', color='blue')
plt.title("Array Size (N) vs Mean var_change_count")
plt.xlabel("Array Size (N)")
plt.ylabel("Mean var_change_count")
plt.grid(True)
plt.xscale("log")  # Log scale for better visualization
plt.tight_layout()
plt.show()
