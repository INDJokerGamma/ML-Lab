import numpy as np
import matplotlib.pyplot as plt 

from statistics import mean, median, mode
# library for geometric mmea, median and mode

from scipy.stats import gmean
# library for geometric mean 



# a function for keep the count of min value changes in random array 
def min_var_change_count(k, N):

    #this list will store the count of min value changes for each chaneg 
    all_counts = [] 

    for i in range(k):
        #generating an array of N random integers from 1 to 1000000
        arr = np.random.randint(1, 1000000, size=N)

        #this is temorary minimum value setted as infinity
        temp_min = float('inf')
        count = 0
        # 
        j = 0

        while j < len(arr):

            #If current value is smaller than previous min, update it,and increase the count 
            if temp_min > arr[j]:
                temp_min = arr[j]
                count += 1

            j = j+ 1

        # add to the list we created 
        all_counts.append(count)
    
      # Plotting the histogram to show distribution of min change counts
    plt.figure(figsize=(8, 4))
    plt.hist(all_counts, bins=30, edgecolor='black')
    plt.title(f"Histogram of VarChangeCount (N={N})")
    plt.xlabel("VarChangeCount")
    plt.ylabel("Frequency")
    plt.show()


     # Calculate statistics from the collected counts
    mean_val = mean(all_counts)
    median_val = median(all_counts)
    mode_val = mode(all_counts)
    gmean_val = gmean(all_counts)


    # Printing all statistics for this array size
    print(f"N = {N} -> Mean: {mean_val}, Median: {median_val}, Mode: {mode_val}, Geometric Mean: {gmean_val:.2f}")

    return mean_val
# the function is ended


Ns = [10, 20, 50, 100, 200, 500, 1000, 2000,
      5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]

Mean_Counts = []
# List to store mean var_change_counts for each N


for N in Ns:
    print(f"Processing N={N} ...")
    mean_val = min_var_change_count(k=100, N=N)
    Mean_Counts.append(mean_val)

# Plot final result: N vs mean_var_change_count
plt.figure(figsize=(10, 6))
plt.plot(Ns, Mean_Counts, marker='o', linestyle='-', color='blue')
plt.title("Array Size (N) vs Mean var_change_count")
plt.xlabel("Array Size (N)")
plt.ylabel("Mean var_change_count")
plt.xscale("log")
plt.show()
