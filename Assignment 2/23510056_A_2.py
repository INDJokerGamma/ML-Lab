
import numpy as np
import matplotlib.pyplot as plt


n = 10000

# A. Gaussian (B)
B = np.random.normal(loc=5, scale=2, size=n)

# B. Power Law (I)
def generate_powerlaw_samples(a, size):
    u = np.random.uniform(0, 1, size)
    return u ** (1 / a)

I = generate_powerlaw_samples(a=0.3, size=n)

# C. Geometric (H)
H = np.random.geometric(p=0.005, size=n)



plt.boxplot([B, I, H], labels=['Gaussian (B)', 'Power Law (I)', 'Geometric (H)'])
plt.title('Original Distributions - Box Plot')
plt.grid(True)
plt.show()


def normalize_max(x):
    return x / np.max(x)

def normalize_sum(x):
    return x / np.sum(x)

def normalize_zscore(x):
    return (x - np.mean(x)) / np.std(x)

def normalize_percentile(x):
    sorted_x = np.sort(x)
    ranks = np.searchsorted(sorted_x, x)
    return ranks / len(x)

def normalize_equal_median(x_list):
    medians = [np.median(x) for x in x_list]
    m1 = np.mean(medians)
    return [x * (m1 / m) for x, m in zip(x_list, medians)]

def quantile_normalize(arrays):
    sorted_arrays = np.array([np.sort(x) for x in arrays])
    avg_sorted = np.mean(sorted_arrays, axis=0)

    qn_arrays = []
    for x in arrays:
        ranks = np.argsort(np.argsort(x))
        qn_x = np.zeros_like(x)
        for i, rank in enumerate(ranks):
            qn_x[i] = avg_sorted[rank]
        qn_arrays.append(qn_x)
    return qn_arrays

def compare_normalization(original, normalized, label):
    plt.figure(figsize=(12, 4))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(original, bins=50, alpha=0.5, label='Original')
    plt.hist(normalized, bins=50, alpha=0.5, label='Normalized')
    plt.title(f'Histogram - {label}')
    plt.legend()

    # Box Plot
    plt.subplot(1, 2, 2)
    plt.boxplot([original, normalized], labels=['Original', 'Normalized'])
    plt.title(f'Box Plot - {label}')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Step 6: Apply and Compare Normalizations

# Divide by Max 
compare_normalization(B, normalize_max(B), 'Divide by Max - B')
compare_normalization(I, normalize_max(I), 'Divide by Max - I')
compare_normalization(H, normalize_max(H), 'Divide by Max - H')

# Divide by Sum 
compare_normalization(B, normalize_sum(B), 'Divide by Sum - B')
compare_normalization(I, normalize_sum(I), 'Divide by Sum - I')
compare_normalization(H, normalize_sum(H), 'Divide by Sum - H')

# Z-score 
compare_normalization(B, normalize_zscore(B), 'Z-score - B')
compare_normalization(I, normalize_zscore(I), 'Z-score - I')
compare_normalization(H, normalize_zscore(H), 'Z-score - H')

# Percentiles -compare_normalization(B, normalize_percentile(B), 'Percentile - B')
compare_normalization(I, normalize_percentile(I), 'Percentile - I')
compare_normalization(H, normalize_percentile(H), 'Percentile - H')

# Equal Median 
B_eq, I_eq, H_eq = normalize_equal_median([B, I, H])
plt.boxplot([B_eq, I_eq, H_eq], labels=['B eq-median', 'I eq-median', 'H eq-median'])
plt.title('Equalized Medians - Box Plot')
plt.grid(True)
plt.show()

# Quantile Normalization 
B_qn, I_qn, H_qn = quantile_normalize([B, I, H])
plt.boxplot([B_qn, I_qn, H_qn], labels=['B_qn', 'I_qn', 'H_qn'])
plt.title('Quantile Normalized - Box Plot')
plt.grid(True)
plt.show()
