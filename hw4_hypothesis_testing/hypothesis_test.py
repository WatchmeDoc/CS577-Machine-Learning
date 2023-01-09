import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import all_observations
from scipy.special import gamma
from tqdm import tqdm

sample_sizes = [25, 100, 1000]
permutations = 100
repeats = 100
# np.random.seed(1)


def compute_p_value(t, df):
    return (t ** ((df - 2) / 2) * np.exp(-t / 2)) / (2 ** (df / 2) * gamma(df / 2))


def chi_square_statistic(samples):
    sample_size = len(samples)
    cont_table = pd.crosstab(index=samples["X"], columns=samples["Y"])
    prob_table = cont_table / sample_size
    col_total = prob_table.sum(numeric_only=True, axis=0)
    row_total = prob_table.sum(numeric_only=True, axis=1)

    # Compute X^2 statistic and the p-value
    expected_data = []
    for m in row_total.index:
        tmp = []
        for k in col_total.index:
            tmp.append(sample_size * row_total[m] * col_total[k])
        expected_data.append(tmp)
    expected_data = pd.DataFrame(expected_data)
    T = ((cont_table - expected_data) ** 2 / expected_data).to_numpy().sum()
    f = (len(row_total) - 1) * (len(col_total) - 1)
    return T, f


p_values = {}
for i, observations in enumerate(all_observations):
    df = observations
    print(f"------------------ DATASET {i + 1} -----------------")
    couples = []
    probabilities = []
    dataset_p_value = {}
    for x in df.index:
        for y in df.columns:
            couples.append((x, y))
            probabilities.append(df.loc[x, y])
    # Create dataset
    for sample_size in sample_sizes:
        samples_i = np.random.choice(len(couples), size=sample_size, p=probabilities)
        samples = pd.DataFrame([couples[x] for x in samples_i], columns=["X", "Y"])

        T_orig, f = chi_square_statistic(samples=samples)
        p_value_orig = compute_p_value(df=f, t=T_orig)
        print(p_value_orig)
        dataset_p_value[sample_size] = p_value_orig
        # Permutation testing
        repeat_pvalues = []
        for repeat_num in tqdm(range(repeats)):
            chi_stats_permutations = []
            samples_i = np.random.choice(
                len(couples), size=sample_size, p=probabilities
            )
            samples = pd.DataFrame([couples[x] for x in samples_i], columns=["X", "Y"])
            T_orig, _ = chi_square_statistic(samples=samples)
            for perm_num in range(permutations):
                # Compute X^2 statistic and the p-value
                samples["X"] = np.random.permutation(samples["X"].values)
                T, _ = chi_square_statistic(samples=samples)
                chi_stats_permutations.append(T)
            final_p_value = float(
                sum(map(lambda x: abs(x) >= abs(T_orig), chi_stats_permutations))
            ) / float(permutations)
            repeat_pvalues.append(final_p_value)

        # Plot the p-values distribution for all repeats
        plt.figure()
        plt.hist(repeat_pvalues)
        plt.xlabel("percentage of times |t_b| >= |t_o|")
        plt.ylabel("Occurances")
        plt.title(f"Dataset {i + 1}, sample_size = {sample_size}")
        plt.savefig(f"results/dataset_{i+1}_{sample_size}.png")
    p_values[i + 1] = dataset_p_value

print(p_values)
