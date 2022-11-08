import pandas as pd
import numpy as np
from datasets import all_observations

sample_sizes = [25, 100, 1000]
permutations = 1000
repeats = 100

for i, observations in enumerate(all_observations):
    df = observations
    print(f'------------------ DATASET {i + 1} -----------------')
    couples = []
    probabilities = []
    for x in df.index:
        for y in df.columns:
            couples.append((x, y))
            probabilities.append(df.loc[x, y])
    # Create dataset
    for sample_size in sample_sizes:
        samples_i = np.random.choice(len(couples), size=sample_size, p=probabilities)
        samples = pd.DataFrame([couples[x] for x in samples_i], columns=['X', 'Y'])
        cont_table = pd.crosstab(index=samples['X'], columns=samples['Y'])
        cont_table.loc['Column_Total'] = cont_table.sum(numeric_only=True, axis=0)
        cont_table.loc[:, 'Row_Total'] = cont_table.sum(numeric_only=True, axis=1)
        prob_table = cont_table / sample_size
        # Compute X^2 statistic and the p-value
        pass
        # Permutation testing
        for repeat_num in range(repeats):
            for perm_num in range(permutations):
                # Compute X^2 statistic and the p-value
                pass

        # Plot the p-values distribution for all repeats
        pass

