import matplotlib.pyplot as plt
import numpy as np

pp = 1e-7
fpr = 0.1
tpr = 0.85

# Part a.
probability = (tpr * pp) / (tpr * pp + fpr * (1 - pp))
print(
    "The probability that a photon package was actually received given the detector reported a detection is ",
    probability,
)

# Part b.

num_photons = 100
N = 2000
energies = [10, 20, 30, 40]
num_energies = len(energies)
energy_sums = []


for _ in range(N):
    # randint from numpy returns random integers from the “discrete uniform” distribution
    rand_energies = [
        energies[index]
        for index in np.random.randint(low=0, high=num_energies, size=num_photons)
    ]
    energy_sums.append(sum(rand_energies))

plt.figure()
plt.hist(energy_sums)
plt.xlabel("Total Package Energy")
plt.ylabel("Occurences")
plt.show()


# Part c.
mu = 1e-7
sigma = 9e-8
num_values = 10000

pdf = np.random.normal(loc=mu, scale=sigma, size=num_values)
results = []

for sample in pdf:
    if sample < 0:
        continue
    probability = (tpr * sample) / (tpr * sample + fpr * (1 - sample))
    results.append(probability)


plt.figure()
plt.hist(results)
plt.xlabel("Package Reception Probability")
plt.ylabel("Occurences")
plt.show()
