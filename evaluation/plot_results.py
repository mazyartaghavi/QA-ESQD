import json
import matplotlib.pyplot as plt

with open("results/performance.json") as f:
    data = json.load(f)

labels = list(data.keys())
means = [sum(v)/len(v) for v in data.values()]
stds = [ (sum((x - m)**2 for x in v)/len(v))**0.5
         for v, m in zip(data.values(), means)]

plt.bar(labels, means, yerr=stds, capsize=6)
plt.ylabel("Average Episodic Reward")
plt.title("Performance Comparison")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("results/performance_comparison.pdf")
plt.show()
