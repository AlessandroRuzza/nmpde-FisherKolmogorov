import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("mesh_element_count.csv")

# Plot
plt.figure(figsize=(8,5))
plt.plot(df["mesh_preset"], df["elements"], marker='o', linestyle='-')
plt.xlabel("Mesh Preset")
plt.ylabel("Number of Elements")
plt.title("Number of Elements per Mesh Preset")
plt.grid(False)
plt.tight_layout()
plt.savefig("mesh_elements_plot.png")