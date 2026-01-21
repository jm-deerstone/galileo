import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import random
import csv
import os

# ---- 1. Load weather data ----
df = pd.read_csv("datasources/b44a3094-bb30-486e-afe2-eb13dcaf81ea/20250714T005835Z_Weather_Data_1980_2024(hourly).csv")
X = df.drop(columns=["temperature", "time"]).values
y = df["temperature"].values

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- 2. Brute-force landscape generation ----
alpha_grid = np.linspace(0.01, 10, 40)
l1r_grid = np.linspace(0.2, 0.8, 40)
landscape = np.zeros((len(alpha_grid), len(l1r_grid)))

for i, alpha in enumerate(alpha_grid):
    for j, l1r in enumerate(l1r_grid):
        model = ElasticNet(alpha=alpha, l1_ratio=l1r, random_state=42, max_iter=10000)
        model.fit(X_train, y_train)
        score = r2_score(y_val, model.predict(X_val))
        landscape[i, j] = score

# ---- 3. Evolutionary Search Logic ----
def mutate_param(val, spec, prob=0.5):
    typ = spec.get("type", "float")
    rng = spec.get("range", [0, 1])
    if random.random() > prob:
        return val
    if typ == "float":
        width = rng[1] - rng[0]
        noise = random.uniform(-0.1, 0.1) * width
        new_val = float(val) + noise
        new_val = max(rng[0], min(rng[1], new_val))
        return new_val
    else:
        return val

def sample_value(spec):
    typ = spec.get("type", "float")
    rng = spec.get("range", [0, 1])
    if typ == "float":
        return random.uniform(rng[0], rng[1])
    else:
        return spec

param_space = {
    "alpha": {"type": "float", "range": [0.01, 10]},
    "l1_ratio": {"type": "float", "range": [0.2, 0.8]},
}

def eval_candidate(params):
    model = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    score = r2_score(y_val, model.predict(X_val))
    return score

pop_size, n_gens = 5, 5
candidate_log = []
population = []
cid_counter = 0
def next_cid():
    global cid_counter
    cid_counter += 1
    return str(cid_counter)

# Initial population
for _ in range(pop_size):
    cid = next_cid()
    params = {k: sample_value(param_space[k]) for k in param_space}
    population.append({"id": cid, "params": params, "generation": 0, "parents": [], "score": None})
    candidate_log.append({**params, "id": cid, "generation": 0, "parents": "", "score": None})

edges = []

for gen in range(n_gens):
    for ind in population:
        if ind["score"] is None:
            ind["score"] = eval_candidate(ind["params"])
            for entry in candidate_log:
                if entry["id"] == ind["id"]:
                    entry["score"] = ind["score"]
                    break
    population.sort(key=lambda x: x["score"], reverse=True)
    elites = population[:max(2, pop_size//2)]
    new_pop = elites.copy()
    while len(new_pop) < pop_size:
        p1, p2 = random.sample(elites, 2)
        child_params = {}
        for k in param_space:
            val = random.choice([p1["params"][k], p2["params"][k]])
            val = mutate_param(val, param_space[k], prob=0.5)
            child_params[k] = val
        cid = next_cid()
        child = {"id": cid, "params": child_params, "generation": gen+1, "parents": [p1["id"], p2["id"]], "score": None}
        new_pop.append(child)
        candidate_log.append({**child_params, "id": cid, "generation": gen+1, "parents": f"{p1['id']},{p2['id']}", "score": None})
        edges.append((p1["id"], cid))
        edges.append((p2["id"], cid))
    population = new_pop

for ind in population:
    if ind["score"] is None:
        ind["score"] = eval_candidate(ind["params"])
        for entry in candidate_log:
            if entry["id"] == ind["id"]:
                entry["score"] = ind["score"]
                break

cand_df = pd.DataFrame(candidate_log)
cand_df.to_csv("evo_candidates_log_weather.csv", index=False)

# ---- 4. Plotting ----
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
A, L = np.meshgrid(alpha_grid, l1r_grid)
ax.plot_surface(A, L, landscape.T, rstride=1, cstride=1, cmap=cm.viridis, alpha=0.6, antialiased=True)

for _, row in cand_df.iterrows():
    ax.scatter(row["alpha"], row["l1_ratio"], row["score"], color="red", s=60, edgecolor="k")
    ax.text(row["alpha"], row["l1_ratio"], row["score"], f'{int(row["id"])}', color="black", fontsize=9, ha="center", va="bottom", weight="bold")

id_to_xyz = {str(row["id"]): (row["alpha"], row["l1_ratio"], row["score"]) for _, row in cand_df.iterrows()}
for _, row in cand_df.iterrows():
    if row["parents"]:
        for p in row["parents"].split(","):
            if p and p in id_to_xyz:
                x0, y0, z0 = id_to_xyz[p]
                x1, y1, z1 = row["alpha"], row["l1_ratio"], row["score"]
                ax.plot([x0, x1], [y0, y1], [z0, z1], color="orange", alpha=0.6, lw=1.2)

ax.set_xlabel("alpha")
ax.set_ylabel("l1_ratio")
ax.set_zlabel("R² score")
ax.set_title("ElasticNet Evolutionary Search vs. Full Landscape (Weather Data)")

plt.tight_layout()
plt.savefig("evo_landscape_weather.png", dpi=300)
plt.savefig("evo_landscape_weather.svg")
plt.show()

# --- TikZ/PGFPlots (LaTeX) export ---
try:
    import tikzplotlib
    tikzplotlib.save("evo_landscape_weather.tex")
    print("Saved TikZ/PGFPlots file as evo_landscape_weather.tex")
except ImportError:
    print("tikzplotlib not installed. To get a LaTeX PGFPlots file, install it via pip and re-run.")

print("Saved as: evo_landscape_weather.png, evo_landscape_weather.svg, (optional) evo_landscape_weather.tex")
print("Evolutionary candidate log: evo_candidates_log_weather.csv")
