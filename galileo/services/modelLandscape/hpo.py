import csv

import numpy as np
from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV, HalvingGridSearchCV
)
import random
from joblib import Parallel, delayed
import multiprocessing

def _spec_to_values(spec, n_values: int = 8):
    """
    Convert one param spec from HPO_PARAM_DISTS into a list of concrete values
    suitable for scikit-learn's param_grid / param_distributions.

    - dict with {type, range}: converted to a list (or list+None for *_or_none)
    - list/tuple: returned as-is (as a list)
    - scalar: wrapped in a 1-element list
    """
    # dict-based spec
    if isinstance(spec, dict):
        typ = spec.get("type", "int")
        rng = spec.get("range")

        if rng is None:
            return []

        # Normalize range, just in case
        low, high = rng
        if high < low:
            low, high = high, low

        # Helper for ints
        def _int_values():
            width = high - low + 1
            if width <= n_values:
                # dense coverage of all integers
                return list(range(low, high + 1))
            # sample n_values distinct ints in [low, high]
            vals = np.linspace(low, high, num=n_values, dtype=int)
            return sorted(set(int(v) for v in vals))

        # Helper for floats
        def _float_values():
            # linspace for now; could be logspace if you prefer
            return list(np.linspace(float(low), float(high), num=n_values))

        if typ == "int":
            return _int_values()

        elif typ == "float":
            return _float_values()

        elif typ == "int_or_none":
            base = _int_values()
            # allow None as an option (e.g. max_depth=None)
            return base + [None]

        elif typ == "float_or_none":
            base = _float_values()
            return base + [None]

        elif typ == "mlp_hidden_layer_sizes":
            # For CV search we skip this param – it's too high-dimensional
            # and you already handle it in evolutionary search.
            return []

        else:
            # Unknown type: fall back to empty; caller will just skip this param
            return []

    # list/tuple: treat as a categorical grid directly
    if isinstance(spec, (list, tuple)):
        return list(spec)

    # scalar value
    return [spec]


def make_sklearn_search_space(param_dists: dict, n_values: int = 8) -> dict:
    """
    Turn HPO_PARAM_DISTS[algo_key] into a dict suitable for
    RandomizedSearchCV / GridSearchCV / HalvingGridSearchCV.

    Skips parameters that can't be reasonably converted (e.g. mlp_hidden_layer_sizes).
    """
    grid = {}
    for name, spec in param_dists.items():
        vals = _spec_to_values(spec, n_values=n_values)
        # Only keep if we got at least one value
        if vals:
            grid[name] = vals
    return grid


def run_local_hpo(model, hpo_strategy, param_dists, X_train, y_train, scoring, n_iter=20, cv=3):
    """
    Run local hyperparameter optimization using scikit-learn's CV search classes.

    `param_dists` is expected to be in HPO_PARAM_DISTS format, so we first
    convert it with make_sklearn_search_space().
    """
    # Convert your rich spec -> sklearn-compatible dict of lists
    search_space = make_sklearn_search_space(param_dists)

    if not search_space:
        raise ValueError(
            f"HPO search space is empty after conversion: {param_dists!r}"
        )

    if hpo_strategy == "random":
        # param_distributions can be dict of lists; RandomizedSearchCV samples from them
        search = RandomizedSearchCV(
            model,
            search_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
        )
    elif hpo_strategy == "grid":
        search = GridSearchCV(
            model,
            search_space,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
        )
    elif hpo_strategy == "halving":
        search = HalvingGridSearchCV(
            model,
            search_space,
            scoring=scoring,
            cv=cv,
        )
    else:
        raise ValueError(f"Unknown hpo_strategy: {hpo_strategy}")

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_



def local_evolutionary_search(
        model_factory,
        param_space,
        X, y,
        X_val=None, y_val=None,
        pop_size=8,
        n_gens=10,
        score_func=None,
        progress_cb=None,
        csv_path="evo_hpo_log.csv"
):
    """
    Evolutionary hyperparameter search with full candidate/provenance logging to CSV.
    Each row in the CSV records: id, generation, parents, <HPO params>, score
    """
    if score_func is None:
        from sklearn.metrics import accuracy_score
        score_func = accuracy_score

    keys = list(param_space)
    candidate_log = []
    candidate_id_counter = 0

    def next_cid():
        nonlocal candidate_id_counter
        candidate_id_counter += 1
        return str(candidate_id_counter)  # Use string for IDs (matches CSV & Graphviz)

    def sample_value(spec):
        if isinstance(spec, dict):
            typ = spec.get("type", "int")
            rng = spec.get("range", [0, 1])
            if typ == "int":
                return random.randint(rng[0], rng[1])
            elif typ == "float":
                return random.uniform(rng[0], rng[1])
            elif typ == "int_or_none":
                return None if random.random() < 0.2 else random.randint(rng[0], rng[1])
            elif typ == "float_or_none":
                return None if random.random() < 0.2 else random.uniform(rng[0], rng[1])
            elif typ == "mlp_hidden_layer_sizes":
                n_layers_rng = rng["n_layers"]
                units_rng = rng["units_per_layer"]
                n_layers = random.randint(n_layers_rng[0], n_layers_rng[1])
                arr = [random.randint(units_rng[0], units_rng[1]) for _ in range(n_layers)]
                return tuple(arr)
            else:
                raise ValueError(f"Unknown HPO type: {typ}")
        elif isinstance(spec, (list, tuple)):
            return random.choice(spec)
        else:
            return spec

    def make_candidate():
        return {k: sample_value(param_space[k]) for k in keys}

    def crossover_layersizes(l1, l2):
        # Crossover for two tuples (can be different length!)
        min_len = min(len(l1), len(l2))
        out = []
        for i in range(max(len(l1), len(l2))):
            if i < min_len:
                out.append(random.choice([l1[i], l2[i]]))
            else:
                # For "extra" layers, just inherit from longer parent
                if len(l1) > len(l2):
                    out.append(l1[i])
                else:
                    out.append(l2[i])
        return tuple(out)

    def mutate_layersizes(layers, units_rng, n_layers_rng):
        # Randomly add/remove a layer or tweak a few
        layers = list(layers)
        # Mutation: Add/remove layer with some probability
        if random.random() < 0.15 and len(layers) < n_layers_rng[1]:
            # Add a layer at random position
            layers.insert(random.randint(0, len(layers)), random.randint(units_rng[0], units_rng[1]))
        if random.random() < 0.15 and len(layers) > n_layers_rng[0]:
            # Remove a random layer
            del layers[random.randint(0, len(layers) - 1)]
        # Mutation: change unit size of a random layer
        if len(layers) > 0 and random.random() < 0.4:
            idx = random.randint(0, len(layers)-1)
            layers[idx] = random.randint(units_rng[0], units_rng[1])
        return tuple(layers)

    # Initial population (generation 0, no parents)
    population = []
    for _ in range(pop_size):
        cid = next_cid()
        cand = make_candidate()
        population.append({
            "id": cid,
            "params": cand,
            "generation": 0,
            "parents": [],
            "score": None
        })
        candidate_log.append({
            "id": cid,
            "generation": 0,
            "parents": "",
            **cand,
            "score": None
        })

    best_candidate, best_score = None, -float("inf")
    num_cores = multiprocessing.cpu_count()
    best_curve = []
    mean_curve = []

    # Initial progress (0 generations done)
    if progress_cb is not None:
        progress_cb({
            "generation": 0,
            "total_generations": n_gens,
            "progress": 0.0,
            "stage": "evolutionary_search",
            "best_curve": [],
            "mean_curve": []
        })

    def eval_candidate(params):
        model = model_factory(**params)
        model.fit(X, y)
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            score = score_func(y_val, y_pred)
        else:
            y_pred = model.predict(X)
            score = score_func(y, y_pred)
        return score

    for gen in range(n_gens):
        # Parallel evaluation
        scores = Parallel(n_jobs=num_cores)(
            delayed(eval_candidate)(cand["params"]) for cand in population
        )
        # Assign scores and log
        for i, score in enumerate(scores):
            population[i]["score"] = score
            # Update candidate_log for this candidate's score
            for entry in candidate_log:
                if entry["id"] == population[i]["id"]:
                    entry["score"] = score
                    break

        # Sort population by score (descending)
        population.sort(reverse=True, key=lambda x: x["score"])

        best_curve.append(population[0]["score"])
        mean_curve.append(np.mean([x["score"] for x in population]))

        if population[0]["score"] > best_score:
            best_score = population[0]["score"]
            best_candidate = population[0]["params"]

        elites = population[:max(2, pop_size // 2)]
        new_pop = elites.copy()

        # In your child generation:
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child_params = {}
            for k in keys:
                if k == "hidden_layer_sizes" and param_space[k]["type"] == "mlp_hidden_layer_sizes":
                    child_layers = crossover_layersizes(
                        p1["params"][k], p2["params"][k]
                    )
                    # Let’s say always mutate layers a bit
                    child_layers = mutate_layersizes(
                        child_layers,
                        param_space[k]["range"]["units_per_layer"],
                        param_space[k]["range"]["n_layers"]
                    )
                    child_params[k] = child_layers
                else:
                    val = random.choice([p1["params"][k], p2["params"][k]])
                    val = mutate_param(val, param_space[k], prob=0.5)
                    child_params[k] = val
            cid = next_cid()
            child = {
                "id": cid,
                "params": child_params,
                "generation": gen + 1,
                "parents": [p1["id"], p2["id"]],
                "score": None
            }
            new_pop.append(child)
            candidate_log.append({
                "id": cid,
                "generation": gen + 1,
                "parents": f"{p1['id']},{p2['id']}",
                **child_params,
                "score": None
            })

        population = new_pop

        # Progress reporting
        if progress_cb is not None:
            progress_cb({
                "generation": gen + 1,
                "total_generations": n_gens,
                "progress": (gen + 1) / n_gens,
                "stage": "evolutionary_search",
                "best_curve": best_curve[:],
                "mean_curve": mean_curve[:]
            })

    # Write log to CSV
    fieldnames = ["id", "generation", "parents"] + list(param_space.keys()) + ["score"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_log)

    # Return best found candidate and (optional) history curves
    return best_candidate, {"best_curve": best_curve, "mean_curve": mean_curve}

def mutate_param(val, spec, prob=0.5):
    if isinstance(spec, dict):
        typ = spec.get("type", "int")
        rng = spec.get("range", [0, 1])
        if random.random() > prob:
            return val  # no mutation
        if typ == "float":
            width = rng[1] - rng[0]
            noise = random.uniform(-0.1, 0.1) * width
            new_val = float(val) + noise
            new_val = max(rng[0], min(rng[1], new_val))
            return new_val
        elif typ == "int":
            delta = random.choice([-2, -1, 0, 1, 2])
            new_val = int(val) + delta
            new_val = max(rng[0], min(rng[1], new_val))
            return new_val
        elif typ == "float_or_none" or typ == "int_or_none":
            if random.random() < 0.1:
                return None
            else:
                if typ.startswith("float"):
                    width = rng[1] - rng[0]
                    noise = random.uniform(-0.1, 0.1) * width
                    new_val = float(val) + noise
                    new_val = max(rng[0], min(rng[1], new_val))
                    return new_val
                else:
                    delta = random.choice([-2, -1, 0, 1, 2])
                    new_val = int(val) + delta
                    new_val = max(rng[0], min(rng[1], new_val))
                    return new_val
        return val
    elif isinstance(spec, (list, tuple)):
        # Categorical: with prob, pick a different value
        if random.random() < prob:
            choices = [v for v in spec if v != val]
            if choices:
                return random.choice(choices)
        return val
    else:
        return val
