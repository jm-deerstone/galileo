import csv
from graphviz import Digraph

def read_candidates(csv_path):
    candidates = {}
    generations = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["id"]
            gen = int(row["generation"])
            info = {k: row[k] for k in row}
            info["params"] = {k: row[k] for k in row if k not in ["id", "generation", "parents", "score"]}
            info["parents"] = [p for p in row.get("parents", "").split(",") if p]
            candidates[cid] = info
            generations.setdefault(gen, []).append(cid)
    return candidates, generations

def plot_evo_tree(csv_path, output_pdf="evo_candidates_weather_tree.pdf", selected_id=None):
    candidates, generations = read_candidates(csv_path)
    dot = Digraph(format='pdf')
    dot.attr(rankdir='TB')

    # Calculate children for each node
    children = {cid: set() for cid in candidates}
    for cid, info in candidates.items():
        for parent in info["parents"]:
            if parent in children:
                children[parent].add(cid)

    # Add nodes with color-coding
    generation_nums = sorted(generations)
    for gen in generation_nums:
        with dot.subgraph() as s:
            s.attr(rank='same')
            for cid in sorted(generations[gen]):
                info = candidates[cid]
                label = f"ID {cid}\nGen {info['generation']}\n"
                for k, v in info["params"].items():
                    label += f"{k}={v}\n"
                label += f"Score: {info['score']}"

                node_color = "#e8f0fe"  # default light blue
                if cid == selected_id:
                    node_color = "#b5e7b0"  # green
                elif children[cid]:
                    node_color = "#b3d1ff"  # blue for internal nodes
                else:
                    node_color = "#ffd6d6"  # red for leaves

                s.node(cid, label, shape="box", style="filled", fillcolor=node_color)

    # Add edges (parent → child)
    for cid, info in candidates.items():
        for pid in info["parents"]:
            if pid and pid in candidates:
                dot.edge(pid, cid)

    # FORCE vertical layer order with invisible edges
    for i in range(len(generation_nums)-1):
        genA = generation_nums[i]
        genB = generation_nums[i+1]
        a_ids = sorted(generations[genA])
        b_ids = sorted(generations[genB])
        for a in a_ids:
            dot.edge(a, b_ids[0], style='invis')

    dot.render(output_pdf, view=False)
    print(f"Saved visualization as {output_pdf}")

if __name__ == "__main__":
    # CHANGE THIS TO YOUR WINNING/BEST CANDIDATE ID:
    plot_evo_tree("evo_candidates_log_weather.csv", selected_id="15")



