#!/usr/bin/env python3
"""
generate_synthetic_data.py

Parameterizable synthetic data generator for AI-as-a-Service simulation.

Outputs:
  - physical_network.graphml
  - models.json
  - tasks.json
  - router_prefs.json

Usage example:
  python generate_synthetic_data.py --out_dir ./synth_data --num_routers 40 --num_servers 20 \
    --Tf 500 --lambda_per_router 0.5 --cpu_scale 1.5 --gpu_scale 1.0 --dirichlet_alpha 0.5
"""
from __future__ import annotations
import argparse, os, json, random
from typing import List, Dict, Any
from collections import defaultdict
import math

import networkx as nx
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="./synth_data")
    p.add_argument("--num_routers", type=int, default=40)
    p.add_argument("--num_servers", type=int, default=20)
    p.add_argument("--num_models", type=int, default=6)
    p.add_argument("--Tf", type=int, default=500, help="number of frame slots / time horizon")
    p.add_argument("--lambda_per_router", type=float, default=0.5, help="Poisson arrivals per router per short-step")
    p.add_argument("--seed", type=int, default=42)
    # resource scaling
    p.add_argument("--cpu_scale", type=float, default=1.0)
    p.add_argument("--gpu_scale", type=float, default=1.0)
    p.add_argument("--mem_scale", type=float, default=1.0)
    p.add_argument("--storage_scale", type=float, default=1.0)
    p.add_argument("--gpu_prob", type=float, default=0.4, help="probability a server has GPU")
    p.add_argument("--heterogeneity", type=float, default=0.3, help="0..1 controls resource heterogeneity")
    p.add_argument("--dirichlet_alpha", type=float, default=0.5, help="Dirichlet alpha for per-router model prefs")
    p.add_argument("--max_subtasks", type=int, default=3)
    p.add_argument("--min_subtasks", type=int, default=1)
    return p.parse_args()

# -----------------------------
# Utilities
# -----------------------------
def save_graphml_clean(G: nx.Graph, path: str):
    H = G.copy()
    # remove None attributes (GraphML cannot store None)
    for n, attrs in list(H.nodes(data=True)):
        for k in list(attrs.keys()):
            if attrs[k] is None:
                del H.nodes[n][k]
    for u, v, attrs in list(H.edges(data=True)):
        for k in list(attrs.keys()):
            if attrs[k] is None:
                del H.edges[u, v][k]
    for k in list(H.graph.keys()):
        if H.graph[k] is None:
            del H.graph[k]
    nx.write_graphml(H, path)

def write_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# -----------------------------
# Physical network generation
# -----------------------------
def make_physical_network(num_routers: int, num_servers: int, seed: int, cpu_scale: float = 1.0, gpu_scale: float = 1.0, mem_scale: float = 1.0, storage_scale: float = 1.0, gpu_prob: float = 0.4, heterogeneity: float = 0.3):
    rng = np.random.RandomState(seed)
    # routers: BA graph (power-law)
    m_param = max(1, min(5, max(1, int(max(1, num_routers * 0.03)))))
    if num_routers <= 2:
        R = nx.path_graph(num_routers)
    else:
        R = nx.barabasi_albert_graph(n=num_routers, m=m_param, seed=seed)
    G = nx.Graph()
    # add routers
    for r in R.nodes():
        G.add_node(f"router_{r}", node_type="router", router_id=int(r))
    # router edges with latency/bandwidth
    for (u, v) in R.edges():
        latency = float(rng.uniform(0.5, 5.0))  # ms
        bw = float(rng.uniform(100.0, 1000.0))  # Mbps
        G.add_edge(f"router_{u}", f"router_{v}", latency=latency, bandwidth=bw)
    # add servers and attach to routers
    for s in range(num_servers):
        sid = f"server_{s}"
        # sample base resource from categorical choices then apply scales and heterogeneity noise
        cpu_base_choices = [4, 8, 16, 32]
        cpu_probs = [0.4, 0.35, 0.2, 0.05]
        cpu_base = int(rng.choice(cpu_base_choices, p=cpu_probs))
        cpu = max(1, int(round(cpu_base * cpu_scale * (1.0 + rng.normal(0, heterogeneity)))))
        # GPU: decide whether has GPU, then sample count
        has_gpu = rng.rand() < gpu_prob
        if has_gpu:
            gpu_base_choices = [1, 1, 2]  # mostly 1, sometimes 2
            gpu = int(max(0, int(rng.choice(gpu_base_choices)) * max(1, int(round(gpu_scale)))))
        else:
            gpu = 0
        mem_base_choices = [32, 64, 128]
        mem_probs = [0.5, 0.35, 0.15]
        mem_base = int(rng.choice(mem_base_choices, p=mem_probs))
        mem_gb = max(4, int(round(mem_base * mem_scale * (1.0 + rng.normal(0, heterogeneity)))))
        storage_base_choices = [500, 1000, 2000]
        storage_base = int(rng.choice(storage_base_choices, p=[0.5, 0.35, 0.15]))
        storage_gb = max(50, int(round(storage_base * storage_scale * (1.0 + rng.normal(0, heterogeneity)))))
        # attach to 1..2 routers
        attach_k = rng.choice([1,1,2], p=[0.6,0.3,0.1])
        attach_k = int(min(max(1, attach_k), num_routers))
        routers = list(R.nodes())
        attach_to = list(rng.choice(routers, size=attach_k, replace=False))
        # provide a location coordinate for clustering (for overlay)
        loc_x = float(s % int(math.sqrt(max(1, num_servers))) + rng.uniform(0, 0.9))
        loc_y = float(s // int(math.sqrt(max(1, num_servers))) + rng.uniform(0, 0.9))
        G.add_node(sid, node_type="server", server_id=s, cpu_cores=cpu, gpu_count=gpu, mem_gb=mem_gb, storage_gb=storage_gb, loc_x=float(loc_x),
loc_y=float(loc_y),)
        for r in attach_to:
            latency = float(rng.uniform(0.2, 2.0))  # ms
            bw = float(rng.uniform(200.0, 2000.0))
            G.add_edge(sid, f"router_{r}", latency=latency, bandwidth=bw)
    return G

# -----------------------------
# Models generation
# -----------------------------
def generate_models(num_models: int, seed: int):
    rng = np.random.RandomState(seed)
    types = ["detection", "classification", "segmentation", "pose", "tracking", "anomaly"]
    models = []
    for m in range(num_models):
        typ = types[m % len(types)]
        cpu_mean = float(round(rng.uniform(0.5, 4.0), 3))
        # GPU usage mean - choose few models requiring GPU
        gpu_mean = float(round(rng.choice([0.0, 0.1, 0.5, 1.0], p=[0.5,0.2,0.2,0.1]), 3))
        gpu_peak = float(round(max(0.0, gpu_mean + rng.uniform(0.0, 0.5)), 3))
        iops = int(rng.choice([50, 100, 200, 400], p=[0.45, 0.3, 0.2, 0.05]))
        acc_mean = float(round(rng.uniform(0.6, 0.98), 3))
        p99_latency = float(round(rng.uniform(20.0, 500.0), 3))
        mem_mb = int(rng.choice([200, 500, 1000, 2000], p=[0.45,0.35,0.15,0.05]))
        models.append({
            "id": m,
            "name": f"model_{m}",
            "type": typ,
            "cpu_mean": cpu_mean,
            "gpu_mean": gpu_mean,
            "gpu_peak": gpu_peak,
            "iops": iops,
            "mem_mb": mem_mb,
            "acc_mean": acc_mean,
            "p99_latency_ms": p99_latency
        })
    return models

# -----------------------------
# Tasks generation with per-router Dirichlet prefs
# -----------------------------
def generate_tasks(num_routers: int, Tf: int, lambda_per_router: float, models: List[Dict[str,Any]], seed: int, dirichlet_alpha: float=0.5, min_subtasks: int=1, max_subtasks: int=3):
    rng = np.random.RandomState(seed)
    M = len(models)
    # per-router Dirichlet preference
    pref_mat = rng.dirichlet([dirichlet_alpha] * M, size=num_routers)
    tasks = []
    tid = 0
    for router in range(num_routers):
        prefs = pref_mat[router]
        for t in range(Tf):
            # number arrivals
            n = rng.poisson(lambda_per_router)
            for _ in range(n):
                k_sub = int(rng.choice(list(range(min_subtasks, max_subtasks+1)), p=None))
                subtasks = []
                # to simulate heterogeneity within task, pick models per preference
                for si in range(k_sub):
                    m = int(rng.choice(range(M), p=prefs))
                    model = models[m]
                    # accuracy requirement (around model mean)
                    acc_req = float(round(max(0.1, min(0.999, rng.normal(model['acc_mean'] - 0.02, 0.03))), 3))
                    lat_req = float(round(max(1.0, rng.uniform(0.8 * model['p99_latency_ms'], 2.5 * model['p99_latency_ms'])), 3))
                    proc_time = float(round(max(1.0, rng.normal(model['p99_latency_ms'] * 0.6, model['p99_latency_ms'] * 0.3)), 3))
                    subtasks.append({
                        "subtask_id": si,
                        "model_id": m,
                        "type": model['type'],
                        "accuracy_req": acc_req,
                        "latency_req_ms": lat_req,
                        "proc_time_ms": proc_time
                    })
                duration = int(max(1, round(rng.exponential(scale=5.0))))
                tasks.append({
                    "task_id": tid,
                    "src_router": int(router),
                    "arrival_time": int(t),
                    "duration_time": int(duration),
                    "subtasks": subtasks
                })
                tid += 1
    return tasks, pref_mat

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)

    print("Generating physical network...")
    G = make_physical_network(num_routers=args.num_routers, num_servers=args.num_servers, seed=seed,
                              cpu_scale=args.cpu_scale, gpu_scale=args.gpu_scale, mem_scale=args.mem_scale, storage_scale=args.storage_scale, gpu_prob=args.gpu_prob, heterogeneity=args.heterogeneity)
    graph_path = os.path.join(args.out_dir, "physical_network.graphml")
    save_graphml_clean(G, graph_path)
    print("Wrote network to", graph_path)

    print("Generating models...")
    models = generate_models(args.num_models, seed + 1)
    models_path = os.path.join(args.out_dir, "models.json")
    write_json(models, models_path)
    print("Wrote models to", models_path)

    print("Generating tasks (this may take a bit)...")
    tasks, pref_mat = generate_tasks(args.num_routers, args.Tf, args.lambda_per_router, models, seed + 2, dirichlet_alpha=args.dirichlet_alpha, min_subtasks=args.min_subtasks, max_subtasks=args.max_subtasks)
    tasks_path = os.path.join(args.out_dir, "tasks.json")
    write_json(tasks, tasks_path)
    print("Wrote tasks to", tasks_path)

    # router preferences
    router_prefs = {i: list(map(float, pref_mat[i].tolist())) for i in range(len(pref_mat))}
    router_prefs_path = os.path.join(args.out_dir, "router_prefs.json")
    write_json(router_prefs, router_prefs_path)
    print("Wrote router_prefs to", router_prefs_path)

    print("Done. Summary:")
    print(f" routers: {args.num_routers}, servers: {args.num_servers}, models: {args.num_models}, tasks: {len(tasks)}")

if __name__ == "__main__":
    main()
