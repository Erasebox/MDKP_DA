#!/usr/bin/env python3
# run_cucb_overlay_da.py
# CUCB + SDCCP-overlay + DA simulation runner (frame-driven, steps_per_frame inner steps)

from __future__ import annotations
import argparse, os, sys, json, math, time, random
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------
# Helpers: load data
# --------------------------
def load_models(models_path):
    with open(models_path, 'r', encoding='utf-8') as f:
        models = json.load(f)
    # normalize to dict by id
    if isinstance(models, list):
        return {m['id']: m for m in models}
    elif isinstance(models, dict):
        return models
    else:
        raise RuntimeError("models.json format not recognized")

def load_tasks(tasks_path):
    with open(tasks_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    # group by arrival_time
    by_time = defaultdict(list)
    for t in tasks:
        by_time[int(t['arrival_time'])].append(t)
    return by_time

def load_graph(graphml_path):
    G = nx.read_graphml(graphml_path)
    # networkx reads attributes as strings sometimes; try to convert numeric strings to float/int where appropriate
    for n, d in list(G.nodes(data=True)):
        for k,v in list(d.items()):
            if isinstance(v, str):
                if v.isdigit():
                    G.nodes[n][k] = int(v)
                else:
                    try:
                        G.nodes[n][k] = float(v)
                    except Exception:
                        pass
    for u,v,d in list(G.edges(data=True)):
        for k,val in list(d.items()):
            if isinstance(val, str):
                if val.isdigit():
                    G.edges[u,v][k] = int(val)
                else:
                    try:
                        G.edges[u,v][k] = float(val)
                    except Exception:
                        pass
    return G

# --------------------------
# Overlay: SDCCP-like recursive grouping (returns overlay dict)
# --------------------------
def _build_base_adj(b, base_type='cycle'):
    adj = {i: [] for i in range(b)}
    if base_type == 'cycle':
        for i in range(b):
            adj[i].append((i+1)%b); adj[i].append((i-1)%b)
    elif base_type == 'path':
        for i in range(b-1):
            adj[i].append(i+1); adj[i+1].append(i)
    else:
        for i in range(b):
            adj[i] = [j for j in range(b) if j!=i]
    for k in list(adj.keys()):
        adj[k] = sorted(set(adj[k]))
    return adj

def _cluster_indices(indices, b, server_profiles, use_coord=True):
    # simple coordinate split: requires server_profiles[s]['location'] present
    if len(indices) == 0:
        return [[] for _ in range(b)]
    if not use_coord or any('location' not in server_profiles.get(s, {}) for s in indices):
        # fallback: round-robin
        clusters = [[] for _ in range(b)]
        for i, s in enumerate(indices):
            clusters[i % b].append(s)
        return clusters
    coords = [(s, server_profiles[s]['location'][0], server_profiles[s]['location'][1]) for s in indices]
    xs = [c[1] for c in coords]; ys = [c[2] for c in coords]
    axis = 0 if np.var(xs) >= np.var(ys) else 1
    coords_sorted = sorted(coords, key=lambda it: it[axis+1])
    n = len(coords_sorted)
    sizes = [n//b] * b
    for i in range(n % b):
        sizes[i] += 1
    groups = []
    start = 0
    for sz in sizes:
        groups.append([coords_sorted[i][0] for i in range(start, start+sz)])
        start += sz
    return groups

def build_overlay_sdccp_servers(server_profiles, delay_mat=None, base_b=3, base_graph_type='cycle', use_coord=True):
    # server_profiles: dict s -> {'location':(x,y)}
    S = len(server_profiles)
    if S == 0:
        return {'nodes':[], 'adj': {}, 'server_to_node': {}}
    # compute k depth so base_b^k >= S
    k = 0
    while (base_b ** k) < max(1, S):
        k += 1
    # lexicographic tuples
    tuples = list(__import__('itertools').product(range(base_b), repeat=k))
    V = len(tuples)
    # build leaf cells by recursive partition: simply split server list into V groups using recursive clustering
    def recursive_group(indices, depth):
        if depth == 0:
            return [indices]
        groups = _cluster_indices(indices, base_b, server_profiles, use_coord=use_coord)
        out = []
        for g in groups:
            sub = recursive_group(g, depth-1)
            out.extend(sub)
        return out
    server_indices = list(server_profiles.keys())
    leaf_cells = recursive_group(server_indices, k)
    # pad leaf_cells to V
    if len(leaf_cells) < V:
        leaf_cells += [[] for _ in range(V - len(leaf_cells))]
    # adjacency: Cartesian product of base graph
    base_adj = _build_base_adj(base_b, base_graph_type)
    tuple_to_idx = {tpl: i for i, tpl in enumerate(tuples)}
    adj = {i: set() for i in range(V)}
    for i, tpl in enumerate(tuples):
        for dim in range(k):
            for nb in base_adj[tpl[dim]]:
                nb_tpl = list(tpl); nb_tpl[dim] = nb; nb_tpl = tuple(nb_tpl)
                adj[i].add(tuple_to_idx[nb_tpl])
    adj = {i: sorted(list(neis)) for i, neis in adj.items()}
    nodes = []
    server_to_node = {}
    for nid in range(V):
        cell = leaf_cells[nid]
        rep = None
        if len(cell) > 0:
            rep = min(cell)
            # pick server representative
            server_to_node.update({s: nid for s in cell})
        nodes.append({'id': nid, 'rep': rep, 'members': cell})
    overlay = {'nodes': nodes, 'adj': adj, 'server_to_node': server_to_node, 'dims': (base_b, k)}
    return overlay

# --------------------------
# Utility: shortest-path latency from router to server
# --------------------------
def build_router_server_delay(G):
    # G includes router_* and server_* nodes; edges have 'latency' attribute (ms)
    # precompute shortest path latency between every router and server
    routers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'router' or str(n).startswith('router_')]
    servers = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'server' or str(n).startswith('server_')]
    # use Dijkstra using 'latency' weight
    delay = {}
    for r in routers:
        lengths = nx.single_source_dijkstra_path_length(G, r, weight='latency')
        for s in servers:
            delay[(r, s)] = float(lengths.get(s, float('inf')))
    return delay, routers, servers

# --------------------------
# Matching: build candidate list and DA
# --------------------------
def build_task_candidates(tasks, instances, G, delay_map, overlay, server_profiles, models, overlay_hop_penalty=0.5, max_cands=50):
    # tasks: list of task dicts (each has subtasks list)
    # instances: dict inst_id -> {'server': s, 'model': m, 'capacity':c}
    # return prefs: dict tid -> list of inst_ids sorted by effective delay
    prefs = {}
    # precompute inst nodes mapping to overlay node
    inst_to_node = {}
    for inst_id, inst in instances.items():
        s = inst['server']
        inst_to_node[inst_id] = overlay['server_to_node'].get(str(s), overlay['server_to_node'].get(s, None))
    # precompute BFS hops on overlay adjacency for nodes that exist
    adj = overlay['adj']
    hops_cache = {}
    def hops_from(n):
        if n in hops_cache:
            return hops_cache[n]
        dist = {n:0}
        q = [n]
        head=0
        while head < len(q):
            u = q[head]; head+=1
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u]+1
                    q.append(v)
        hops_cache[n]=dist
        return dist
    for t in tasks:
        # flatten by subtask: for this simulation we consider whole task accepted if any subtask matched?
        # But per your scenario, each subtask must be matched; here we will build prefs per subtask
        # For simplicity produce prefs for whole task by considering first subtask type (most common)
        subt = t.get('subtasks', [])
        if len(subt) == 0:
            prefs[t['task_id']] = []
            continue
        st = subt[0]
        typ = int(st['model_id'])
        acceptable = []
        src_router = f"router_{t['src_router']}" if not isinstance(t['src_router'], str) else t['src_router']
        for inst_id, inst in instances.items():
            if inst['model'] != typ:
                continue
            s = inst['server']
            # compute phys delay
            key = (src_router, str(s)) if (src_router, str(s)) in delay_map else (src_router, s)
            phys_delay = delay_map.get(key, float('inf'))
            # overlay hops
            src_node = overlay['server_to_node'].get(str(t['src_router']), overlay['server_to_node'].get(t['src_router'], None))
            inst_node = inst_to_node.get(inst_id, None)
            if src_node is None or inst_node is None:
                eff = phys_delay
            else:
                hops = hops_from(src_node)
                hopcount = hops.get(inst_node, 999999)
                eff = phys_delay + overlay_hop_penalty * float(hopcount)
            # check accuracy feasible
            model_acc = models[typ].get('acc_mean', 0.0)
            if model_acc + 1e-9 < float(st.get('accuracy_req', 0.0)):
                continue
            # latency check: total eff must be <= subtask latency_req
            if eff <= float(st.get('latency_req_ms', 1e9)) + 1e-9:
                acceptable.append((inst_id, eff))
        acceptable.sort(key=lambda x: x[1])
        prefs[t['task_id']] = [inst for inst, _ in acceptable[:max_cands]]
    return prefs

def deferred_acceptance(tasks, instances, prefs):
    # classical DA: tasks propose to instances
    unmatched = set([t['task_id'] for t in tasks])
    proposal_idx = {t['task_id']: 0 for t in tasks}
    accepted = {j: set() for j in instances.keys()}
    match = {t['task_id']: None for t in tasks}
    while True:
        proposals = defaultdict(list)
        any_prop = False
        for tid in list(unmatched):
            plist = prefs.get(tid, [])
            idx = proposal_idx[tid]
            if idx < len(plist):
                j = plist[idx]
                proposals[j].append(tid)
                proposal_idx[tid] += 1
                any_prop = True
            else:
                unmatched.discard(tid)
        if not any_prop:
            break
        for j, proposers in proposals.items():
            cap = instances[j]['capacity']
            cur = accepted[j]
            pool = list(cur) + proposers
            # simple ranking: prefer tasks in ascending order of task id (could be by better metric)
            pool_sorted = pool  # no special ranking among tasks here
            keep = set(pool_sorted[:cap])
            rejected = set(pool_sorted[cap:])
            accepted[j] = keep
            for tid in keep:
                match[tid] = j
                if tid in unmatched: unmatched.discard(tid)
            for tid in rejected:
                match[tid] = None
                if tid not in unmatched: unmatched.add(tid)
    return match

# --------------------------
# CUCB: combinatorial semi-bandit (simple implementation)
# --------------------------
class CombinatorialUCB:
    def __init__(self, server_profiles, models, alpha=1.0, k_select=None):
        self.server_profiles = server_profiles
        self.models = models
        self.alpha = float(alpha)
        self.k_select = k_select
        # build arms list (server_id, model_id)
        servers = list(server_profiles.keys())
        models_ids = list(models.keys())
        self.arms = [(s, m) for s in servers for m in models_ids]
        self.counts = {arm: 0 for arm in self.arms}
        self.sums = {arm: 0.0 for arm in self.arms}
        self.t = 1
    def arm_resource(self, arm):
        s, m = arm
        mp = self.models[m]
        return {'cpu': float(mp.get('cpu_mean', 1.0)), 'gpu': float(mp.get('gpu_mean', 0.0)), 'stor': float(mp.get('mem_mb', 0.0))}
    def server_caps_init(self):
        caps = {}
        for s, prof in self.server_profiles.items():
            caps[s] = {'cpu': float(prof.get('cpu_cores', 0)), 'gpu': float(prof.get('gpu_count', 0)), 'stor': float(prof.get('storage_gb', 0))}
        return caps
    def ucb_values(self):
        vals = {}
        for arm in self.arms:
            n = self.counts[arm]
            mean = (self.sums[arm] / n) if n > 0 else 0.0
            bonus = self.alpha * math.sqrt(math.log(max(1, self.t)) / (1 + n))
            vals[arm] = mean + bonus
        return vals
    def select(self):
        vals = self.ucb_values()
        arm_res = {arm: self.arm_resource(arm) for arm in self.arms}
        caps = self.server_caps_init()
        chosen = greedy_pack(vals, arm_res, caps, k_select=self.k_select)
        return chosen
    def update(self, chosen, observed_rewards):
        for arm in chosen:
            r = float(observed_rewards.get(arm, 0.0))
            self.counts[arm] += 1
            self.sums[arm] += r
        self.t += 1

def composite_cost(res):
    return res.get('cpu',0.0) + 2.0 * res.get('gpu',0.0) + 0.001 * res.get('stor',0.0)

def greedy_pack(vals, arm_res, server_caps, k_select=None):
    # defensive normalization for server_caps and arm_res; server ids may be strings
    dims = ['cpu','gpu','stor']
    caps = {s: {d: float(server_caps[s].get(d, 0.0)) for d in dims} for s in server_caps.keys()}
    arm_res_norm = {arm: {d: float(arm_res.get(arm, {}).get(d, 0.0)) for d in dims} for arm in vals.keys()}
    items = sorted(list(vals.items()), key=lambda iv: iv[1] / max(1e-9, composite_cost(arm_res_norm.get(iv[0], {}))), reverse=True)
    chosen = []
    deployed = 0
    if k_select is None:
        k_select = len(items)
    for arm, v in items:
        if len(chosen) >= k_select:
            break
        if not (isinstance(arm, tuple) and len(arm) == 2):
            continue
        s, m = arm
        if s not in caps:
            continue
        need = arm_res_norm.get(arm, {d:0.0 for d in dims})
        feasible = True
        for d in dims:
            if caps[s].get(d, 0.0) + 1e-9 < need.get(d, 0.0):
                feasible = False; break
        if feasible:
            chosen.append(arm)
            for d in dims:
                caps[s][d] -= need.get(d, 0.0)
            deployed += 1
    return chosen

# --------------------------
# Simulation driver
# --------------------------
def run_one_seed(graphml, models_json, tasks_json, out_dir, seed=0, frames=200, steps_per_frame=5, base_b=3, alpha=1.0, k_select=None, overlay_hop_penalty=0.5, verbose=False):
    random.seed(seed); np.random.seed(seed)
    G = load_graph(graphml)
    models = load_models(models_json)
    tasks_by_time = load_tasks(tasks_json)
    # server_profiles: map server_id (string) -> attrs
    server_profiles = {}
    for n, d in G.nodes(data=True):
        if d.get('node_type') == 'server' or str(n).startswith('server_'):
            # convert node key to consistent type (string)
            server_profiles[str(n)] = d.copy()
            # ensure location exists; if not, create grid
            if 'location' not in server_profiles[str(n)]:
                # try to create from server_id index
                try:
                    sid = int(d.get('server_id', ''.join([c for c in str(n) if c.isdigit()])))
                    server_profiles[str(n)]['location'] = (sid % 10, sid // 10)
                except Exception:
                    server_profiles[str(n)]['location'] = (random.random()*10, random.random()*10)
    # build router-server delay map
    delay_map, routers, servers = build_router_server_delay(G)
    # normalize model keys to int
    models_int = {int(k): v for k,v in models.items()}
    cu = CombinatorialUCB(server_profiles, models_int, alpha=alpha, k_select=k_select)
    # results arrays per frame
    per_frame_list = []
    per_selected_list = []
    cumulative = 0
    # prepare output dir
    os.makedirs(out_dir, exist_ok=True)
    # run frames: note tasks_by_time keys are arrival_time within original Tf
    for f in range(frames):
        # upper-layer select once per frame
        chosen = cu.select()
        # transform chosen into instances dict indexed 0..len-1 with server id as stored strings
        instances = {}
        for idx, arm in enumerate(chosen):
            s, m = arm
            s_str = str(s)
            instances[idx] = {'server': s_str, 'model': int(m), 'capacity': 1}
        # get tasks that arrive at time f
        arrivals = tasks_by_time.get(f, [])
        # partition arrivals into steps_per_frame chunks (evenly; last chunk may be shorter)
        if len(arrivals) == 0:
            chunks = [[] for _ in range(steps_per_frame)]
        else:
            shuffled = list(arrivals); random.shuffle(shuffled)
            chunks = [[] for _ in range(steps_per_frame)]
            for i, t in enumerate(shuffled):
                chunks[i % steps_per_frame].append(t)
        frame_rewards = defaultdict(int)
        frame_accepted = 0
        # overlay built once per frame (depends only on instances / server_profiles)
        overlay = build_overlay_sdccp_servers(server_profiles, base_b=base_b, base_graph_type='cycle', use_coord=True)
        for step in range(steps_per_frame):
            tasks_step = chunks[step]
            prefs = build_task_candidates(tasks_step, instances, G, delay_map, overlay, server_profiles, models_int, overlay_hop_penalty=overlay_hop_penalty)
            matching = deferred_acceptance(tasks_step, instances, prefs)
            # count accepted and per-arm reward
            for tid, inst in matching.items():
                if inst is not None:
                    arm = chosen[inst] if inst < len(chosen) else None
                    if arm is not None:
                        frame_rewards[arm] += 1
                        frame_accepted += 1
        # update CUCB with frame-level semi-bandit feedback
        cu.update(chosen, frame_rewards)
        per_frame_list.append(frame_accepted)
        per_selected_list.append(len(chosen))
        cumulative += frame_accepted
        if verbose and f % 50 == 0:
            print(f"[seed {seed}] frame {f}: accepted={frame_accepted} cumulative={cumulative}")
    # save per-frame csv
    df = pd.DataFrame({'frame': list(range(frames)), 'accepted': per_frame_list, 'selected': per_selected_list})
    csv_path = os.path.join(out_dir, f"per_frame_seed{seed}.csv")
    df.to_csv(csv_path, index=False)
    return {'per_frame': per_frame_list, 'selected': per_selected_list, 'cumulative': cumulative, 'csv': csv_path}

# --------------------------
# Batch runner over seeds
# --------------------------
def batch_run(args):
    ts = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_out = os.path.join(args.out_dir, ts); os.makedirs(base_out, exist_ok=True)
    all_runs = []
    for r in range(args.runs):
        seed = args.seed + r
        outdir = os.path.join(base_out, f"seed{seed}")
        os.makedirs(outdir, exist_ok=True)
        print(f"Running seed {seed} ...")
        res = run_one_seed(args.graphml, args.models, args.tasks, outdir, seed=seed, frames=args.frames, steps_per_frame=args.steps_per_frame, base_b=args.base_b, alpha=args.ucb_alpha, k_select=args.k_select, overlay_hop_penalty=args.overlay_hop_penalty, verbose=args.verbose)
        all_runs.append(res)
    # aggregate per-frame across runs
    mats = [r['per_frame'] for r in all_runs]
    maxlen = max(len(m) for m in mats)
    matpad = np.array([np.pad(m, (0, maxlen - len(m)), 'constant', constant_values=0) for m in mats])
    mean_curve = matpad.mean(axis=0)
    std_curve = matpad.std(axis=0)
    # save summary csv
    summary_rows = []
    for i, r in enumerate(all_runs):
        summary_rows.append({'seed': args.seed + i, 'cumulative': r['cumulative'], 'csv': r['csv']})
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(os.path.join(base_out, 'summary.csv'), index=False)
    # plotting
    plt.figure(figsize=(10,6))
    plt.plot(mean_curve, label='CUCB (mean accepted/frame)')
    plt.fill_between(range(len(mean_curve)), mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
    plt.xlabel("Frame"); plt.ylabel("Accepted tasks (per frame)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(base_out, 'accepted_per_frame_mean.png'), dpi=200)
    plt.close()
    # cumulative
    cum = np.cumsum(matpad, axis=1)
    mean_cum = cum.mean(axis=0); std_cum = cum.std(axis=0)
    plt.figure(figsize=(10,6))
    plt.plot(mean_cum, label='CUCB (mean cumulative accepted)')
    plt.fill_between(range(len(mean_cum)), mean_cum - std_cum, mean_cum + std_cum, alpha=0.15)
    plt.xlabel("Frame"); plt.ylabel("Cumulative accepted tasks")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(base_out, 'cumulative_accepted_mean.png'), dpi=200)
    plt.close()
    print("Batch finished. Outputs in", base_out)
    return base_out

# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graphml", type=str, required=True)
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--tasks", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./cucb_out")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--steps_per_frame", type=int, default=5)
    p.add_argument("--base_b", type=int, default=3)
    p.add_argument("--k_select", type=int, default=None)
    p.add_argument("--ucb_alpha", type=float, default=1.0)
    p.add_argument("--overlay_hop_penalty", type=float, default=0.5)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    batch_run(args)
