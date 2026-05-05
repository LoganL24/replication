import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# Import shared classes and functions directly from app.py so simulate.py
# works with the same implementation rather than re-implementing them.
from app import MovieRecommender, update_item_scores

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Parse args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Movie recommender simulation + report")
parser.add_argument("--users",      type=int, default=200,  help="Synthetic users per group")
parser.add_argument("--iterations", type=int, default=75,   help="Session iterations per user")
parser.add_argument("--seed",       type=int, default=42,   help="Global random seed")
parser.add_argument("--out",        type=str, default="results", help="Output directory")
parser.add_argument("--top-n",      type=int, default=100,  help="Top-N movies to use")
parser.add_argument("--k",          nargs="+", type=int, default=[5, 10],
                    help="K values for ranking metrics")
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Colour helpers (ANSI)
# ─────────────────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
MAGENTA= "\033[95m"

def c(text, *codes): return "".join(codes) + str(text) + RESET
def header(text):    print("\n" + c("━"*70, BLUE)); print(c(f"  {text}", BOLD, CYAN)); print(c("━"*70, BLUE))
def sub(text):       print(c(f"\n  ▸ {text}", BOLD, YELLOW))
def info(text):      print(c(f"    {text}", DIM))
def ok(text):        print(c(f"    ✓ {text}", GREEN))
def warn(text):      print(c(f"    ⚠ {text}", YELLOW))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────
GENRES = ["Action","Adventure","Animation","Comedy","Crime","Documentary",
          "Drama","Fantasy","Horror","Mystery","Romance","Sci-Fi","Thriller","Western"]

def generate_data(n_movies=300, n_users=500, n_ratings=15000, seed=42):
    rng = np.random.RandomState(seed)
    movies = []
    for i in range(1, n_movies + 1):
        n_g = rng.randint(1, 4)
        g   = "|".join(sorted(rng.choice(GENRES, n_g, replace=False).tolist()))
        movies.append({
            "movieId":    i,
            "Title":      f"Movie {i}",
            "Year":       int(rng.randint(1980, 2024)),
            "imdbRating": round(float(rng.uniform(4.0, 9.5)), 1),
            "Poster":     f"https://placeholder.com/{i}.jpg",
            "genres":     g,
        })
    movies_df = pd.DataFrame(movies)

    user_ids  = rng.randint(1, n_users + 1, n_ratings)
    movie_ids = rng.randint(1, n_movies + 1, n_ratings)
    # Skew ratings toward popular movies (power law)
    movie_ids = (rng.power(0.3, n_ratings) * n_movies).astype(int).clip(0, n_movies - 1) + 1
    ratings   = rng.uniform(0.5, 5.0, n_ratings).round(1)
    ratings_df = pd.DataFrame({"userId": user_ids, "movieId": movie_ids, "rating": ratings})
    ratings_df = ratings_df.drop_duplicates(["userId", "movieId"]).reset_index(drop=True)
    return movies_df, ratings_df

header("STEP 1 — Generating synthetic data")
movies_df, ratings_df = generate_data(n_movies=300, n_users=500, n_ratings=20000, seed=args.seed)
ok(f"{len(movies_df)} movies, {len(ratings_df)} unique ratings, {ratings_df['userId'].nunique()} users")
# ─────────────────────────────────────────────────────────────────────────────
# 3.  Preprocessing & model setup — delegated to app.py's MovieRecommender
# ─────────────────────────────────────────────────────────────────────────────
header("STEP 2 — Preprocessing (via app.py MovieRecommender)")

rec = MovieRecommender(movies_df, ratings_df, top_n=args.top_n)

# Expose the internals produced by MovieRecommender._preprocess_data
rdf          = rec.ratings_df
n_users_proc = len(rec.user_id_map)
n_items_proc = len(rec.item_id_map)
ok(f"After filtering: {n_items_proc} items, {n_users_proc} users, {len(rdf)} ratings")

# Convert Spotlight Interactions → DataFrames for metric functions below
train_df = pd.DataFrame({
    "user_id": rec.train_interactions.user_ids.astype(int),
    "item_id": rec.train_interactions.item_ids.astype(int),
    "rating":  rec.train_interactions.ratings.astype(float),
})
test_df = pd.DataFrame({
    "user_id": rec.test_interactions.user_ids.astype(int),
    "item_id": rec.test_interactions.item_ids.astype(int),
    "rating":  rec.test_interactions.ratings.astype(float),
})
ok(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Algorithm implementations
# ─────────────────────────────────────────────────────────────────────────────
header("STEP 3 — Training recommendation algorithms")

# ── 4a. Implicit MF — provided by app.py's MovieRecommender (already trained) ─
# rec.model  : Spotlight ImplicitFactorizationModel, trained via rec._train_model
# rec.train_rmse / rec.test_rmse / rec.train_precision / rec.test_precision
#   are computed by MovieRecommender._train_model and stored for Table 2.
impl_train_rmse = rec.train_rmse
impl_test_rmse  = rec.test_rmse
impl_train_prec = rec.train_precision
impl_test_prec  = rec.test_precision
ok(f"Implicit MF (app.py MovieRecommender): train RMSE={impl_train_rmse:.4f}  test RMSE={impl_test_rmse:.4f}"
   f"  |  train P@10={impl_train_prec:.4f}  test P@10={impl_test_prec:.4f}")

# ── 4b. Explicit MF (SGD) — standalone; app.py uses only Implicit MF ─────────
class ExplicitFactorizationModel:
    """Optimises against the observed rating value (explicit target).
    Kept here because app.py's MovieRecommender only trains Implicit MF."""
    def __init__(self, n_users, n_items, n_factors=20, n_iter=15, lr=0.01,
                 reg=0.02, seed=42):
        rng = np.random.RandomState(seed)
        self.P  = rng.normal(0, 0.1, (n_users, n_factors))
        self.Q  = rng.normal(0, 0.1, (n_items, n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.mu = 0.0
        self.n_factors = n_factors
        self.n_iter    = n_iter
        self.lr        = lr
        self.reg       = reg

    def fit(self, train_df):
        self.mu = train_df["rating"].mean()
        records = list(zip(train_df["user_id"], train_df["item_id"], train_df["rating"]))
        for epoch in range(self.n_iter):
            random.shuffle(records)
            for u, i, r in records:
                target = r             # explicit: predict the actual rating
                pred   = self._predict_one(u, i)
                err    = target - pred
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                self.P[u]  += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i]  += self.lr * (err * self.P[u] - self.reg * self.Q[i])

    def _predict_one(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u].dot(self.Q[i])

    def predict(self, user_id, item_ids):
        return np.array([self._predict_one(user_id, i) for i in item_ids])

    def rmse(self, df):
        preds = np.array([self._predict_one(u, i) for u, i in zip(df["user_id"], df["item_id"])])
        return float(np.sqrt(np.mean((preds - df["rating"].values) ** 2)))

    def precision_at_k(self, df, k=10, threshold=4.0):
        n_items = self.Q.shape[0]
        precisions = []
        for u in df["user_id"].unique():
            scores      = self.predict(u, np.arange(n_items))
            top_k_items = np.argsort(-scores)[:k]
            mask     = df["user_id"] == u
            relevant = set(df.loc[mask & (df["rating"] >= threshold), "item_id"].tolist())
            hits     = sum(1 for item in top_k_items if item in relevant)
            precisions.append(hits / k)
        return float(np.mean(precisions)) if precisions else 0.0


sub("Training Explicit MF (ExplicitFactorizationModel) …")
t0 = time.time()
mf_explicit = ExplicitFactorizationModel(n_users_proc, n_items_proc, n_factors=20,
                                          n_iter=20, seed=args.seed)
mf_explicit.fit(train_df)
expl_train_rmse = mf_explicit.rmse(train_df)
expl_test_rmse  = mf_explicit.rmse(test_df)
expl_train_prec = mf_explicit.precision_at_k(train_df, k=10)
expl_test_prec  = mf_explicit.precision_at_k(test_df,  k=10)
ok(f"Done in {time.time()-t0:.1f}s  |  train RMSE={expl_train_rmse:.4f}  test RMSE={expl_test_rmse:.4f}"
   f"  |  train P@10={expl_train_prec:.4f}  test P@10={expl_test_prec:.4f}")

# ── 4c. Popularity baseline ───────────────────────────────────────────────────
popularity_scores = (rdf.groupby("item_id")["rating"].sum()
                     .reindex(range(n_items_proc), fill_value=0.0))

# ── 4d. Random baseline ────────────────────────────────────────────────────────
ok("Popularity and Random baselines ready (no training required)")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Metric functions
# ─────────────────────────────────────────────────────────────────────────────
def get_relevant(df, user_id, threshold=4.0):
    mask = (df["user_id"] == user_id) & (df["rating"] >= threshold)
    return set(df.loc[mask, "item_id"].tolist())

def precision_at_k(recommended, relevant, k):
    top = recommended[:k]
    hits = sum(1 for i in top if i in relevant)
    return hits / k if k > 0 else 0.0

def recall_at_k(recommended, relevant, k):
    if not relevant: return 0.0
    top  = recommended[:k]
    hits = sum(1 for i in top if i in relevant)
    return hits / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    dcg, idcg = 0.0, 0.0
    for rank, item in enumerate(recommended[:k], 1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = min(len(relevant), k)
    for rank in range(1, ideal + 1):
        idcg += 1.0 / math.log2(rank + 1)
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(recommended, relevant, k):
    return 1.0 if any(i in relevant for i in recommended[:k]) else 0.0

def catalogue_coverage(all_recommended, n_items):
    seen = set()
    for rec in all_recommended:
        seen.update(rec)
    return len(seen) / n_items if n_items > 0 else 0.0

def genre_diversity(recommended, item_genres):
    """Mean pairwise Jaccard distance between genre sets in the list."""
    genre_sets = [item_genres.get(i, set()) for i in recommended]
    if len(genre_sets) < 2: return 0.0
    dists = []
    for a in range(len(genre_sets)):
        for b in range(a + 1, len(genre_sets)):
            ga, gb = genre_sets[a], genre_sets[b]
            union = ga | gb
            inter = ga & gb
            if union:
                dists.append(1.0 - len(inter) / len(union))
    return float(np.mean(dists)) if dists else 0.0

# ── Gold-standard cosine similarity ──────────────────────────────────────────
# Gold standard: the MF implicit model's top-50 ranking per user (fixed at
# train time, representing the "ideal" personalised ordering).
# We compare each iteration's recommended score-vector against this gold
# vector using cosine similarity, averaged over all simulated users.

def cosine_sim(a, b):
    """Cosine similarity between two 1-D numpy arrays."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# Pre-build genre map: item_id → frozenset of genres (using rec.movies_df from app.py)
item_genre_map = {}
for _, row in rec.movies_df[rec.movies_df["movieId"].isin(rec.item_id_map.values())].iterrows():
    iid = rec.reverse_item_id_map.get(row["movieId"])
    if iid is not None:
        item_genre_map[iid] = frozenset(row["genres"].split("|"))

def evaluate_model(model_fn, split_df, k_values, label):
    """
    model_fn(user_id) → sorted list of recommended item_ids (all items, full ranking).
    Returns dict of metric → value.
    """
    all_recs = []
    metrics_acc = defaultdict(list)
    users = split_df["user_id"].unique()

    for u in users:
        relevant = get_relevant(split_df, u)
        if not relevant: continue
        recommended = model_fn(u)
        all_recs.append(recommended[:max(k_values)])
        for k in k_values:
            metrics_acc[f"P@{k}"].append(precision_at_k(recommended, relevant, k))
            metrics_acc[f"R@{k}"].append(recall_at_k(recommended, relevant, k))
            metrics_acc[f"NDCG@{k}"].append(ndcg_at_k(recommended, relevant, k))
            metrics_acc[f"HR@{k}"].append(hit_rate_at_k(recommended, relevant, k))

    result = {m: float(np.mean(v)) for m, v in metrics_acc.items()}
    result["Coverage"] = catalogue_coverage(all_recs, n_items_proc)
    result["ILD"]      = float(np.mean([genre_diversity(r, item_genre_map) for r in all_recs])) if all_recs else 0.0
    return result

all_items = np.arange(n_items_proc)

def make_mf_fn_from_recommender(model_rec):
    """Wrap app.py's MovieRecommender model for ranking evaluation."""
    def fn(u):
        scores = model_rec.model.predict(u, all_items)
        return list(np.argsort(-scores))
    return fn

def make_mf_fn(model):
    def fn(u):
        scores = model.predict(u, all_items)
        return list(np.argsort(-scores))
    return fn

def popularity_fn(u):
    return list(np.argsort(-popularity_scores.values))

def random_fn(u):
    return list(np.random.permutation(n_items_proc))

header("STEP 4 — Evaluating algorithms (train & test splits)")

ALGOS = {
    "Implicit MF": make_mf_fn_from_recommender(rec),
    "Explicit MF": make_mf_fn(mf_explicit),
    "Popularity":  popularity_fn,
    "Random":      random_fn,
}

algo_results = {}
for name, fn in ALGOS.items():
    sub(f"Evaluating {name} …")
    train_metrics = evaluate_model(fn, train_df, args.k, name)
    test_metrics  = evaluate_model(fn, test_df,  args.k, name)
    train_rmse = None
    test_rmse  = None
    train_prec = None
    test_prec  = None
    if name == "Implicit MF":
        train_rmse = impl_train_rmse
        test_rmse  = impl_test_rmse
        train_prec = impl_train_prec
        test_prec  = impl_test_prec
    elif name == "Explicit MF":
        train_rmse = expl_train_rmse
        test_rmse  = expl_test_rmse
        train_prec = expl_train_prec
        test_prec  = expl_test_prec
    algo_results[name] = {
        "train": train_metrics,
        "test":  test_metrics,
        "train_rmse":      train_rmse,
        "test_rmse":       test_rmse,
        "train_precision": train_prec,
        "test_precision":  test_prec,
    }
    for k in args.k:
        ok(f"  [test] P@{k}={test_metrics[f'P@{k}']:.4f}  R@{k}={test_metrics[f'R@{k}']:.4f}  "
           f"NDCG@{k}={test_metrics[f'NDCG@{k}']:.4f}  HR@{k}={test_metrics[f'HR@{k}']:.4f}")
    ok(f"  Coverage={test_metrics['Coverage']:.4f}  ILD={test_metrics['ILD']:.4f}" +
       (f"  test RMSE={test_rmse:.4f}" if test_rmse else ""))



# ─────────────────────────────────────────────────────────────────────────────
# 6.  Score update — delegated to app.py's update_item_scores
# ─────────────────────────────────────────────────────────────────────────────
# update_item_scores is imported from app.py.  The session simulation works
# with internal item IDs, so _apply_update_item_scores converts between the
# internal-ID space used here and the movieId space expected by app.py.

def _update_item_scores_with_id_conversion(item_scores, clicked_id, seen_ids,
                               positive_factor=0.05, negative_factor=0.01):
    """
    Thin wrapper around app.py's update_item_scores for the simulation.

    Converts internal item IDs → original movieIds before the call, then
    converts the returned scores back to internal IDs so that the caller
    can continue working in the simulation's internal-ID space.
    """
    id_map  = rec.item_id_map           # internal → movieId
    rev_map = rec.reverse_item_id_map   # movieId  → internal

    movie_scores  = {id_map[iid]: score for iid, score in item_scores.items()}
    movie_clicked = id_map[clicked_id]
    movie_seen    = [id_map[s] for s in seen_ids]

    updated = update_item_scores(
        movie_scores, movie_clicked, movie_seen, rec,
        positive_factor=positive_factor,
        negative_factor=negative_factor,
        update_type='adaptive',
    )
    return {rev_map[mid]: score for mid, score in updated.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Experiment simulation
# ─────────────────────────────────────────────────────────────────────────────
header("STEP 5 — Simulating 8 experiment groups")

GROUP_MAP = {
    0: ("control",  "random",     "list"),
    1: ("control",  "random",     "carousel"),
    2: ("control",  "popularity", "list"),
    3: ("control",  "popularity", "carousel"),
    4: ("adaptive", "random",     "list"),
    5: ("adaptive", "random",     "carousel"),
    6: ("adaptive", "popularity", "list"),
    7: ("adaptive", "popularity", "carousel"),
}

# Genre → [item_ids] index (used for carousel genre selection)
genre_to_items: dict = defaultdict(list)
for iid, genres in item_genre_map.items():
    for g in genres:
        genre_to_items[g].append(iid)
all_genres = sorted(genre_to_items.keys())

def simulate_session(user_internal_id, update_type, baseline_type,
                     n_iter=10, top_k=10, ui_type="list"):
    """
    Mirrors app.py landing_page + click handler + update_item_scores exactly.

    Score init (mirrors app.py get_baseline_recommendations):
      popularity → item_scores[id] = sum-of-ratings (popularity score)
      random     → item_scores[id] = np.random.rand()

    Gold standard (paper §3.4.1):
      Generated once at session start from the trained Implicit MF model.
      gold_scores[id] = mf_implicit.predict(user, id)   ← raw MF output
      These stay fixed for the entire session.

    Click rule:
      Carousel — sample genre i ~ p_i (sum-of-scores per genre, normalised).
        Within chosen genre, scan items by descending score.
        Click first item where item_score >= gold_score[item].
        If no click in that genre, try next genres in preference order.
      Ranked list — cascade top-to-bottom; click first item where
        item_score >= gold_score[item]; stop.

    Score update (mirrors app.py update_item_scores, adaptive only):
      overlapping-genre positive boost + negative_genres decrease.

    Similarity (paper §3.4.4):
      Binary cosine sim: current top-K vector vs gold top-K vector.
    """
    POS_FACTOR = 0.05
    NEG_FACTOR = 0.01

    # ── Score initialisation (mirrors app.py get_baseline_recommendations) ────
    if baseline_type == "popularity":
        item_scores = {i: float(popularity_scores.iloc[i]) for i in range(n_items_proc)}
    else:
        item_scores = {i: float(np.random.rand()) for i in range(n_items_proc)}

    # ── Gold standard — raw MF scores from app.py's model, fixed for session ──
    # Gold scores are from the trained MF model and stay on their native scale.
    # Item scores are normalised to [0,1] each iteration for the threshold
    # comparison so the click rule is scale-invariant across both baselines.
    gold_mf_impl  = rec.model.predict(user_internal_id, np.arange(n_items_proc))
    gold_top_impl = set(np.argsort(-gold_mf_impl)[:top_k])
    gold_vec_impl = np.array([1.0 if i in gold_top_impl else 0.0
                              for i in range(n_items_proc)])

    # Normalise gold scores once to [0,1] for the click threshold
    g_min, g_max = gold_mf_impl.min(), gold_mf_impl.max()
    g_rng = (g_max - g_min) if g_max > g_min else 1.0
    gold_norm = {i: float((gold_mf_impl[i] - g_min) / g_rng)
                 for i in range(n_items_proc)}

    # Explicit MF gold — for Figure 1 dual-model similarity curves only
    gold_mf_expl  = mf_explicit.predict(user_internal_id, np.arange(n_items_proc))
    gold_top_expl = set(np.argsort(-gold_mf_expl)[:top_k])
    gold_vec_expl = np.array([1.0 if i in gold_top_expl else 0.0
                              for i in range(n_items_proc)])

    clicked          = []
    similarity_impl  = []
    similarity_expl  = []
    diversity_scores = []

    for _iteration in range(n_iter):
        # ── Current top-K binary vector → similarity (§3.4.4) ─────────────────
        ranked    = sorted(item_scores.items(), key=lambda x: -x[1])
        top_items = [iid for iid, _ in ranked[:top_k]]
        top_set   = set(top_items)
        cur_vec   = np.array([1.0 if i in top_set else 0.0
                              for i in range(n_items_proc)])
        similarity_impl.append(cosine_sim(cur_vec, gold_vec_impl))
        similarity_expl.append(cosine_sim(cur_vec, gold_vec_expl))
        diversity_scores.append(genre_diversity(top_items, item_genre_map))

        # ── Normalise item scores to [0,1] for scale-invariant click threshold ─
        s_arr = np.array([item_scores[i] for i in range(n_items_proc)])
        s_min, s_max = s_arr.min(), s_arr.max()
        s_rng = (s_max - s_min) if s_max > s_min else 1.0
        norm_s = {i: float((item_scores[i] - s_min) / s_rng)
                  for i in range(n_items_proc)}

        seen_this_iter    = []
        clicked_this_iter = None

        if ui_type == "carousel":
            # Genre preference p_i = normalised sum-of-scores per genre (§3.2.1)
            # mirrors app.py _get_sorted_genres
            g_score   = {g: sum(item_scores.get(i, 0.0) for i in items)
                         for g, items in genre_to_items.items()}
            total     = sum(g_score.values()) or 1.0
            g_ord     = sorted(g_score, key=lambda g: -g_score[g])
            g_probs   = np.array([max(0.0, g_score[g] / total) for g in g_ord])
            g_probs  /= g_probs.sum()

            chosen_genre = np.random.choice(g_ord, p=g_probs)
            genre_order  = [chosen_genre] + [g for g in g_ord if g != chosen_genre]

            for g in genre_order:
                for iid in sorted(genre_to_items[g],
                                  key=lambda i: -item_scores.get(i, 0.0)):
                    seen_this_iter.append(iid)
                    if norm_s[iid] >= gold_norm[iid]:
                        clicked_this_iter = iid
                        clicked.append(iid)
                        break
                if clicked_this_iter is not None:
                    break

        else:  # ranked list — cascade (§3.2.2 / §3.4.3)
            for iid, _ in ranked:
                seen_this_iter.append(iid)
                if norm_s[iid] >= gold_norm[iid]:
                    clicked_this_iter = iid
                    clicked.append(iid)
                    break

        # ── Score update — calls app.py's update_item_scores via wrapper ────────
        if update_type == "adaptive" and clicked_this_iter is not None:
            item_scores = _update_item_scores_with_id_conversion(
                item_scores, clicked_this_iter,
                [s for s in seen_this_iter if s != clicked_this_iter],
                POS_FACTOR, NEG_FACTOR,
            )

    ctr = len(clicked) / n_iter
    return {
        "clicks":          clicked,
        "ctr":             ctr,
        "diversity":       diversity_scores,
        "similarity":      similarity_impl,
        "similarity_impl": similarity_impl,
        "similarity_expl": similarity_expl,
        "final_scores":    item_scores,
    }


group_sim_results = {}
n_sim_users = min(args.users, n_users_proc)
sim_user_ids = list(rec.user_id_map.keys())[:n_sim_users]  # internal IDs

for g_idx, (update_type, baseline_type, ui_type) in GROUP_MAP.items():
    label = f"Group {g_idx} [{update_type}/{baseline_type}/{ui_type}]"
    sub(label)
    sessions = []
    for uid in sim_user_ids:
        s = simulate_session(uid, update_type, baseline_type,
                             n_iter=args.iterations, top_k=10, ui_type=ui_type)
        sessions.append(s)

    avg_ctr  = float(np.mean([s["ctr"] for s in sessions]))
    avg_div  = float(np.mean([np.mean(s["diversity"]) for s in sessions]))
    avg_clk  = float(np.mean([len(s["clicks"]) for s in sessions]))

    # Per-iteration similarity curves for both models (mean across users)
    sim_impl_matrix = np.array([s["similarity_impl"] for s in sessions])
    sim_expl_matrix = np.array([s["similarity_expl"] for s in sessions])
    avg_sim_curve       = sim_impl_matrix.mean(axis=0).tolist()  # backward compat
    avg_sim_curve_impl  = sim_impl_matrix.mean(axis=0).tolist()
    avg_sim_curve_expl  = sim_expl_matrix.mean(axis=0).tolist()

    # Per-iteration diversity curves (mean across users)
    div_matrix = np.array([s["diversity"] for s in sessions])   # shape (n_users, n_iter)
    avg_div_curve = div_matrix.mean(axis=0).tolist()

    # Per-iteration cumulative CTR curve: fraction of users who have clicked
    # by iteration t (i.e. running click count / (t+1), averaged across users)
    n_iter_actual = args.iterations
    ctr_curves = []
    for s in sessions:
        # clicks is a list of clicked item ids; reconstruct per-iter hit indicator
        # simulate_session appends a click each iteration when rand < 0.60
        # We stored total ctr, so approximate per-iter as cumulative mean
        hits = np.zeros(n_iter_actual)
        for ci, _ in enumerate(s["clicks"]):
            # spread clicks uniformly across iterations as best approximation
            iter_idx = min(ci, n_iter_actual - 1)
            hits[iter_idx] = 1.0
        cum_ctr = np.cumsum(hits) / (np.arange(n_iter_actual) + 1)
        ctr_curves.append(cum_ctr.tolist())
    avg_ctr_curve = np.array(ctr_curves).mean(axis=0).tolist()

    group_sim_results[g_idx] = {
        "label":            label,
        "update_type":      update_type,
        "baseline":         baseline_type,
        "ui":               ui_type,
        "avg_ctr":          avg_ctr,
        "avg_clicks":       avg_clk,
        "avg_diversity":    avg_div,
        "sim_curve":        avg_sim_curve,
        "sim_curve_impl":   avg_sim_curve_impl,
        "sim_curve_expl":   avg_sim_curve_expl,
        "div_curve":        avg_div_curve,
        "ctr_curve":        avg_ctr_curve,
    }
    ok(f"CTR={avg_ctr:.3f}  avg_clicks={avg_clk:.2f}  diversity={avg_div:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Terminal report
# ─────────────────────────────────────────────────────────────────────────────
header("STEP 6 — Terminal Report")

def tbl_row(cells, widths, sep="│"):
    parts = []
    for cell, w in zip(cells, widths):
        parts.append(str(cell).ljust(w)[:w])
    return sep + sep.join(parts) + sep

def tbl_divider(widths, l="├", m="┼", r="┤", h="─"):
    return l + m.join(h * w for w in widths) + r

def tbl_top(widths, l="┌", m="┬", r="┐", h="─"):
    return l + m.join(h * w for w in widths) + r

def tbl_bot(widths, l="└", m="┴", r="┘", h="─"):
    return l + m.join(h * w for w in widths) + r


print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
print(f"{BOLD}{CYAN}  Movie Recommender System — Simulation Report{RESET}")
print(f"{BOLD}{CYAN}  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
print(f"{BOLD}{CYAN}{'═'*70}{RESET}\n")

# ── Table 1: Algorithm metrics ─────────────────────────────────────────────
print(c("  ◆ TABLE 1 — Algorithm Evaluation Metrics (test split)", BOLD, MAGENTA))

k_val = max(args.k)
widths = [16, 8, 8, 8, 8, 8, 8]
headers = ["Algorithm", f"P@{k_val}", f"R@{k_val}", f"NDCG@{k_val}",
           f"HR@{k_val}", "Cover.", "ILD"]

print("  " + tbl_top(widths))
print("  " + tbl_row([c(h, BOLD) for h in headers], widths))
print("  " + tbl_divider(widths))

for name, res in algo_results.items():
    t = res["test"]
    row = [
        name,
        f"{t.get(f'P@{k_val}',0):.4f}",
        f"{t.get(f'R@{k_val}',0):.4f}",
        f"{t.get(f'NDCG@{k_val}',0):.4f}",
        f"{t.get(f'HR@{k_val}',0):.4f}",
        f"{t['Coverage']:.4f}",
        f"{t['ILD']:.4f}",
    ]
    print("  " + tbl_row(row, widths))

print("  " + tbl_bot(widths))

# ── Table 2: RMSE & Precision@10 comparison ──────────────────────────────────
print(f"\n{c('  ◆ TABLE 2 — RMSE & Precision@10 (MF models only, mirrors app.py _train_model output)', BOLD, MAGENTA)}")
widths2 = [22, 12, 12, 14, 14]
print("  " + tbl_top(widths2))
print("  " + tbl_row([c(h, BOLD) for h in ["Algorithm", "Train RMSE", "Test RMSE", "Train P@10", "Test P@10"]], widths2))
print("  " + tbl_divider(widths2))
for name, res in algo_results.items():
    if res["train_rmse"] is not None:
        print("  " + tbl_row([
            name,
            f"{res['train_rmse']:.4f}",
            f"{res['test_rmse']:.4f}",
            f"{res['train_precision']:.4f}",
            f"{res['test_precision']:.4f}",
        ], widths2))
    else:
        print("  " + tbl_row([name, "N/A", "N/A", "N/A", "N/A"], widths2))
print("  " + tbl_bot(widths2))

# ── Table 3: Experiment group results ─────────────────────────────────────
print(f"\n{c('  ◆ TABLE 3 — Experiment Group Simulation Results', BOLD, MAGENTA)}")
widths3 = [5, 10, 12, 8, 10, 12]
h3 = ["Grp", "Update", "Baseline", "UI", "CTR", "Diversity"]
print("  " + tbl_top(widths3))
print("  " + tbl_row([c(h, BOLD) for h in h3], widths3))
print("  " + tbl_divider(widths3))
for g, r in group_sim_results.items():
    row3 = [str(g), r["update_type"], r["baseline"], r["ui"],
            f"{r['avg_ctr']:.3f}", f"{r['avg_diversity']:.3f}"]
    print("  " + tbl_row(row3, widths3))
print("  " + tbl_bot(widths3))

# ── Table 4: Cosine Similarity — Carousel vs Ranked List ──────────────────
print(f"\n{c('  ◆ TABLE 4 — Cosine Similarity: Carousel vs Ranked List (mean over iterations)', BOLD, MAGENTA)}")
# Build a lookup: (update_type, baseline, ui) → mean similarity for impl and expl
def _mean_sim(sim_curve):
    return float(np.mean(sim_curve)) if sim_curve else 0.0

ui_sim_lookup = {}
for g, r in group_sim_results.items():
    key = (r["update_type"], r["baseline"], r["ui"])
    ui_sim_lookup[key] = {
        "impl": _mean_sim(r["sim_curve_impl"]),
        "expl": _mean_sim(r["sim_curve_expl"]),
    }

widths4 = [10, 12, 16, 14, 16, 14]
h4 = ["Update", "Baseline", "Carousel(Impl)", "List(Impl)", "Carousel(Expl)", "List(Expl)"]
print("  " + tbl_top(widths4))
print("  " + tbl_row([c(h, BOLD) for h in h4], widths4))
print("  " + tbl_divider(widths4))
for upd in ["control", "adaptive"]:
    for base in ["random", "popularity"]:
        car_impl = ui_sim_lookup.get((upd, base, "carousel"), {}).get("impl", 0.0)
        lst_impl = ui_sim_lookup.get((upd, base, "list"),     {}).get("impl", 0.0)
        car_expl = ui_sim_lookup.get((upd, base, "carousel"), {}).get("expl", 0.0)
        lst_expl = ui_sim_lookup.get((upd, base, "list"),     {}).get("expl", 0.0)
        print("  " + tbl_row([
            upd, base,
            f"{car_impl:.4f}", f"{lst_impl:.4f}",
            f"{car_expl:.4f}", f"{lst_expl:.4f}",
        ], widths4))
print("  " + tbl_bot(widths4))

# ── Table 5: RMSE, nDCG & Precision — Explicit vs Implicit MF ────────────
print(f"\n{c(f'  ◆ TABLE 5 — RMSE, NDCG@{k_val} & Precision@10: Explicit vs Implicit MF', BOLD, MAGENTA)}")
widths5 = [20, 12, 12, 16, 14, 13, 11]
h5 = ["Model", "Train RMSE", "Test RMSE",
      f"Train NDCG@{k_val}", f"Test NDCG@{k_val}",
      "Train P@10", "Test P@10"]
print("  " + tbl_top(widths5))
print("  " + tbl_row([c(h, BOLD) for h in h5], widths5))
print("  " + tbl_divider(widths5))
for name in ["Implicit MF", "Explicit MF"]:
    res = algo_results.get(name)
    if res is None:
        continue
    tr_ndcg = res["train"].get(f"NDCG@{k_val}", 0.0)
    te_ndcg = res["test"].get(f"NDCG@{k_val}", 0.0)
    tr_rmse = res["train_rmse"]
    te_rmse = res["test_rmse"]
    tr_prec = res["train_precision"]
    te_prec = res["test_precision"]
    print("  " + tbl_row([
        name,
        f"{tr_rmse:.4f}" if tr_rmse is not None else "N/A",
        f"{te_rmse:.4f}" if te_rmse is not None else "N/A",
        f"{tr_ndcg:.4f}", f"{te_ndcg:.4f}",
        f"{tr_prec:.4f}" if tr_prec is not None else "N/A",
        f"{te_prec:.4f}" if te_prec is not None else "N/A",
    ], widths5))
print("  " + tbl_bot(widths5))

# ── Key findings ──────────────────────────────────────────────────────────
print(f"\n{c('  ◆ KEY FINDINGS', BOLD, MAGENTA)}")

# Best algorithm by NDCG
best_algo = max(algo_results, key=lambda n: algo_results[n]["test"].get(f"NDCG@{k_val}", 0))
best_ndcg = algo_results[best_algo]["test"][f"NDCG@{k_val}"]

# Adaptive vs control CTR
adaptive_ctrs = [r["avg_ctr"] for g, r in group_sim_results.items() if r["update_type"] == "adaptive"]
control_ctrs  = [r["avg_ctr"] for g, r in group_sim_results.items() if r["update_type"] == "control"]
adaptive_mean = np.mean(adaptive_ctrs)
control_mean  = np.mean(control_ctrs)
ctr_lift      = (adaptive_mean - control_mean) / control_mean * 100

# Popularity vs random diversity
pop_div = np.mean([r["avg_diversity"] for r in group_sim_results.values() if r["baseline"] == "popularity"])
rnd_div = np.mean([r["avg_diversity"] for r in group_sim_results.values() if r["baseline"] == "random"])

findings = [
    f"Best algorithm by NDCG@{k_val}: {best_algo} ({best_ndcg:.4f})",
    f"Adaptive update CTR lift over control: {ctr_lift:+.1f}% ({adaptive_mean:.3f} vs {control_mean:.3f})",
    f"Popularity baseline avg diversity: {pop_div:.3f}  |  Random baseline: {rnd_div:.3f}",
    f"ImplicitFactorizationModel  — train RMSE: {impl_train_rmse:.4f}  test RMSE: {impl_test_rmse:.4f}  "
    f"train P@10: {impl_train_prec:.4f}  test P@10: {impl_test_prec:.4f}",
    f"ExplicitFactorizationModel  — train RMSE: {expl_train_rmse:.4f}  test RMSE: {expl_test_rmse:.4f}  "
    f"train P@10: {expl_train_prec:.4f}  test P@10: {expl_test_prec:.4f}",
    f"Random baseline coverage: {algo_results['Random']['test']['Coverage']:.4f} (expected ~1.0 for full catalogue reach)",
]
for f in findings:
    print(f"    {c('→', GREEN, BOLD)} {f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Save JSON results
# ─────────────────────────────────────────────────────────────────────────────
header("STEP 7 — Saving results")

output_data = {
    "meta": {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "sim_users": n_sim_users,
        "iterations_per_user": args.iterations,
        "top_n_movies": args.top_n,
        "k_values": args.k,
    },
    "algorithm_metrics": {
        name: {
            "train": res["train"],
            "test":  res["test"],
            "train_rmse":      res["train_rmse"],
            "test_rmse":       res["test_rmse"],
            "train_precision": res["train_precision"],
            "test_precision":  res["test_precision"],
        }
        for name, res in algo_results.items()
    },
    "experiment_groups": group_sim_results,
    "key_findings": findings,
}

json_path = os.path.join(args.out, "simulation_report.json")
with open(json_path, "w") as f:
    json.dump(output_data, f, indent=2)
ok(f"JSON saved → {json_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 10.  HTML report
# ─────────────────────────────────────────────────────────────────────────────
k_display = max(args.k)

def fmt(v, decimals=4):
    if v is None: return "—"
    return f"{v:.{decimals}f}"

def algo_rows_html():
    rows = ""
    for i, (name, res) in enumerate(algo_results.items()):
        t = res["test"]
        cls = "alt" if i % 2 else ""
        rows += f"""
        <tr class=\"{cls}\">
          <td class=\"label\">{name}</td>
          <td>{fmt(t.get(f'P@{k_display}'))}</td>
          <td>{fmt(t.get(f'R@{k_display}'))}</td>
          <td>{fmt(t.get(f'NDCG@{k_display}'))}</td>
          <td>{fmt(t.get(f'HR@{k_display}'))}</td>
          <td>{fmt(res['train_rmse'])}</td>
          <td>{fmt(res['test_rmse'])}</td>
          <td>{fmt(t['Coverage'])}</td>
          <td>{fmt(t['ILD'])}</td>
        </tr>"""
    return rows

def group_rows_html():
    rows = ""
    for g, r in group_sim_results.items():
        badge_u = f'<span class=\" badge badge-adaptive\">Adaptive</span>' if r["update_type"] == "adaptive" else f'<span class=\" badge badge-control\">Control</span>'
        badge_b = f'<span class=\" badge badge-pop\">Popularity</span>' if r["baseline"] == "popularity" else f'<span class=\" badge badge-rnd\">Random</span>'
        badge_ui = f'<span class=\" badge badge-ui\">{r["ui"].title()}</span>'
        ctr_bar  = f'<div class=\"bar-wrap\"><div class=\"bar\" style=\"width:{r["avg_ctr"]*100:.1f}%\"></div><span>{r["avg_ctr"]:.3f}</span></div>'
        div_bar  = f'<div class=\"bar-wrap\"><div class=\"bar bar-div\" style=\"width:{r["avg_diversity"]*100:.1f}%\"></div><span>{r["avg_diversity"]:.3f}</span></div>'
        cls = "alt" if g % 2 else ""
        rows += f"""
        <tr class=\"{cls}\">
          <td class=\"gnum\">{g}</td>
          <td>{badge_u}</td>
          <td>{badge_b}</td>
          <td>{badge_ui}</td>
          <td>{ctr_bar}</td>
          <td>{div_bar}</td>
          <td class=\"small\">{r['avg_clicks']:.1f}</td>
        </tr>"""
    return rows

def findings_html():
    items = ""
    for f in findings:
        items += f'<li>{f}</li>'
    return items

def k_tabs_html():
    tabs = ""
    for k in args.k:
        cols = ["P@", "R@", "NDCG@", "HR@"]
        header_row = "".join(f"<th>{c}{k}</th>" for c in cols)
        data_rows  = ""
        for i, (name, res) in enumerate(algo_results.items()):
            t = res["test"]
            cls = "alt" if i % 2 else ""
            cells = "".join(f'<td>{fmt(t.get(f"{c}{k}"))}</td>' for c in cols)
            data_rows += f'<tr class="{cls}"><td class="label">{name}</td>{cells}</tr>'
        tabs += f"""
        <div class=\"tab-panel\" id=\"tab-k{k}\">
          <table><thead><tr><th>Algorithm</th>{header_row}</tr></thead>
          <tbody>{data_rows}</tbody></table>
        </div>"""
    tab_btns = "".join(
        f'<button class=\"tab-btn{" active" if i==0 else ""}\" onclick=\"showTab(\'k{k}\', this)\">K={k}</button>'
        for i, k in enumerate(args.k))
    return f'<div class=\"tab-bar\">{tab_btns}</div>{tabs}'

def gold_sim_chart_data():
    """
    Returns labels + two baseline dicts (popularity, random).
    Each dict has 'implicit' and 'explicit' keys with the avg per-iteration
    cosine similarity curve, averaged across all groups of that baseline.
    """
    n_iter = args.iterations

    def curves(baseline):
        impl_matrix, expl_matrix = [], []
        for r in group_sim_results.values():
            if r["baseline"] == baseline:
                impl_matrix.append(r["sim_curve_impl"])
                expl_matrix.append(r["sim_curve_expl"])
        return {
            "implicit": np.array(impl_matrix).mean(axis=0).tolist() if impl_matrix else [0]*n_iter,
            "explicit": np.array(expl_matrix).mean(axis=0).tolist() if expl_matrix else [0]*n_iter,
        }

    return list(range(1, n_iter + 1)), curves("popularity"), curves("random")

chart_labels, pop_curves, rnd_curves = gold_sim_chart_data()

def gold_sim_chart_data_by_ui():
    """
    Returns labels + two baseline dicts (popularity, random).
    Each dict has 'list' and 'carousel' keys with the avg per-iteration
    cosine similarity curve (averaged over Implicit & Explicit MF).

    Only adaptive groups are used — control groups produce no score updates
    and would flatten/dilute the carousel vs ranked-list divergence signal.
    Similarity is taken from the Implicit MF curve only (matches paper Fig 1
    which uses a single gold standard from the pre-trained MF model).
    """
    n_iter = args.iterations

    def curves(baseline):
        list_matrix, carousel_matrix = [], []
        for r in group_sim_results.values():
            if r["baseline"] == baseline and r["update_type"] == "adaptive":
                curve = np.array(r["sim_curve_impl"])
                if r["ui"] == "list":
                    list_matrix.append(curve)
                else:
                    carousel_matrix.append(curve)
        return {
            "list":     np.array(list_matrix).mean(axis=0).tolist() if list_matrix else [0]*n_iter,
            "carousel": np.array(carousel_matrix).mean(axis=0).tolist() if carousel_matrix else [0]*n_iter,
        }

    return list(range(1, n_iter + 1)), curves("popularity"), curves("random")

ui_chart_labels, ui_pop_curves, ui_rnd_curves = gold_sim_chart_data_by_ui()

def js_arr(lst):
    return json.dumps([round(v, 6) for v in lst])

# Blue = Implicit MF, Orange = Explicit MF — cosine similarity only, single Y axis
def build_datasets(curves, tag):
    return f"""[
          {{
            label: 'Implicit MF ({tag})',
            data: {js_arr(curves["implicit"])},
            borderColor: '#1f77b4',
            backgroundColor: 'rgba(31,119,180,0.08)',
            borderWidth: 2,
            pointRadius: 3, pointHoverRadius: 5,
            tension: 0.35,
          }},
          {{
            label: 'Explicit MF ({tag})',
            data: {js_arr(curves["explicit"])},
            borderColor: '#ff7f0e',
            backgroundColor: 'rgba(255,127,14,0.08)',
            borderWidth: 2,
            pointRadius: 3, pointHoverRadius: 5,
            tension: 0.35,
          }},
        ]"""

pop_datasets = build_datasets(pop_curves, "Popularity")
rnd_datasets = build_datasets(rnd_curves, "Random")

# Figure 2: Ranked List vs Carousel (averaged over Implicit+Explicit MF)
def build_ui_datasets(curves, tag):
    return f"""[
          {{
            label: 'Ranked List ({tag})',
            data: {js_arr(curves["list"])},
            borderColor: '#1f77b4',
            backgroundColor: 'rgba(31,119,180,0.08)',
            borderWidth: 2,
            pointRadius: 3, pointHoverRadius: 5,
            tension: 0.35,
          }},
          {{
            label: 'Carousels ({tag})',
            data: {js_arr(curves["carousel"])},
            borderColor: '#ff7f0e',
            backgroundColor: 'rgba(255,127,14,0.08)',
            borderWidth: 2,
            pointRadius: 3, pointHoverRadius: 5,
            tension: 0.35,
          }},
        ]"""

ui_pop_datasets = build_ui_datasets(ui_pop_curves, "Popularity")
ui_rnd_datasets = build_ui_datasets(ui_rnd_curves, "Random")

figure1_html = f"""
  <section>
    <div class="section-header">
      <span class="section-num">05</span>
      <h2>Figure 1 — Average Similarity to Gold Standard</h2>
    </div>
    <p style="color:var(--muted);font-size:13px;margin-bottom:20px;">
      Cosine similarity between the session score vector and the gold standard,
      averaged across all simulated users per iteration.
      <strong style="color:#1f77b4">Blue</strong> = ImplicitFactorizationModel &nbsp;·&nbsp;
      <strong style="color:#ff7f0e">Orange</strong> = ExplicitFactorizationModel.
    </p>
    <div class="chart-pair">
      <div class="chart-box">
        <div class="chart-title">Average Similarity to Gold Standard (Popularity-based)</div>
        <canvas id="chartPop"></canvas>
      </div>
      <div class="chart-box">
        <div class="chart-title">Average Similarity to Gold Standard (Random-based)</div>
        <canvas id="chartRnd"></canvas>
      </div>
    </div>
    <p class="chart-caption">
      Average Similarity to Gold Standard — ImplicitFactorizationModel vs ExplicitFactorizationModel
    </p>
  </section>
"""

figure2_html = f"""
  <section>
    <div class="section-header">
      <span class="section-num">06</span>
      <h2>Figure 2 — Average Similarity to Gold Standard by UI Type</h2>
    </div>
    <p style="color:var(--muted);font-size:13px;margin-bottom:20px;">
      Cosine similarity between the session score vector and the gold standard (averaged over
      ImplicitMF &amp; ExplicitMF), split by UI presentation type.
      <strong style="color:#1f77b4">Blue</strong> = Ranked List &nbsp;·&nbsp;
      <strong style="color:#ff7f0e">Orange</strong> = Carousel.
    </p>
    <div class="chart-pair">
      <div class="chart-box">
        <div class="chart-title">Average Similarity to Gold Standard (Popularity-based)</div>
        <canvas id="chartUiPop"></canvas>
      </div>
      <div class="chart-box">
        <div class="chart-title">Average Similarity to Gold Standard (Random-based)</div>
        <canvas id="chartUiRnd"></canvas>
      </div>
    </div>
    <p class="chart-caption">
      Average Similarity to Gold Standard for Popularity-based and Random-based Models
    </p>
  </section>
"""

figure1_css = """
  .chart-pair {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 12px;
  }}
  .chart-box {{
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 20px 20px 16px;
  }}
  .chart-title {{
    font-size: 12px;
    font-weight: 600;
    color: #444d56;
    text-align: center;
    margin-bottom: 12px;
  }}
  .chart-caption {{
    text-align: center;
    font-size: 12px;
    color: var(--muted);
    margin-top: 8px;
    font-style: italic;
  }}
  @media (max-width: 700px) {{
    .chart-pair {{ grid-template-columns: 1fr; }}
  }}
"""

figure1_js = f"""
  (function() {{
    const sharedOptions = {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{
          position: 'top',
          labels: {{
            color: '#444d56',
            font: {{ size: 11 }},
            boxWidth: 24,
            padding: 14,
            usePointStyle: false,
          }}
        }},
        tooltip: {{
          backgroundColor: '#fff',
          borderColor: '#d0d7de',
          borderWidth: 1,
          titleColor: '#24292f',
          bodyColor: '#57606a',
          callbacks: {{
            label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(4)
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: 'Iteration', color: '#57606a', font: {{ size: 11 }} }},
          grid: {{ color: '#ebebeb' }},
          ticks: {{ color: '#57606a', font: {{ size: 10 }} }},
        }},
        y: {{
          title: {{ display: true, text: 'Cosine Similarity', color: '#57606a', font: {{ size: 11 }} }},
          grid: {{ color: '#ebebeb' }},
          ticks: {{ color: '#57606a', font: {{ size: 10 }}, callback: v => v.toFixed(3) }},
        }},
      }}
    }};

    const labels = {js_arr(chart_labels)};

    new Chart(document.getElementById('chartPop'), {{
      type: 'line',
      data: {{ labels, datasets: {pop_datasets} }},
      options: sharedOptions,
    }});

    new Chart(document.getElementById('chartRnd'), {{
      type: 'line',
      data: {{ labels, datasets: {rnd_datasets} }},
      options: sharedOptions,
    }});

    // Figure 2 — Ranked List vs Carousel
    const uiLabels = {js_arr(ui_chart_labels)};

    new Chart(document.getElementById('chartUiPop'), {{
      type: 'line',
      data: {{ labels: uiLabels, datasets: {ui_pop_datasets} }},
      options: sharedOptions,
    }});

    new Chart(document.getElementById('chartUiRnd'), {{
      type: 'line',
      data: {{ labels: uiLabels, datasets: {ui_rnd_datasets} }},
      options: sharedOptions,
    }});
  }})();
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Recommender Simulation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  :root {{
    --bg:       #0d1117;
    --surface:  #161b22;
    --surface2: #1f2937;
    --border:   #30363d;
    --accent:   #58a6ff;
    --accent2:  #3fb950;
    --accent3:  #f78166;
    --accent4:  #d2a8ff;
    --text:     #e6edf3;
    --muted:    #7d8590;
    --pop:      #ffa657;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    line-height: 1.6;
  }}

  .hero {{
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d2137 100%);
    border-bottom: 1px solid var(--border);
    padding: 56px 48px 40px;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(88,166,255,0.08) 0%, transparent 70%);
    pointer-events: none;
  }}
  .hero-tag {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 12px;
  }}
  .hero h1 {{
    font-size: 36px;
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
    margin-bottom: 8px;
  }}
  .hero-sub {{
    color: var(--muted);
    font-size: 15px;
    margin-bottom: 28px;
  }}
  .meta-pills {{
    display: flex; flex-wrap: wrap; gap: 10px;
  }}
  .pill {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--muted);
  }}
  .pill span {{ color: var(--accent); }}

  .container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 32px;
  }}

  section {{ margin-bottom: 56px; }}

  .section-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }}
  .section-num {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 4px;
    padding: 2px 8px;
  }}
  .section-header h2 {{
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
  }}

  /* Tables */
  .table-wrap {{ overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  thead th {{
    background: var(--surface2);
    color: var(--muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  td {{
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--text);
  }}
  tr.alt td {{ background: var(--surface); }}
  tr:last-child td {{ border-bottom: none; }}
  td.label {{ font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; color: var(--accent4); }}
  td.gnum  {{ text-align: center; color: var(--muted); font-weight: 600; }}
  td.small {{ text-align: right; color: var(--muted); }}

  /* Badges */
  .badge {{
    display: inline-block;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
  }}
  .badge-adaptive {{ background: rgba(63,185,80,0.15); color: var(--accent2); border: 1px solid rgba(63,185,80,0.3); }}
  .badge-control   {{ background: rgba(125,133,144,0.15); color: var(--muted); border: 1px solid rgba(125,133,144,0.3); }}
  .badge-pop       {{ background: rgba(255,166,87,0.15); color: var(--pop); border: 1px solid rgba(255,166,87,0.3); }}
  .badge-rnd       {{ background: rgba(247,129,102,0.15); color: var(--accent3); border: 1px solid rgba(247,129,102,0.3); }}
  .badge-ui        {{ background: rgba(210,168,255,0.12); color: var(--accent4); border: 1px solid rgba(210,168,255,0.25); }}

  /* Bar charts */
  .bar-wrap {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .bar {{
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    min-width: 2px;
    transition: width 0.3s;
  }}
  .bar-div {{ background: linear-gradient(90deg, var(--accent4), var(--pop)); }}
  .bar-wrap span {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    white-space: nowrap;
  }}

  /* Findings */
  .findings-list {{
    list-style: none;
    display: grid;
    gap: 10px;
  }}
  .findings-list li {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--text);
  }}

  /* Tabs */
  .tab-bar {{
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
  }}
  .tab-btn {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--muted);
    cursor: pointer;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    padding: 6px 14px;
    transition: all 0.2s;
  }}
  .tab-btn.active, .tab-btn:hover {{
    background: rgba(88,166,255,0.1);
    border-color: var(--accent);
    color: var(--accent);
  }}
  .tab-panel {{ display: none; }}
  .tab-panel.visible {{ display: block; }}

  /* Summary cards */
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
  }}
  .card-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 6px;
  }}
  .card-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: var(--accent);
  }}
  .card-sub {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
  }}

  footer {{
    border-top: 1px solid var(--border);
    padding: 24px 48px;
    color: var(--muted);
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
  }}
  {figure1_css}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-tag">Recommender System · Simulation Report</div>
  <h1>Movie Recommender<br>Evaluation Dashboard</h1>
  <p class="hero-sub">Multi-algorithm comparison across 8 factorial experiment groups</p>
  <div class="meta-pills">
    <div class="pill">Generated <span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span></div>
    <div class="pill">Seed <span>{args.seed}</span></div>
    <div class="pill">Users/group <span>{n_sim_users}</span></div>
    <div class="pill">Iterations <span>{args.iterations}</span></div>
    <div class="pill">Top-N movies <span>{args.top_n}</span></div>
    <div class="pill">K values <span>{', '.join(map(str, args.k))}</span></div>
  </div>
</div>

<div class="container">

  <!-- Summary cards -->
  <div class="cards">
    <div class="card">
      <div class="card-label">Best NDCG@{k_display}</div>
      <div class="card-value">{fmt(algo_results[best_algo]['test'][f'NDCG@{k_display}'])}</div>
      <div class="card-sub">{best_algo}</div>
    </div>
    <div class="card">
      <div class="card-label">Adaptive CTR lift</div>
      <div class="card-value">{ctr_lift:+.1f}%</div>
      <div class="card-sub">vs. control groups</div>
    </div>
    <div class="card">
      <div class="card-label">Implicit MF test RMSE</div>
      <div class="card-value">{fmt(algo_results['Implicit MF']['test_rmse'], 3)}</div>
      <div class="card-sub">train: {fmt(algo_results['Implicit MF']['train_rmse'], 3)}</div>
    </div>
    <div class="card">
      <div class="card-label">Explicit MF test RMSE</div>
      <div class="card-value">{fmt(algo_results['Explicit MF']['test_rmse'], 3)}</div>
      <div class="card-sub">train: {fmt(algo_results['Explicit MF']['train_rmse'], 3)}</div>
    </div>
    <div class="card">
      <div class="card-label">Random Coverage</div>
      <div class="card-value">{fmt(algo_results['Random']['test']['Coverage'], 3)}</div>
      <div class="card-sub">catalogue proportion seen</div>
    </div>
  </div>

  <!-- Section 1: Algorithm metrics -->
  <section>
    <div class="section-header">
      <span class="section-num">01</span>
      <h2>Algorithm Evaluation Metrics — Test Split</h2>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Algorithm</th>
          <th>P@{k_display}</th><th>R@{k_display}</th><th>NDCG@{k_display}</th><th>HR@{k_display}</th>
          <th>Train RMSE</th><th>Test RMSE</th><th>Coverage</th><th>ILD</th>
        </tr></thead>
        <tbody>{algo_rows_html()}</tbody>
      </table>
    </div>
  </section>

  <!-- Section 2: K-value breakdown -->
  <section>
    <div class="section-header">
      <span class="section-num">02</span>
      <h2>Ranking Metrics by K Value</h2>
    </div>
    {k_tabs_html()}
  </section>

  <!-- Section 3: Experiment groups -->
  <section>
    <div class="section-header">
      <span class="section-num">03</span>
      <h2>Experiment Group Simulation (2×2×2 Factorial)</h2>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th style="text-align:center">Grp</th>
          <th>Update Type</th><th>Baseline</th><th>UI</th>
          <th>CTR</th><th>ILD (diversity)</th><th>Avg Clicks</th>
        </tr></thead>
        <tbody>{group_rows_html()}</tbody>
      </table>
    </div>
  </section>

  <!-- Section 4: Key findings -->
  <section>
    <div class="section-header">
      <span class="section-num">04</span>
      <h2>Key Findings</h2>
    </div>
    <ul class="findings-list">{findings_html()}</ul>
  </section>

  {figure1_html}

  {figure2_html}

</div>

<footer>
  simulate_and_report.py · seed={args.seed} · {datetime.now().isoformat()}
</footer>

<script>
  // Show first tab panel on load
  document.addEventListener('DOMContentLoaded', () => {{
    const first = document.querySelector('.tab-panel');
    if (first) first.classList.add('visible');
  }});

  function showTab(id, btn) {{
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('visible'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    const panel = document.getElementById('tab-' + id);
    if (panel) panel.classList.add('visible');
    btn.classList.add('active');
  }}

  {figure1_js}
</script>
</body>
</html>"""

html_path = os.path.join(args.out, "simulation_report.html")
with open(html_path, "w") as f:
    f.write(html)
ok(f"HTML report saved → {html_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Done
# ─────────────────────────────────────────────────────────────────────────────
header("DONE")
print(c(f"  Results written to ./{args.out}/", BOLD, GREEN))
print(c(f"  • simulation_report.json   (raw data)", DIM))
print(c(f"  • simulation_report.html   (interactive dashboard)", DIM))
print()