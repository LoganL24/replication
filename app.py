from flask import Flask, request, redirect, url_for, render_template, flash, jsonify, session
from flask_session import Session
import pandas as pd
import numpy as np
import random
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score
import json
import os
from flask_wtf import CSRFProtect
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "insight_general_ai_evaluation_system"

csrf = CSRFProtect(app)

# Configure session to use filesystem (instead of signed cookies)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Configuration for data paths
MOVIES_PATH = 'data/movie_metadata.json'  # Update with the correct path
RATINGS_PATH = 'data/ratings.csv'         # Update with the correct path

# MovieRecommender Class
class MovieRecommender:
    def __init__(self, movies_df, ratings_df, top_n=100):
        np.random.seed(42)
        random.seed(42)
        self.top_n = top_n
        self.movies_df = movies_df[movies_df['genres'] != '(no genres listed)'].copy()
        self._preprocess_data(ratings_df)
        self._build_interactions()
        self._train_model()

    def precision_at_k(self, interactions, k=10, threshold=4.0):
        precisions = []

        # Get all unique user internal IDs in this split
        user_ids = np.unique(interactions.user_ids)
        n_items = len(self.item_id_map)

        for user_id in user_ids:
            # Predict scores for all items
            scores = self.model.predict(user_id, np.arange(n_items))

            # Top-K recommended item internal IDs
            top_k_items = np.argsort(-scores)[:k]

            # Ground truth: items this user actually rated >= threshold
            user_mask = interactions.user_ids == user_id
            user_item_ids = interactions.item_ids[user_mask]
            user_ratings = interactions.ratings[user_mask]
            relevant_items = set(user_item_ids[user_ratings >= threshold])

            # Count hits
            hits = sum(1 for item in top_k_items if item in relevant_items)
            precisions.append(hits / k)

        return np.mean(precisions)
    
    def _preprocess_data(self, ratings_df):
        # Filter ratings for movies present in movies_df
        self.ratings_df = ratings_df[ratings_df['movieId'].isin(self.movies_df['movieId'])].copy()

        # Get top_n movies based on total ratings
        top_movies = (
            self.ratings_df.groupby('movieId')['rating']
            .sum()
            .sort_values(ascending=False)
            .head(self.top_n)
            .index.tolist()
        )
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(top_movies)].reset_index(drop=True)

        if self.ratings_df.empty:
            raise ValueError("No ratings available after preprocessing. Please check your ratings.csv and filtering criteria.")

        # Categorize users and items
        user_id_cats = pd.Categorical(self.ratings_df['userId'])
        item_id_cats = pd.Categorical(self.ratings_df['movieId'])
        self.ratings_df['user_id'] = user_id_cats.codes
        self.ratings_df['item_id'] = item_id_cats.codes
        self.user_id_map = dict(enumerate(user_id_cats.categories))
        self.item_id_map = dict(enumerate(item_id_cats.categories))
        self.reverse_user_id_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_item_id_map = {v: k for k, v in self.item_id_map.items()}

    def _build_interactions(self):
        self.interactions = Interactions(
            user_ids=self.ratings_df['user_id'].values,
            item_ids=self.ratings_df['item_id'].values,
            ratings=self.ratings_df['rating'].values.astype(np.float32)
        )
        self.train_interactions, self.test_interactions = random_train_test_split(
            self.interactions,
            test_percentage=0.2,
            random_state=np.random.RandomState(42)
        )

    def _train_model(self):
        self.model = ImplicitFactorizationModel(n_iter=10, random_state=np.random.RandomState(42))
        self.model.fit(self.train_interactions)
        self.train_rmse = rmse_score(self.model, self.train_interactions)
        self.test_rmse = rmse_score(self.model, self.test_interactions)
        self.train_precision = self.precision_at_k(self.train_interactions, k=10)
        self.test_precision  = self.precision_at_k(self.test_interactions,  k=10)

        print(f'Train RMSE: {self.train_rmse:.4f} | Train Precision@10: {self.train_precision:.4f}')
        print(f'Test  RMSE: {self.test_rmse:.4f}  | Test  Precision@10: {self.test_precision:.4f}')
        
    

    def get_recommendations(self, user_id, recommendation_type='carousel', num_items_per_carousel=10, num_carousels=5):
        if user_id not in self.reverse_user_id_map:
            raise ValueError(f"User ID {user_id} not found in the dataset.")
        user_internal_id = self.reverse_user_id_map[user_id]
        n_items = len(self.item_id_map)
        item_internal_ids = np.arange(n_items)

        scores = self.model.predict(user_internal_id, item_internal_ids)
        

        # Fetch known positives from the dataset
        known_positives = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].unique()

        # Prepare recommendations DataFrame
        scores_df = pd.DataFrame({'movieId': list(self.item_id_map.values()), 'score': scores})
        scores_df = scores_df[~scores_df['movieId'].isin(known_positives)]
        recommendations = scores_df.merge(self.movies_df, on='movieId')

        if recommendation_type == 'carousel':
            return self._organize_recommendations_by_genre(recommendations, num_items_per_carousel, num_carousels)
        elif recommendation_type == 'list':
            return recommendations.sort_values(by='score', ascending=False).reset_index(drop=True)
        else:
            raise ValueError("Invalid recommendation_type. Choose 'carousel' or 'list'.")

    def _organize_recommendations_by_genre(self, recommendations, num_items_per_carousel=10, num_carousels=5):
        recommendations_genres = recommendations.copy()
        recommendations_genres['genres'] = recommendations_genres['genres'].str.split('|')
        recommendations_genres = recommendations_genres.explode('genres')
        genre_groups = recommendations_genres.groupby('genres')
        genre_recommendations = {}
        for genre, group in genre_groups:
            genre_recommendations[genre] = group.sort_values(by='score', ascending=False).head(num_items_per_carousel)
        sorted_genres = self._get_sorted_genres(genre_recommendations)
        sorted_genres = sorted_genres[:num_carousels]
        return {genre: genre_recommendations[genre].to_dict(orient='records') for genre in sorted_genres}

    def _get_sorted_genres(self, genre_recommendations):
        genre_scores = {genre: df['score'].sum() for genre, df in genre_recommendations.items()}
        return sorted(genre_scores, key=genre_scores.get, reverse=True)

def load_movies(json_path):
    with open(json_path, 'r') as f:
        movies = json.load(f)
    movies_df = pd.DataFrame(movies)
    expected_columns = {'movieId', 'Title', 'Year', 'imdbRating', 'Poster', 'genres'}
    
    missing = expected_columns - set(movies_df.columns)
    if missing:
        raise ValueError(f"Missing columns in movie_metadata.json: {missing}")
    
    return movies_df

def load_ratings(ratings_path):
    try:
        ratings_df = pd.read_csv(ratings_path)
        required_columns = {'userId', 'movieId', 'rating'}
        if not required_columns.issubset(ratings_df.columns):
            missing = required_columns - set(ratings_df.columns)
            raise ValueError(f"Missing columns in ratings.csv: {missing}")
        print("Ratings data loaded successfully.")
        return ratings_df
    except Exception as e:
        print(f"Error loading ratings data: {e}")
        raise

# Lazy initialization — allows app.py to be imported by simulate.py without
# requiring the data files to exist at import time.
_recommender = None

def get_recommender():
    """Return the global MovieRecommender, initialising it on first call."""
    global _recommender
    if _recommender is None:
        movies_df = load_movies(MOVIES_PATH)
        ratings_df = load_ratings(RATINGS_PATH)
        _recommender = MovieRecommender(movies_df, ratings_df)
    return _recommender

# Helper function to update item scores based on user interaction
def update_item_scores(item_scores, clicked_item_id, seen_items, recommender, positive_factor=0.05, negative_factor=0.01, update_type='adaptive'):
    if update_type == 'control':
        # In control, do not update based on clicks
        return item_scores

    if update_type == 'adaptive':
        # Retrieve clicked item's genres
        clicked_item_row = recommender.movies_df[recommender.movies_df['movieId'] == clicked_item_id]
        if clicked_item_row.empty:
            print(f"Clicked item ID {clicked_item_id} not found in movies_df.")
            return item_scores  # Exit if movie not found

        clicked_item_genres = set(clicked_item_row['genres'].values[0].split('|'))

        # Collect all genres from seen items
        seen_genres = set()
        for item_id in seen_items:
            item_row = recommender.movies_df[recommender.movies_df['movieId'] == item_id]
            if not item_row.empty:
                seen_genres.update(item_row['genres'].values[0].split('|'))

        # Exclude genres present in the clicked item
        negative_genres = seen_genres - clicked_item_genres

        # Decrease scores for movies in negative genres
        for item_id, score in item_scores.items():
            item_row = recommender.movies_df[recommender.movies_df['movieId'] == item_id]
            if item_row.empty:
                continue  # Skip if movie not found

            item_genres = set(item_row['genres'].values[0].split('|'))

            # Check if movie has any negative genres
            if item_genres & negative_genres:
                item_scores[item_id] -= item_scores[item_id] * negative_factor
                if item_scores[item_id] < 0:
                    item_scores[item_id] = 0

            # Additionally, handle exact genre combination matches
            if item_genres == (clicked_item_genres - negative_genres):
                item_scores[item_id] -= item_scores[item_id] * negative_factor
                if item_scores[item_id] < 0:
                    item_scores[item_id] = 0

        # Increase scores for movies sharing genres with the clicked item
        for item_id, score in item_scores.items():
            if item_id == clicked_item_id:
                continue  # Skip the clicked item itself

            item_row = recommender.movies_df[recommender.movies_df['movieId'] == item_id]
            if item_row.empty:
                continue  # Skip if movie not found

            item_genres = set(item_row['genres'].values[0].split('|'))

            if clicked_item_genres & item_genres:
                item_scores[item_id] += item_scores[item_id] * positive_factor

        return item_scores

    raise ValueError("Invalid update_type. Choose 'adaptive' or 'control'.")

# Helper function to get baseline recommendations
def get_baseline_recommendations(recommender, baseline_type='popularity', num_items=50, recommendation_type='carousel', random_baseline=False):
    if baseline_type == 'random' or random_baseline:
        # Fetch random movies from the dataset
        recommendations = recommender.movies_df.sample(n=num_items, random_state=42).copy()
        recommendations['score'] = np.random.rand(num_items)
    elif baseline_type == 'popularity':
        # Calculate popularity based on total ratings
        popularity = (
            recommender.ratings_df.groupby('movieId')['rating']
            .sum()
            .sort_values(ascending=False)
            .head(num_items)
            .reset_index()
        )
        recommendations = recommender.movies_df[recommender.movies_df['movieId'].isin(popularity['movieId'])]
        recommendations = recommendations.merge(popularity, on='movieId')
        recommendations.rename(columns={'rating': 'score'}, inplace=True)
    else:
        raise ValueError("Invalid baseline_type. Choose 'random' or 'popularity'.")

    if recommendation_type == 'carousel':
        return recommender._organize_recommendations_by_genre(recommendations, num_items_per_carousel=10, num_carousels=5)
    elif recommendation_type == 'list':
        return recommendations.sort_values(by='score', ascending=False).reset_index(drop=True)
    else:
        raise ValueError("Invalid recommendation_type. Choose 'carousel' or 'list'.")

# Helper function to get current recommendations
def get_current_recommendations(item_scores, clicked_items, recommender, recommendation_type='list', num_items=50, random_update=False, update_type='adaptive'):
    if update_type == 'adaptive':
        # Adaptive logic already applied when updating item_scores
        pass
    elif update_type == 'control':
        # Do not modify item_scores
        pass
    else:
        raise ValueError("Invalid update_type. Choose 'adaptive' or 'control'.")

    # Create DataFrame from item_scores
    item_scores_df = pd.DataFrame({
        'movieId': list(item_scores.keys()),
        'score': list(item_scores.values())
    })

    # Merge with movies data
    recommendations = item_scores_df.merge(recommender.movies_df, on='movieId')

    # Exclude clicked items
    recommendations = recommendations[~recommendations['movieId'].isin(clicked_items)]

    if recommendation_type == 'list':
        recommendations = recommendations.sort_values(by='score', ascending=False).head(num_items)
        return recommendations
    elif recommendation_type == 'carousel':
        return recommender._organize_recommendations_by_genre(recommendations, num_items_per_carousel=10, num_carousels=5)
    else:
        raise ValueError("Invalid recommendation_type. Choose 'carousel' or 'list'.")


import random
import string


def generate_user_id():
    """Generates a user ID with the format: one digit - three letters - four digits."""
    return ''.join(random.choices(string.digits[1:], k=8))

# Landing page route
@app.route('/study', methods=['GET'])
@csrf.exempt
def landing_page():
    if 'parameters' not in session:
        # Get query parameters
        type_param = request.args.get('type', 'adaptive')          # 'adaptive' or 'control'
        baseline_param = request.args.get('baseline', 'popularity')  # 'random' or 'popularity'
        ui_param = request.args.get('ui', 'carousel')             # 'carousel' or 'list'
        mouse_logging_param = request.args.get('mouse_logging', 'true')
        mouse_logging_freq_param = request.args.get('mouse_logging_freq', '1000')  # Default 500ms

        # Set session parameters
        session['parameters'] = {
            'type': type_param,
            'baseline': baseline_param,
            'ui': ui_param,
            'mouse_logging': mouse_logging_param.lower() == 'true',
            'mouse_logging_freq': int(mouse_logging_freq_param),
        }

    params = session['parameters']
    recommendation_type = params['ui']          # 'carousel' or 'list'
    update_type = params['type']                # 'adaptive' or 'control'
    baseline_type = params['baseline']          # 'random' or 'popularity'


    if 'iteration' not in session:
        # First visit, initialize user state
        session['iteration'] = 1
        session['clicked_items'] = []
        # Generate a new user_id
        user_id = generate_user_id()
        session['user_id'] = user_id
        # Initialize item_scores based on baseline
        num_items = 500
        baseline_recommendations = get_baseline_recommendations(
            get_recommender(),
            baseline_type=baseline_type,
            num_items=num_items,
            recommendation_type=recommendation_type,
            random_baseline=(baseline_type == 'random')
        )
        if recommendation_type == 'carousel':
            # Flatten the recommendations to create item_scores
            item_scores = {}
            for genre, items in baseline_recommendations.items():
                for item in items:
                    item_scores[int(item['movieId'])] = item['score']
        else:
            # For list, use the 'score' directly
            item_scores = {int(row['movieId']): row['score'] for _, row in baseline_recommendations.iterrows()}
        session['item_scores'] = item_scores
        user_id = session['user_id']
    else:
        # Subsequent visits, retain session state
        user_id = session['user_id']

    item_scores = session['item_scores']
    clicked_items = session['clicked_items']
    iteration = session['iteration']

    if iteration > 10:
        uid = session['user_id']
        return render_template('completion.html', uid=uid)

    # Generate current recommendations
    recommendations = get_current_recommendations(
        item_scores,
        clicked_items,
        get_recommender(),
        recommendation_type=recommendation_type,
        num_items=50,
        random_update=False,
        update_type=update_type
    )

    # Store the seen items for use in the click handler
    if recommendation_type == 'carousel':
        seen_items = []
        for genre, items in recommendations.items():
            seen_items.extend([item['movieId'] for item in items])
        session['seen_items'] = seen_items
    else:
        session['seen_items'] = [item['movieId'] for _, item in recommendations.iterrows()]

    # Convert recommendations to appropriate format for rendering
    if recommendation_type == 'carousel':
        recommendations_dict = recommendations
    else:
        recommendations_dict = recommendations.to_dict(orient='records')

    # For debugging: get current session parameters
    current_parameters = session['parameters']

    # Render the recommendations
    return render_template(
        'recommendations.html',
        recommendations=recommendations_dict,
        iteration=iteration,
        recommendation_type=recommendation_type,
        current_parameters=current_parameters,
        user_id=user_id
    )

# Click handler route
@app.route('/click', methods=['POST'])
@csrf.exempt
def click():
    phase = request.form.get('phase', None)
    if phase == 'end_of_decision':
        # No movie selected, proceed to next iteration
        iteration = session.get('iteration', 1)
        session['iteration'] = iteration + 1
        # Check if the session should end
        if iteration >= 10:
            return redirect(url_for('completion'))
        else:
            return redirect(url_for('landing_page'))
    else:
        # Existing code for handling movie clicks
        clicked_item_id = int(request.form['movieId'])

        # Retrieve user state from session
        item_scores = session.get('item_scores', {})
        clicked_items = session.get('clicked_items', [])
        iteration = session.get('iteration', 1)
        user_id = session.get('user_id')
        seen_items = session.get('seen_items', [])
        params = session.get('parameters', {})
        recommendation_type = params.get('ui', 'carousel')
        update_type = params.get('type', 'adaptive')

        # Append clicked item to clicked_items
        clicked_items.append(clicked_item_id)
        session['clicked_items'] = clicked_items

        # Remove clicked_item_id from seen_items
        if clicked_item_id in seen_items:
            seen_items.remove(clicked_item_id)
        session['seen_items'] = seen_items

        # Update item_scores based on update_type
        item_scores = update_item_scores(
            item_scores,
            clicked_item_id,
            seen_items,
            get_recommender(),
            positive_factor=0.05,
            negative_factor=0.01,
            update_type=update_type
        )
        session['item_scores'] = item_scores

        # Increment iteration
        session['iteration'] = iteration + 1

        # Redirect back to landing page with current parameters
        if session['iteration'] >= 10:
            return redirect(url_for('completion'))
        else:
            return redirect(url_for('landing_page'))

# Completion page route
@app.route('/completion', methods=['GET'])
@csrf.exempt
def completion():
    uid = session['user_id']
    return render_template('completion.html',uid=uid)

# Route to clear the session
@app.route('/clear_session', methods=['GET'])
@csrf.exempt
def clear_session_route():
    session.clear()  # Removes all session data
    flash('Your session has been completely reset.')
    return redirect(url_for('landing_page'))

# Route to handle data submission
@app.route('/submit_data', methods=['POST'])
@csrf.exempt
def submit_data():
    data = request.get_json(force=True)

    if data is None:
        print("No data received")
        return jsonify({"message": "No data received"}), 400

    user_id = data.get('user_id')
    folder = 'experiment_data/'  
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    # Save the data to a JSON file
    filename = os.path.join(folder, f'{user_id}.json')
    with open(filename, 'w') as file:
        json.dump(data, file)
    # Return a response
    return jsonify({"message": "Data received and saved successfully"}), 200


# Route to handle survey data submission
@app.route('/submit_survey', methods=['POST'])
@csrf.exempt
def submit_survey():
    data = request.get_json(force=True)

    if data is None:
        print("No data received")
        return jsonify({"message": "No data received"}), 400

    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"message": "User ID is missing"}), 400

    # Prepare the folder for saving data
    folder = 'experiment_data/survey'  
    os.makedirs(folder, exist_ok=True)

    # Save the data to a JSON file named after the user_id
    filename = os.path.join(folder, f'{user_id}_survey.json')
    with open(filename, 'w') as file:
        json.dump(data, file)

    # Return a success response
    return jsonify({"message": "Survey data received and saved successfully"}), 200

@app.route('/show_params', methods=['GET'])
def show_params():
    # Pass Flask context variables explicitly
    context = {
        'session': dict(session),  # Convert session to a regular dictionary
        'args': request.args.to_dict(),  # Query parameters (GET)
        'form': request.form.to_dict(),  # Form data (POST)
        'json_body': request.get_json(silent=True)  # JSON body (POST)
    }
    return render_template('show_params.html', context=context)


GROUP_FILE = 'experiment_data/groups/group_data.json'

# Initialize the groups
def load_groups():
    if os.path.exists(GROUP_FILE):
        with open(GROUP_FILE, 'r') as f:
            return json.load(f)
    else:
        return [0] * 8  # 8 groups for the 2x2x2 factorial design

def save_groups(groups):
    with open(GROUP_FILE, 'w') as f:
        json.dump(groups, f)

def balanced_assign_participant(groups):
    min_index = np.argmin(groups)
    groups[min_index] += 1
    save_groups(groups)  # Save updated groups after assignment
    return min_index

@app.route('/start')
@csrf.exempt  # Exempt this route from CSRF protection
def assign_participant():
    groups = load_groups()  # Load the current group distribution
    group_index = balanced_assign_participant(groups)
    
    # Mapping based on group index
    mapping = {
        0: ('control', 'random', 'list'),
        1: ('control', 'random', 'carousel'),
        2: ('control', 'popularity', 'list'),
        3: ('control', 'popularity', 'carousel'),
        4: ('adaptive', 'random', 'list'),
        5: ('adaptive', 'random', 'carousel'),
        6: ('adaptive', 'popularity', 'list'),
        7: ('adaptive', 'popularity', 'carousel')
    }
    
    # Get the mapped parameters based on the group index
    type_param, baseline_param, ui_param = mapping[group_index]

    # Construct the redirect URL
    redirect_url = url_for('landing_page', type=type_param, baseline=baseline_param, ui=ui_param)
    
    # Redirect to the /study route with the selected parameters
    return redirect(redirect_url)


if __name__ == '__main__':
    import math
    from collections import defaultdict

    # ── ANSI colour helpers ───────────────────────────────────────────────────
    _RESET   = "\033[0m"
    _BOLD    = "\033[1m"
    _CYAN    = "\033[96m"
    _MAGENTA = "\033[95m"

    def _tbl_row(cells, widths, sep="│"):
        return sep + sep.join(str(cell).ljust(w)[:w] for cell, w in zip(cells, widths)) + sep

    def _tbl_div(widths, l="├", m="┼", r="┤", h="─"):
        return l + m.join(h * w for w in widths) + r

    def _tbl_top(widths):
        return "┌" + "┬".join("─" * w for w in widths) + "┐"

    def _tbl_bot(widths):
        return "└" + "┴".join("─" * w for w in widths) + "┘"

    # ── Load data & train implicit model (via MovieRecommender) ───────────────
    print(f"\n{_BOLD}{_CYAN}{'━'*70}{_RESET}")
    print(f"{_BOLD}{_CYAN}  Loading data and training models…{_RESET}")
    print(f"{_BOLD}{_CYAN}{'━'*70}{_RESET}")

    movies_df_main  = load_movies(MOVIES_PATH)
    ratings_df_main = load_ratings(RATINGS_PATH)
    rec_main = MovieRecommender(movies_df_main, ratings_df_main)

    # ── Train Explicit MF model (spotlight ExplicitFactorizationModel) ────────
    print("Training Explicit MF model…")
    expl_model = ExplicitFactorizationModel(n_iter=10, random_state=np.random.RandomState(42))
    expl_model.fit(rec_main.train_interactions)
    expl_train_rmse = float(rmse_score(expl_model, rec_main.train_interactions))
    expl_test_rmse  = float(rmse_score(expl_model, rec_main.test_interactions))

    # ── Metric helpers ────────────────────────────────────────────────────────
    def _ndcg_at_k(ranked, relevant, k):
        dcg  = sum(1.0 / math.log2(r + 1)
                   for r, item in enumerate(ranked[:k], 1) if item in relevant)
        ideal = min(len(relevant), k)
        idcg  = sum(1.0 / math.log2(r + 1) for r in range(1, ideal + 1))
        return dcg / idcg if idcg > 0 else 0.0

    def _prec_at_k(ranked, relevant, k):
        return sum(1 for i in ranked[:k] if i in relevant) / k if k > 0 else 0.0

    def _eval_split(model, interactions_df, k=10, threshold=4.0):
        n_items    = len(rec_main.item_id_map)
        all_items_arr = np.arange(n_items)
        ndcg_vals, prec_vals = [], []
        for u in interactions_df["user_id"].unique():
            mask     = interactions_df["user_id"] == u
            relevant = set(interactions_df.loc[
                mask & (interactions_df["rating"] >= threshold), "item_id"])
            if not relevant:
                continue
            scores = model.predict(u, all_items_arr)
            ranked = list(np.argsort(-scores))
            ndcg_vals.append(_ndcg_at_k(ranked, relevant, k))
            prec_vals.append(_prec_at_k(ranked, relevant, k))
        ndcg = float(np.mean(ndcg_vals)) if ndcg_vals else 0.0
        prec = float(np.mean(prec_vals)) if prec_vals else 0.0
        return ndcg, prec

    # Build DataFrames from split interactions
    train_df_main = pd.DataFrame({
        "user_id": rec_main.train_interactions.user_ids.astype(int),
        "item_id": rec_main.train_interactions.item_ids.astype(int),
        "rating":  rec_main.train_interactions.ratings.astype(float),
    })
    test_df_main = pd.DataFrame({
        "user_id": rec_main.test_interactions.user_ids.astype(int),
        "item_id": rec_main.test_interactions.item_ids.astype(int),
        "rating":  rec_main.test_interactions.ratings.astype(float),
    })

    print("Computing NDCG & Precision for Implicit MF…")
    impl_train_ndcg, impl_train_prec = _eval_split(rec_main.model, train_df_main)
    impl_test_ndcg,  impl_test_prec  = _eval_split(rec_main.model, test_df_main)

    print("Computing NDCG & Precision for Explicit MF…")
    expl_train_ndcg, expl_train_prec = _eval_split(expl_model, train_df_main)
    expl_test_ndcg,  expl_test_prec  = _eval_split(expl_model, test_df_main)

    # ── Cosine similarity: Carousel vs Ranked List ────────────────────────────
    print("Computing Carousel vs List cosine similarity…")

    # Build internal item_id → genres and genre → [item_ids] maps
    _item_genre_map = {}
    for _, row in rec_main.movies_df[
            rec_main.movies_df["movieId"].isin(rec_main.item_id_map.values())].iterrows():
        iid = rec_main.reverse_item_id_map.get(row["movieId"])
        if iid is not None:
            _item_genre_map[iid] = row["genres"].split("|")

    _genre_to_items = defaultdict(list)
    for iid, genres in _item_genre_map.items():
        for g in genres:
            _genre_to_items[g].append(iid)

    _N_ITEMS      = len(rec_main.item_id_map)
    _TOP_K        = 10
    _NUM_CAROUSELS    = 5
    _ITEMS_PER_CAR    = 10
    _N_SAMPLE_USERS   = min(100, len(rec_main.user_id_map))
    _all_items_arr    = np.arange(_N_ITEMS)

    def _cosine_sim(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    def _carousel_set(scores_arr):
        """Return the set of internal item IDs shown in a carousel view."""
        item_scores_loc = {i: float(scores_arr[i]) for i in range(_N_ITEMS)}
        genre_sums = {g: sum(item_scores_loc.get(i, 0.0) for i in items)
                      for g, items in _genre_to_items.items()}
        top_genres = sorted(genre_sums, key=genre_sums.get, reverse=True)[:_NUM_CAROUSELS]
        shown: set = set()
        for g in top_genres:
            top_in_genre = sorted(_genre_to_items[g],
                                  key=lambda i: -item_scores_loc.get(i, 0.0))[:_ITEMS_PER_CAR]
            shown.update(top_in_genre)
        return shown

    cos_sim_impl_vals = []
    cos_sim_expl_vals = []
    for uid_int in range(_N_SAMPLE_USERS):
        # Implicit MF
        impl_scores = rec_main.model.predict(uid_int, _all_items_arr)
        list_top_impl = set(np.argsort(-impl_scores)[:_TOP_K])
        list_vec_impl = np.array([1.0 if i in list_top_impl else 0.0 for i in range(_N_ITEMS)])
        car_items_impl = _carousel_set(impl_scores)
        car_vec_impl   = np.array([1.0 if i in car_items_impl else 0.0 for i in range(_N_ITEMS)])
        cos_sim_impl_vals.append(_cosine_sim(list_vec_impl, car_vec_impl))

        # Explicit MF
        expl_scores = expl_model.predict(uid_int, _all_items_arr)
        list_top_expl = set(np.argsort(-expl_scores)[:_TOP_K])
        list_vec_expl = np.array([1.0 if i in list_top_expl else 0.0 for i in range(_N_ITEMS)])
        car_items_expl = _carousel_set(expl_scores)
        car_vec_expl   = np.array([1.0 if i in car_items_expl else 0.0 for i in range(_N_ITEMS)])
        cos_sim_expl_vals.append(_cosine_sim(list_vec_expl, car_vec_expl))

    avg_cos_impl = float(np.mean(cos_sim_impl_vals))
    avg_cos_expl = float(np.mean(cos_sim_expl_vals))

    # ── Simulate 75-iteration trajectories: cosine similarity to gold standard ─
    print("Simulating 75-iteration adaptive trajectories (this may take a moment)…")

    _N_ITERS_SIM = 75
    _N_SIM_USERS = min(50, len(rec_main.user_id_map))

    def _scores_arr_from_dict(item_scores_dict):
        """Convert movieId-keyed score dict to an internal-ID-indexed numpy array."""
        arr = np.zeros(_N_ITEMS)
        for movie_id, sc in item_scores_dict.items():
            iid = rec_main.reverse_item_id_map.get(movie_id)
            if iid is not None:
                arr[iid] = float(sc)
        return arr

    def _shown_iids_for_ui(item_scores_dict, ui):
        """Return set of internal item IDs shown given current scores and UI type."""
        if ui == 'carousel':
            return _carousel_set(_scores_arr_from_dict(item_scores_dict))
        arr = _scores_arr_from_dict(item_scores_dict)
        return set(int(i) for i in np.argsort(-arr)[:_TOP_K])

    def _init_item_scores_for(baseline_type, ui_type):
        """Build the initial movieId-keyed item_scores for a given (baseline, ui) combo."""
        recs = get_baseline_recommendations(
            rec_main, baseline_type=baseline_type, num_items=500,
            recommendation_type=ui_type,
            random_baseline=(baseline_type == 'random')
        )
        if ui_type == 'carousel':
            return {int(item['movieId']): float(item['score'])
                    for genre_items in recs.values() for item in genre_items}
        return {int(row['movieId']): float(row['score']) for _, row in recs.iterrows()}

    _init_scores_cache = {
        ('popularity', 'list'):     _init_item_scores_for('popularity', 'list'),
        ('popularity', 'carousel'): _init_item_scores_for('popularity', 'carousel'),
        ('random',     'list'):     _init_item_scores_for('random',     'list'),
        ('random',     'carousel'): _init_item_scores_for('random',     'carousel'),
    }

    def _simulate_trajectory(baseline_type, ui_type):
        """Return array of shape (_N_ITERS_SIM,) of mean cosine similarity
        to the implicit-MF gold standard across _N_SIM_USERS simulated users."""
        init_ref   = _init_scores_cache[(baseline_type, ui_type)]
        iter_sims  = np.zeros(_N_ITERS_SIM)
        for uid_int in range(_N_SIM_USERS):
            gold_scores = rec_main.model.predict(uid_int, _all_items_arr)
            gold_top_k  = set(int(i) for i in np.argsort(-gold_scores)[:_TOP_K])
            gold_vec    = np.array([1.0 if i in gold_top_k else 0.0
                                    for i in range(_N_ITEMS)])
            item_scores = dict(init_ref)
            for it in range(_N_ITERS_SIM):
                shown = _shown_iids_for_ui(item_scores, ui_type)
                shown_vec = np.array([1.0 if i in shown else 0.0
                                      for i in range(_N_ITEMS)])
                iter_sims[it] += _cosine_sim(shown_vec, gold_vec)
                # Simulate click: pick highest-scored visible item
                scored_shown = {
                    iid: item_scores.get(rec_main.item_id_map.get(iid, -1), 0.0)
                    for iid in shown
                    if rec_main.item_id_map.get(iid) is not None
                }
                if not scored_shown:
                    continue
                best_iid       = max(scored_shown, key=scored_shown.get)
                click_movie_id = rec_main.item_id_map[best_iid]
                seen_movie_ids = [
                    rec_main.item_id_map[i] for i in shown
                    if i != best_iid and rec_main.item_id_map.get(i) is not None
                ]
                item_scores = update_item_scores(
                    dict(item_scores), click_movie_id, seen_movie_ids,
                    rec_main, update_type='adaptive'
                )
        return iter_sims / _N_SIM_USERS

    print("  Running Popularity-based trajectories…")
    pop_list_traj  = _simulate_trajectory('popularity', 'list')
    pop_car_traj   = _simulate_trajectory('popularity', 'carousel')
    print("  Running Random-based trajectories…")
    rand_list_traj = _simulate_trajectory('random', 'list')
    rand_car_traj  = _simulate_trajectory('random', 'carousel')

    # ── Print tables ──────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{_CYAN}{'═'*70}{_RESET}")
    print(f"{_BOLD}{_CYAN}  Movie Recommender — Terminal Report{_RESET}")
    print(f"{_BOLD}{_CYAN}  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'═'*70}{_RESET}\n")

    # Table 1: Cosine similarity — Carousel vs Ranked List
    print(f"{_BOLD}{_MAGENTA}  ◆ TABLE 1 — Cosine Similarity: Carousel vs Ranked List"
          f" (top-{_TOP_K}, mean over {_N_SAMPLE_USERS} users){_RESET}")
    _w1 = [20, 24]
    print("  " + _tbl_top(_w1))
    print("  " + _tbl_row([f"{_BOLD}Model{_RESET}", f"{_BOLD}Cosine Similarity{_RESET}"], _w1))
    print("  " + _tbl_div(_w1))
    print("  " + _tbl_row(["Implicit MF", f"{avg_cos_impl:.4f}"], _w1))
    print("  " + _tbl_row(["Explicit MF", f"{avg_cos_expl:.4f}"], _w1))
    print("  " + _tbl_bot(_w1))

    # Table 2: RMSE, NDCG@10 & Precision@10 — Explicit vs Implicit MF
    print(f"\n{_BOLD}{_MAGENTA}  ◆ TABLE 2 — RMSE, NDCG@10 & Precision@10:"
          f" Explicit vs Implicit MF{_RESET}")
    _w2 = [14, 12, 10, 14, 12, 12, 10]
    _h2 = ["Model", "Train RMSE", "Test RMSE",
           "Train NDCG@10", "Test NDCG@10", "Train P@10", "Test P@10"]
    print("  " + _tbl_top(_w2))
    print("  " + _tbl_row([f"{_BOLD}{h}{_RESET}" for h in _h2], _w2))
    print("  " + _tbl_div(_w2))
    print("  " + _tbl_row([
        "Implicit MF",
        f"{rec_main.train_rmse:.4f}", f"{rec_main.test_rmse:.4f}",
        f"{impl_train_ndcg:.4f}", f"{impl_test_ndcg:.4f}",
        f"{impl_train_prec:.4f}", f"{impl_test_prec:.4f}",
    ], _w2))
    print("  " + _tbl_row([
        "Explicit MF",
        f"{expl_train_rmse:.4f}", f"{expl_test_rmse:.4f}",
        f"{expl_train_ndcg:.4f}", f"{expl_test_ndcg:.4f}",
        f"{expl_train_prec:.4f}", f"{expl_test_prec:.4f}",
    ], _w2))
    print("  " + _tbl_bot(_w2))

    # Table 3: Popularity-based cosine similarity over 75 iterations
    print(f"\n{_BOLD}{_MAGENTA}  ◆ TABLE 3 — Cosine Similarity to Gold Standard:"
          f" Popularity-based (mean over {_N_SIM_USERS} users){_RESET}")
    _w3 = [12, 24, 22]
    print("  " + _tbl_top(_w3))
    print("  " + _tbl_row(
        [f"{_BOLD}Iteration{_RESET}", f"{_BOLD}Ranked List (Popularity){_RESET}",
         f"{_BOLD}Carousel (Popularity){_RESET}"], _w3))
    print("  " + _tbl_div(_w3))
    for _it in range(_N_ITERS_SIM):
        print("  " + _tbl_row(
            [str(_it + 1),
             f"{pop_list_traj[_it]:.4f}",
             f"{pop_car_traj[_it]:.4f}"], _w3))
    print("  " + _tbl_bot(_w3))

    # Table 4: Random-based cosine similarity over 75 iterations
    print(f"\n{_BOLD}{_MAGENTA}  ◆ TABLE 4 — Cosine Similarity to Gold Standard:"
          f" Random-based (mean over {_N_SIM_USERS} users){_RESET}")
    _w4 = [12, 22, 20]
    print("  " + _tbl_top(_w4))
    print("  " + _tbl_row(
        [f"{_BOLD}Iteration{_RESET}", f"{_BOLD}Ranked List (Random){_RESET}",
         f"{_BOLD}Carousel (Random){_RESET}"], _w4))
    print("  " + _tbl_div(_w4))
    for _it in range(_N_ITERS_SIM):
        print("  " + _tbl_row(
            [str(_it + 1),
             f"{rand_list_traj[_it]:.4f}",
             f"{rand_car_traj[_it]:.4f}"], _w4))
    print("  " + _tbl_bot(_w4))

    app.run(debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true')
