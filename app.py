from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, commonplayerinfo, commonteamroster
from nba_api.stats.static import players, teams
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import time
import numpy as np
app = Flask(__name__)
CORS(app)

# Cache for storing trained models
MODEL_CACHE = {}

TEAM_CACHE = {}
CACHE_EXPIRATION = 24  # Cache expiration time in hours

def get_player_id(player_name):
    player_list = players.get_players()
    for player in player_list:
        if player['full_name'].lower() == player_name.lower():
            return player['id']
    return None

def get_team_id(team_abbr):
    team_list = teams.get_teams()
    for team in team_list:
        if team['abbreviation'].lower() == team_abbr.lower():
            return team['id'], team['abbreviation']
    return None, None

# Function to get all team IDs and abbreviations

def get_all_teams():

    team_list = teams.get_teams()

    return {team['id']: team['abbreviation'] for team in team_list}

# Function to fetch team defensive stats with rate limiting and caching
def get_team_defensive_stats(season="2024-25"):
    cache_key = f"team_defense_{season}"
    
    # Check if in cache and not expired
    if cache_key in TEAM_CACHE:
        cache_data = TEAM_CACHE[cache_key]
        if datetime.now() - cache_data['timestamp'] < timedelta(hours=CACHE_EXPIRATION):
            return cache_data['data']
    
    try:
        # Get overall team defensive stats
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame"
        )
        time.sleep(1)  # Rate limiting
        
        team_defense_df = team_stats.get_data_frames()[0]
        
        # Add opponent shooting stats
        team_opp_shooting = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Opponent",
            per_mode_detailed="PerGame"
        )
        time.sleep(1)  # Rate limiting
        
        team_opp_df = team_opp_shooting.get_data_frames()[0]
        
        # Merge datasets
        team_defense_df = pd.merge(
            team_defense_df,
            team_opp_df[['TEAM_ID', 'OPP_FG_PCT', 'OPP_FG3_PCT', 'OPP_FT_PCT', 'OPP_PTS']],
            on='TEAM_ID'
        )
        
        # Prepare team dictionary for fast lookups
        all_teams = get_all_teams()
        
        # Add team abbreviation for easier reference
        team_defense_df['TEAM_ABBREVIATION'] = team_defense_df['TEAM_ID'].map(all_teams)
        
        # Cache the result
        TEAM_CACHE[cache_key] = {
            'data': team_defense_df,
            'timestamp': datetime.now()
        }
        
        return team_defense_df
        
    except Exception as e:
        print(f"Error retrieving team defensive stats: {e}")
        return pd.DataFrame()

# Function to get a specific team's defensive metrics
def get_defensive_metrics_for_team(team_abbr, season="2024-25"):
    team_id, abbr = get_team_id(team_abbr)
    if not team_id:
        return {}
    
    # Get league-wide defensive stats
    all_defense = get_team_defensive_stats(season)
    
    if all_defense.empty:
        return {}
    
    # Find the specific team
    team_defense = all_defense[all_defense['TEAM_ID'] == team_id]
    
    if team_defense.empty:
        return {}
    
    # Return a dictionary of defensive metrics
    metrics = {
        'DEF_RATING': team_defense['DEF_RATING'].values[0],
        'OPP_PTS': team_defense['OPP_PTS'].values[0],
        'OPP_FG_PCT': team_defense['OPP_FG_PCT'].values[0],
        'OPP_FG3_PCT': team_defense['OPP_FG3_PCT'].values[0],
        'DREB': team_defense['DREB'].values[0],
        'BLK': team_defense['BLK'].values[0],
        'STL': team_defense['STL'].values[0]
    }
    
    # Calculate percentile ranks (higher percentile = better defense)
    for metric in ['DEF_RATING', 'OPP_PTS', 'OPP_FG_PCT', 'OPP_FG3_PCT']:
        # For these metrics, lower is better
        percentile = (all_defense[metric] > team_defense[metric].values[0]).mean() * 100
        metrics[f"{metric}_PERCENTILE"] = percentile
    
    for metric in ['DREB', 'BLK', 'STL']:
        # For these metrics, higher is better
        percentile = (all_defense[metric] < team_defense[metric].values[0]).mean() * 100
        metrics[f"{metric}_PERCENTILE"] = percentile
    
    return metrics

def get_player_stats(player_name, stat):
    player_id = get_player_id(player_name)
    if not player_id:
        return ValueError("Player not found")
    
    games = playergamelog.PlayerGameLog(player_id=player_id, season="2024-25")
    data = games.get_data_frames()[0]

    data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'], format='%b %d, %Y')
    data = data.sort_values(by='GAME_DATE', ascending=False)

    game_log = data.head(10).sort_values(by='GAME_DATE')

    return game_log[['GAME_DATE', 'MATCHUP', stat, 'MIN']], None

@app.route("/api/player-stats")
def player_stats():
    player = request.args.get("player")
    stat = request.args.get("stat")

    if not player or not stat:
        return jsonify({"error": "Missing player or stat"}), 400

    stats_df, err = get_player_stats(player, stat)
    if err:
        return jsonify({"error": str(err)}), 500

    result = [
        {"date": row["GAME_DATE"].strftime("%Y-%m-%d"), "value": row[stat]}
        for _, row in stats_df.iterrows()
    ]
    return jsonify(result)

# Function to get historical player game data
def get_player_game_data(player_id, seasons=None):
    if seasons is None:
        current_year = datetime.now().year
        month = datetime.now().month
        if month >= 10:  # NBA season typically starts in October
            seasons = [f"{current_year-2}-{str(current_year-1)[2:]}", 
                      f"{current_year-1}-{str(current_year)[2:]}", 
                      f"{current_year}-{str(current_year+1)[2:]}"]
        else:
            seasons = [f"{current_year-3}-{str(current_year-2)[2:]}", 
                      f"{current_year-2}-{str(current_year-1)[2:]}", 
                      f"{current_year-1}-{str(current_year)[2:]}"]
    
    all_games = []
    for season in seasons:
        try:
            game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            games_df = game_log.get_data_frames()[0]
            if not games_df.empty:
                games_df['SEASON'] = season
                all_games.append(games_df)
        except Exception as e:
            print(f"Error retrieving data for season {season}: {e}")
            continue
    
    if not all_games:
        return pd.DataFrame()
    
    return pd.concat(all_games)

# Feature engineering function
def create_features(player_df, include_defense=True):
    """
    Create features for the model including rolling averages and other derived features
    """
    if player_df.empty:
        return pd.DataFrame()
    
    # Convert GAME_DATE to datetime if it's not already
    if 'GAME_DATE' in player_df.columns and not pd.api.types.is_datetime64_dtype(player_df['GAME_DATE']):
        player_df['GAME_DATE'] = pd.to_datetime(player_df['GAME_DATE'], format='%b %d, %Y')
    
    # Sort by date
    player_df = player_df.sort_values('GAME_DATE')
    
    # Create rolling averages (last 5, 10, 20 games)
    for stat in ['PTS', 'AST', 'REB', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'TOV']:
        if stat in player_df.columns:
            for window in [5, 10, 20]:
                player_df[f'{stat}_ROLLING_{window}'] = player_df[stat].rolling(window=window, min_periods=1).mean()
    
    # Home/Away feature
    player_df['IS_HOME'] = player_df['MATCHUP'].str.contains('vs').astype(int)
    
    # Days rest
    player_df['DAYS_REST'] = player_df['GAME_DATE'].diff().dt.days.fillna(3)
    
    # Extract opponent team
    player_df['OPPONENT'] = player_df['MATCHUP'].apply(
        lambda x: x.split()[-1].replace('@', '').replace('vs.', '')
    )
    # Add defensive metrics for opponents if requested
    if include_defense:
        # Get team defensive stats for relevant seasons
        seasons = player_df['SEASON'].unique()
        defense_data = {}
        
        for season in seasons:
            team_defense = get_team_defensive_stats(season)
            if not team_defense.empty:
                defense_data[season] = team_defense
        
        # Initialize defensive metric columns
        defensive_metrics = ['DEF_RATING', 'OPP_PTS', 'OPP_FG_PCT', 'OPP_FG3_PCT', 
                           'DREB', 'BLK', 'STL']
        
        for metric in defensive_metrics:
            player_df[f'OPP_{metric}'] = np.nan
        
        # Fill in defensive metrics for each game
        for idx, row in player_df.iterrows():
            if row['SEASON'] in defense_data:
                season_defense = defense_data[row['SEASON']]
                opp_defense = season_defense[season_defense['TEAM_ABBREVIATION'] == row['OPPONENT']]
                
                if not opp_defense.empty:
                    for metric in defensive_metrics:
                        if metric in opp_defense.columns:
                            player_df.at[idx, f'OPP_{metric}'] = opp_defense[metric].values[0]

    return player_df

# Function to train XGBoost model for a specific stat
def train_xgboost_model(player_id, stat):
    # Check if model is in cache and not expired
    cache_key = f"{player_id}_{stat}"
    if cache_key in MODEL_CACHE:
        model_info = MODEL_CACHE[cache_key]
        if datetime.now() - model_info['timestamp'] < timedelta(hours=CACHE_EXPIRATION):
            print(f"Using cached model for {player_id} - {stat}")
            return model_info['model'], model_info['mae'], model_info['feature_cols']
    
    # Get historical data
    player_data = get_player_game_data(player_id)
    if player_data.empty:
        return None, "No historical data available", None
    
    # Create features with defense metrics included
    player_features = create_features(player_data, include_defense=True)
    
    # Define your basic and defensive features
    basic_features = [
        'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT',
        'PTS_ROLLING_5', 'PTS_ROLLING_10', 'PTS_ROLLING_20',
        'AST_ROLLING_5', 'AST_ROLLING_10', 'AST_ROLLING_20',
        'REB_ROLLING_5', 'REB_ROLLING_10', 'REB_ROLLING_20',
        'MIN_ROLLING_5', 'MIN_ROLLING_10', 'MIN_ROLLING_20',
        'IS_HOME', 'DAYS_REST'
    ]
    defensive_features = [
        'OPP_DEF_RATING', 'OPP_OPP_PTS', 'OPP_OPP_FG_PCT', 'OPP_OPP_FG3_PCT',
        'OPP_DREB', 'OPP_BLK', 'OPP_STL'
    ]
    # Combine both sets into one list of potential features
    potential_features = basic_features + defensive_features

    # Filter columns: only include those that exist and have sufficient non-null values (for example >50%)
    feature_cols = []
    for col in potential_features:
        if col in player_features.columns and player_features[col].notna().sum() > len(player_features) * 0.5:
            feature_cols.append(col)
    
    # Ensure the target stat exists
    if stat not in player_features.columns:
        return None, f"Stat {stat} not found in player data", None
    
    if len(feature_cols) < 3:
        return None, "Not enough features available for prediction", None
    
    # Prepare X and y
    X = player_features[feature_cols]
    y = player_features[stat]
    
    # Handle missing values by replacing them with the column averages
    X = X.fillna(X.mean())
    
    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train an XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model using MAE
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cache the model along with scaler and feature columns
    MODEL_CACHE[cache_key] = {
        'model': model,
        'scaler': scaler,
        'mae': mae,
        'feature_cols': feature_cols,
        'timestamp': datetime.now()
    }
    
    return model, mae, feature_cols

# Function to predict next game stats
def predict_next_game_stats(player_id, stat, opponent, is_home=True, days_rest=1):
    # Try to fetch a cached model
    model_info = MODEL_CACHE.get(f"{player_id}_{stat}")

    opp_defense = get_defensive_metrics_for_team(opponent)
    
    if not model_info:
        model, mae, feature_cols = train_xgboost_model(player_id, stat)
        if not model:
            return None, mae  # mae contains error message if model is None
        # Update model_info from cache after training
        model_info = MODEL_CACHE.get(f"{player_id}_{stat}")
    
    # Now unpack needed items
    model = model_info['model']
    scaler = model_info['scaler']
    mae = model_info['mae']
    feature_cols = model_info['feature_cols']
    
    # Get player's recent data for feature creation
    recent_games = get_player_game_data(player_id, seasons=["2024-25"])
    if recent_games.empty:
        return None, "No recent games found"
    
    # Create features
    features_df = create_features(recent_games)
    if features_df.empty:
        return None, "Failed to create features"
    
    # Get the last 3 games if available and compute a weighted average of their numeric features
    if len(features_df) >= 3:
        recent_games = features_df.iloc[-3:].copy()
        weights = [0.4, 0.4, 0.2]  # Define weights
        # Select only numeric columns to avoid multiplying non-numerics (e.g., datetime)
        numeric_games = recent_games.select_dtypes(include=[np.number])
        averaged_features = (numeric_games.multiply(weights, axis=0)).sum() / sum(weights)
        next_game_features = averaged_features.to_frame().T
    else:
        next_game_features = features_df.iloc[-1:].copy()

    # Continue to update next_game_features with opponent info and upcoming game details:
    next_game_features['OPPONENT'] = opponent
    next_game_features['IS_HOME'] = int(is_home)
    next_game_features['DAYS_REST'] = days_rest

    # Add opponent defensive metrics
    for metric, value in opp_defense.items():
        next_game_features[f'OPP_{metric}'] = value
    
    # Prepare and scale features for prediction
    available_features = [col for col in feature_cols if col in next_game_features.columns]
    X_pred = next_game_features[available_features].fillna(next_game_features[available_features].mean())
    
    # Ensure we have all necessary features, fill missing ones with averages
    for col in feature_cols:
        if col not in X_pred.columns:
            X_pred[col] = 0  # Default value if missing
    
    X_pred = X_pred[feature_cols]  # Reorder columns to match training
    
    # Scale features
    X_pred_scaled = scaler.transform(X_pred)
    
    # Make prediction
    prediction = model.predict(X_pred_scaled)[0]
    
    return prediction, mae

# New endpoint for predictions
@app.route("/api/predict-stat")
def predict_stat():
    player = request.args.get("player")
    stat = request.args.get("stat")
    opponent = request.args.get("opponent")
    is_home = request.args.get("is_home", "true").lower() == "true"
    days_rest = int(request.args.get("days_rest", 1))
    
    if not player or not stat or not opponent:
        return jsonify({"error": "Missing player, stat, or opponent"}), 400
    
    player_id = get_player_id(player)
    if not player_id:
        return jsonify({"error": "Player not found"}), 404
    
    prediction, mae = predict_next_game_stats(player_id, stat, opponent, is_home, days_rest)
    
    if prediction is None:
        return jsonify({"error": str(mae)}), 500
    
    return jsonify({
        "player": player,
        "stat": stat,
        "opponent": opponent,
        "is_home": is_home,
        "days_rest": days_rest,
        "prediction": round(float(prediction), 1),
        "margin_error": round(float(mae), 2)
    })

# New endpoint to get player's last 10 games with predictions
@app.route("/api/player-stats-with-predictions")
def player_stats_with_predictions():
    player = request.args.get("player")
    stat = request.args.get("stat")

    if not player or not stat:
        return jsonify({"error": "Missing player or stat"}), 400
    
    player_id = get_player_id(player)
    if not player_id:
        return jsonify({"error": "Player not found"}), 404

    # Get actual stats
    stats_df, err = get_player_stats(player, stat)
    if err:
        return jsonify({"error": str(err)}), 500
    
    # Train model if needed
    model, mae, _ = train_xgboost_model(player_id, stat)
    if not model:
        return jsonify({"error": str(mae)}), 500
    
    # Format result with both actual values and what model would have predicted
    result = []
    for _, row in stats_df.iterrows():
        # For each game, we would need to predict what would have happened
        # This is a simplified approach - in a real app, you'd need to use data from before this game
        game_date = row["GAME_DATE"].strftime("%Y-%m-%d")
        actual_value = float(row[stat])
        
        # Get opponent from matchup
        opponent = row["MATCHUP"].split()[-1].replace('@', '').replace('vs.', '')
        is_home = "vs" in row["MATCHUP"]
        
        result.append({
            "date": game_date,
            "matchup": row["MATCHUP"],
            "actual": actual_value,
            "minutes": float(row["MIN"]),
            "opponent": opponent,
            "is_home": is_home
        })
    
    return jsonify({
        "player": player,
        "stat": stat,
        "games": result,
        "model_accuracy": round(float(mae), 2)
    })

@app.route("/api/predict-all")
def predict_all():
    player = request.args.get("player")
    opponent = request.args.get("opponent")
    is_home = request.args.get("is_home", "true").lower() == "true"
    days_rest = int(request.args.get("days_rest", 1))

    if not player or not opponent:
        return jsonify({"error": "Missing player or opponent"}), 400

    player_id = get_player_id(player)
    if not player_id:
        return jsonify({"error": "Player not found"}), 404

    stats_to_predict = ["PTS", "AST", "REB"]
    predictions = {}
    errors = {}

    for stat in stats_to_predict:
        prediction, mae = predict_next_game_stats(player_id, stat, opponent, is_home, days_rest)
        if prediction is None:
            errors[stat] = mae  # mae here contains an error message
        else:
            predictions[stat] = {
                "prediction": round(float(prediction), 1),
                "margin_error": round(float(mae), 2)
            }

    if errors:
        return jsonify({"error": errors}), 500

    return jsonify({
        "player": player,
        "opponent": opponent,
        "is_home": is_home,
        "days_rest": days_rest,
        "predictions": predictions
    })

@app.route("/")
def index():
    return "NBA Stats API is running!"

if __name__ == "__main__":
    app.run(debug=True)