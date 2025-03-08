import logging

# At the top of your script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ufc_predictor")

print("Script is starting...")

import pandas as pd
print("Imported pandas")
import numpy as np
print("Imported numpy")
import yaml
print("Imported yaml")
from datetime import datetime
print("Imported datetime")
from sklearn.preprocessing import StandardScaler
print("Imported StandardScaler")
from sklearn.model_selection import train_test_split
print("Imported train_test_split")
from xgboost import XGBClassifier
print("Imported XGBClassifier")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Imported metrics")
import json
import os
import argparse
import sys

def load_fight_data():
    """Load all UFC fight data"""
    try:
        with open("scrape_ufc_stats_config.yaml") as f:
            config = yaml.safe_load(f)
        
        # Check if files exist
        for file_key in ['fighter_details_file_name', 'fighter_tott_file_name', 
                        'fight_results_file_name', 'fight_stats_file_name']:
            if not os.path.exists(config[file_key]):
                print(f"Warning: {config[file_key]} not found")
        
        fighters = pd.read_csv(config['fighter_details_file_name'])
        fighter_stats = pd.read_csv(config['fighter_tott_file_name'])
        fights = pd.read_csv(config['fight_results_file_name'])
        fight_stats = pd.read_csv(config['fight_stats_file_name'])
        
        return fighters, fighter_stats, fights, fight_stats
    except Exception as e:
        print(f"Error loading fight data: {str(e)}")
        raise

def clean_fighter_name(name):
    """Clean fighter name to standardize format"""
    # Remove common prefixes/suffixes and special characters
    name = name.lower().strip()
    name = name.replace("'", "").replace('"', '')
    # Remove nicknames in parentheses or quotes if present
    if '(' in name:
        name = name.split('(')[0].strip()
    return name

def find_best_fighter_match(search_name, df, column='FIGHTER', debug=False):
    """Find the best matching fighter name in the dataframe using fuzzy matching"""
    search_name = clean_fighter_name(search_name)
    search_parts = search_name.split()
    
    if debug:
        print(f"\nSearching for: {search_name}")
        print(f"Search parts: {search_parts}")
    
    # First try exact match after cleaning
    df['clean_name'] = df[column].apply(clean_fighter_name)
    mask = df['clean_name'] == search_name
    matches = df[mask]
    if len(matches) > 0:
        if debug:
            print(f"Found exact match: {matches.iloc[0][column]}")
        return matches.iloc[0]
    
    # Then try matching last name and first name separately
    if len(search_parts) > 1:
        last_name = search_parts[-1]
        first_name = search_parts[0]
        
        # Try matching both first and last name
        mask = (df['clean_name'].str.contains(first_name, case=False, na=False) & 
                df['clean_name'].str.contains(last_name, case=False, na=False))
        matches = df[mask]
        if len(matches) > 0:
            if debug:
                print(f"Found first+last match: {matches.iloc[0][column]}")
            return matches.iloc[0]
        
        # Try matching just last name if it's distinctive enough (longer than 3 chars)
        if len(last_name) > 3:
            mask = df['clean_name'].str.contains(last_name, case=False, na=False)
            matches = df[mask]
            if len(matches) > 0:
                if debug:
                    print(f"Found last name match: {matches.iloc[0][column]}")
                return matches.iloc[0]
    
    if debug:
        print("No match found")
    return None

def create_fight_features(fight_stats_df, fighter_stats_df, fights_df):
    """Create features for model training"""
    features = []
    labels = []
    
    print(f"\nTotal fights in dataset: {len(fights_df)}")
    print("\nDataset sizes:")
    print(f"Fight stats: {len(fight_stats_df)} rows")
    print(f"Fighter stats: {len(fighter_stats_df)} rows")
    print(f"Fights: {len(fights_df)} rows")
    
    # Sample some columns to understand data format
    print("\nSample data formats:")
    if not fight_stats_df.empty:
        print("Fight stats sample:")
        for col in ['SIG.STR.', 'TD', 'SUB.ATT']:
            if col in fight_stats_df.columns:
                print(f"{col} example: {fight_stats_df[col].iloc[0] if len(fight_stats_df) > 0 else 'N/A'}")
    
    # Create two entries per fight - one from each fighter's perspective
    for idx, fight in fights_df.iterrows():
        if pd.isna(fight['OUTCOME']):
            continue
            
        # Split the bout into fighter names
        fighters = fight['BOUT'].split(' vs. ')
        if len(fighters) != 2:
            if idx < 5:
                print(f"Skipping fight with invalid format: {fight['BOUT']}")
            continue
            
        fighter_a = fighters[0].strip()
        fighter_b = fighters[1].strip()
        
        if idx < 5:  # Print first 5 fights processing details
            print(f"\nProcessing fight {idx}:")
            print(f"Fighter A: {fighter_a}")
            print(f"Fighter B: {fighter_b}")
        
        try:
            # Get fighter stats
            phys_a = find_best_fighter_match(fighter_a, fighter_stats_df, debug=(idx < 5))
            phys_b = find_best_fighter_match(fighter_b, fighter_stats_df, debug=(idx < 5))
            
            if phys_a is None or phys_b is None:
                if idx < 5:
                    print(f"Could not find physical stats for {fighter_a} vs {fighter_b}")
                continue
            
            # Get fight stats - use average of past fights
            fighter_a_stats = fight_stats_df[
                fight_stats_df['FIGHTER'].apply(lambda x: 
                    clean_fighter_name(str(x)) == clean_fighter_name(fighter_a))
            ]
            
            fighter_b_stats = fight_stats_df[
                fight_stats_df['FIGHTER'].apply(lambda x: 
                    clean_fighter_name(str(x)) == clean_fighter_name(fighter_b))
            ]
            
            if len(fighter_a_stats) == 0 or len(fighter_b_stats) == 0:
                if idx < 5:
                    print(f"No fight stats found for {fighter_a} or {fighter_b}")
                continue
            
            # Extract numeric values safely
            def extract_numeric(val, default=0):
                if pd.isna(val):
                    return default
                if isinstance(val, (int, float)):
                    return val
                if isinstance(val, str) and ' of ' in val:
                    try:
                        return float(val.split(' of ')[0])
                    except:
                        return default
                try:
                    return float(val)
                except:
                    return default
            
            # Calculate average stats for each fighter
            fighter_a_avg = fighter_a_stats.mean(numeric_only=True)
            fighter_b_avg = fighter_b_stats.mean(numeric_only=True)
            
            # Create feature set from A's perspective (A vs B)
            feature_set_a = {
                'sig_str_diff': extract_numeric(fighter_a_avg.get('SIG.STR.', 0)) - 
                               extract_numeric(fighter_b_avg.get('SIG.STR.', 0)),
                'td_diff': extract_numeric(fighter_a_avg.get('TD', 0)) - 
                          extract_numeric(fighter_b_avg.get('TD', 0)),
                'sub_att_diff': extract_numeric(fighter_a_avg.get('SUB.ATT', 0)) - 
                               extract_numeric(fighter_b_avg.get('SUB.ATT', 0)),
                'height_diff': convert_height_to_inches(phys_a['HEIGHT']) - 
                             convert_height_to_inches(phys_b['HEIGHT']),
                'reach_diff': convert_reach_to_inches(phys_a['REACH']) - 
                            convert_reach_to_inches(phys_b['REACH'])
            }
            
            # Create feature set from B's perspective (B vs A) - reverse the differences
            feature_set_b = {k: -v for k, v in feature_set_a.items()}
            
            # Determine outcome (1 for win, 0 for loss)
            outcome = fight['OUTCOME'].split('/')[0].lower()
            
            # Only add both perspectives if we have a clear winner
            if outcome == 'w':
                # A won, B lost
                features.append(feature_set_a)
                labels.append(1)
                features.append(feature_set_b)
                labels.append(0)
            elif outcome == 'l':
                # A lost, B won
                features.append(feature_set_a)
                labels.append(0)
                features.append(feature_set_b)
                labels.append(1)
            # Skip draws
            
            if len(features) % 200 == 0:  # Adjusted since we're adding 2 at a time
                print(f"Successfully created features for {len(features)} examples")
                
        except Exception as e:
            print(f"Error processing fight {fighter_a} vs {fighter_b}: {str(e)}")
            continue
    
    print(f"\nTotal features created: {len(features)}")
    print(f"Total labels created: {len(labels)}")
    if len(labels) > 0:
        print(f"Label distribution: {np.bincount(labels)}")  # Show distribution of wins/losses
    
    # Ensure features and labels have the same length
    if len(features) != len(labels):
        print(f"WARNING: Features and labels have different lengths. Truncating to match.")
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]
        print(f"After truncation: {len(features)} features, {len(labels)} labels")
    
    if len(features) == 0:
        raise ValueError("No features were created. Check the data processing.")
        
    return pd.DataFrame(features), np.array(labels)

def convert_height_to_inches(height_str):
    """Convert height string to inches"""
    if pd.isna(height_str) or height_str == '--':
        return 0
    try:
        if "'" in height_str:
            parts = height_str.replace('"', '').split("'")
            feet = int(parts[0])
            inches = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            return feet * 12 + inches
        return float(height_str)
    except:
        return 0

def convert_reach_to_inches(reach_str):
    """Convert reach string to inches"""
    if pd.isna(reach_str) or reach_str == '--':
        return 0
    try:
        return float(reach_str.replace('"', ''))
    except:
        return 0

def train_prediction_model(features, labels):
    """Train XGBoost model for fight prediction"""
    print(f"\nTraining model with {len(features)} examples")
    
    # Convert to numpy arrays if needed
    if isinstance(features, pd.DataFrame):
        X = features.values
    else:
        X = features
        
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} examples")
    print(f"Test set: {X_test.shape[0]} examples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    
    print("\nModel Performance:")
    print(f"Train Accuracy: {accuracy_score(y_train, train_preds):.3f}")
    print(f"Test Accuracy: {accuracy_score(y_test, test_preds):.3f}")
    print(f"Test Precision: {precision_score(y_test, test_preds):.3f}")
    print(f"Test Recall: {recall_score(y_test, test_preds):.3f}")
    print(f"Test F1 Score: {f1_score(y_test, test_preds):.3f}")
    
    # Feature importance
    if isinstance(features, pd.DataFrame):
        feature_importance = model.feature_importances_
        feature_names = features.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        for i, row in importance_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Create a model wrapper that includes the scaler
    model_wrapper = {
        'model': model,
        'scaler': scaler
    }
    
    return model_wrapper

def advanced_fight_analysis(fighter1, fighter2, fighter_stats_df, fight_stats_df, fights_df):
    """
    Analyze UFC fights using advanced MMA-specific metrics
    """
    # Get fighter stats
    phys_1 = find_best_fighter_match(fighter1, fighter_stats_df, debug=False)
    phys_2 = find_best_fighter_match(fighter2, fighter_stats_df, debug=False)
    
    if phys_1 is None or phys_2 is None:
        return None, "Missing fighter data"
    
    # Get fight history
    fighter1_fights = fights_df[
        fights_df['BOUT'].apply(lambda x: 
            clean_fighter_name(fighter1) in clean_fighter_name(x))
    ].sort_values('EVENT', ascending=False).head(5)
    
    fighter2_fights = fights_df[
        fights_df['BOUT'].apply(lambda x: 
            clean_fighter_name(fighter2) in clean_fighter_name(x))
    ].sort_values('EVENT', ascending=False).head(5)
    
    # Get fight stats
    fighter1_stats = fight_stats_df[
        fight_stats_df['FIGHTER'].apply(lambda x: 
            clean_fighter_name(str(x)) == clean_fighter_name(fighter1))
    ]
    
    fighter2_stats = fight_stats_df[
        fight_stats_df['FIGHTER'].apply(lambda x: 
            clean_fighter_name(str(x)) == clean_fighter_name(fighter2))
    ]
    
    # Extract weight class from most recent fight
    weight_class = None
    if len(fighter1_fights) > 0:
        weight_class = fighter1_fights.iloc[0]['WEIGHTCLASS'] if 'WEIGHTCLASS' in fighter1_fights.columns else None
    elif len(fighter2_fights) > 0:
        weight_class = fighter2_fights.iloc[0]['WEIGHTCLASS'] if 'WEIGHTCLASS' in fighter2_fights.columns else None
    
    # 1. Historical Trends - Weight class finish rates
    ko_rate = 0.30  # Default
    sub_rate = 0.20  # Default
    dec_rate = 0.50  # Default
    
    # Adjust based on weight class
    if weight_class:
        if 'Heavyweight' in weight_class:
            ko_rate, sub_rate, dec_rate = 0.48, 0.17, 0.35
        elif 'Light Heavyweight' in weight_class:
            ko_rate, sub_rate, dec_rate = 0.42, 0.18, 0.40
        elif 'Middleweight' in weight_class:
            ko_rate, sub_rate, dec_rate = 0.38, 0.22, 0.40
        elif 'Welterweight' in weight_class:
            ko_rate, sub_rate, dec_rate = 0.35, 0.25, 0.40
        elif 'Lightweight' in weight_class:
            ko_rate, sub_rate, dec_rate = 0.30, 0.28, 0.42
        elif 'Featherweight' in weight_class:
            ko_rate, sub_rate, dec_rate = 0.28, 0.30, 0.42
        elif 'Bantamweight' in weight_class:
            ko_rate, sub_rate, dec_rate = 0.25, 0.30, 0.45
        elif 'Flyweight' in weight_class:
            ko_rate, sub_rate, dec_rate = 0.20, 0.25, 0.55
        elif "Women's Strawweight" in weight_class:
            ko_rate, sub_rate, dec_rate = 0.15, 0.17, 0.68
        elif "Women's" in weight_class:
            ko_rate, sub_rate, dec_rate = 0.18, 0.20, 0.62
    
    # 2. Fighter Metrics
    def extract_numeric(val, default=0):
        if pd.isna(val):
            return default
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, str) and ' of ' in val:
            try:
                return float(val.split(' of ')[0])
            except:
                return default
        try:
            return float(val)
        except:
            return default
    
    # Calculate average stats
    fighter1_avg = fighter1_stats.mean(numeric_only=True) if len(fighter1_stats) > 0 else pd.Series()
    fighter2_avg = fighter2_stats.mean(numeric_only=True) if len(fighter2_stats) > 0 else pd.Series()
    
    # Strike differential
    f1_sig_strikes = extract_numeric(fighter1_avg.get('SIG.STR.', 0))
    f2_sig_strikes = extract_numeric(fighter2_avg.get('SIG.STR.', 0))
    strike_diff = f1_sig_strikes - f2_sig_strikes
    
    # Takedown metrics
    f1_td = extract_numeric(fighter1_avg.get('TD', 0))
    f2_td = extract_numeric(fighter2_avg.get('TD', 0))
    f1_td_pct = extract_numeric(fighter1_avg.get('TD %', 0))
    f2_td_pct = extract_numeric(fighter2_avg.get('TD %', 0))
    
    # Submission attempts
    f1_sub_att = extract_numeric(fighter1_avg.get('SUB.ATT', 0))
    f2_sub_att = extract_numeric(fighter2_avg.get('SUB.ATT', 0))
    
    # 3. Recency Bias
    f1_recent_wins = 0
    f2_recent_wins = 0
    f1_losing_streak = 0
    f2_losing_streak = 0
    
    # Calculate recent performance
    for i, fight in enumerate(fighter1_fights.iterrows()):
        outcome = fight[1]['OUTCOME'].split('/')[0] if 'OUTCOME' in fight[1] else None
        if outcome == 'W':
            f1_recent_wins += 1
            f1_losing_streak = 0
        elif outcome == 'L':
            f1_losing_streak += 1
    
    for i, fight in enumerate(fighter2_fights.iterrows()):
        outcome = fight[1]['OUTCOME'].split('/')[0] if 'OUTCOME' in fight[1] else None
        if outcome == 'W':
            f2_recent_wins += 1
            f2_losing_streak = 0
        elif outcome == 'L':
            f2_losing_streak += 1
    
    # Apply recency bias
    recency_factor1 = 1.0 + (0.15 * (f1_recent_wins / max(1, len(fighter1_fights))))
    recency_factor2 = 1.0 + (0.15 * (f2_recent_wins / max(1, len(fighter2_fights))))
    
    # Losing streak penalty
    if f1_losing_streak >= 2:
        recency_factor1 *= 0.85
    if f2_losing_streak >= 2:
        recency_factor2 *= 0.85
    
    # 4. Grappling-Striking Nexus
    # Estimate submission defense (default 0.5 if unknown)
    f1_sub_def = 0.5
    f2_sub_def = 0.5
    
    # Calculate ground control risk
    f1_ground_risk = f2_td * (1 - f1_sub_def)
    f2_ground_risk = f1_td * (1 - f2_sub_def)
    
    # 5. Physical attributes
    height_diff = convert_height_to_inches(phys_1['HEIGHT']) - convert_height_to_inches(phys_2['HEIGHT'])
    reach_diff = convert_reach_to_inches(phys_1['REACH']) - convert_reach_to_inches(phys_2['REACH'])
    
    # Calculate overall score
    f1_score = (
        (strike_diff * 0.3) + 
        (f1_td * f1_td_pct * 0.2) + 
        (f1_sub_att * 0.1) + 
        (height_diff * 0.05) + 
        (reach_diff * 0.05) - 
        (f1_ground_risk * 0.2)
    ) * recency_factor1
    
    f2_score = (
        (-strike_diff * 0.3) + 
        (f2_td * f2_td_pct * 0.2) + 
        (f2_sub_att * 0.1) + 
        (-height_diff * 0.05) + 
        (-reach_diff * 0.05) - 
        (f2_ground_risk * 0.2)
    ) * recency_factor2
    
    # Calculate win probability
    total_score = abs(f1_score) + abs(f2_score)
    if total_score == 0:
        win_prob = 0.5
    else:
        win_prob = max(0.1, min(0.9, (f1_score + total_score/2) / total_score))
    
    # Predict finish method
    ko_prob = ko_rate * (strike_diff > 0)
    sub_prob = sub_rate * (f1_sub_att > f2_sub_att)
    dec_prob = 1 - ko_prob - sub_prob
    
    # Format analysis
    analysis = {
        'win_probability': win_prob,
        'ko_probability': ko_prob,
        'sub_probability': sub_prob,
        'dec_probability': dec_prob,
        'fighter1_stats': {
            'height': phys_1['HEIGHT'],
            'reach': phys_1['REACH'],
            'recent_wins': f1_recent_wins,
            'losing_streak': f1_losing_streak,
            'sig_strikes': f1_sig_strikes,
            'takedowns': f1_td,
            'sub_attempts': f1_sub_att
        },
        'fighter2_stats': {
            'height': phys_2['HEIGHT'],
            'reach': phys_2['REACH'],
            'recent_wins': f2_recent_wins,
            'losing_streak': f2_losing_streak,
            'sig_strikes': f2_sig_strikes,
            'takedowns': f2_td,
            'sub_attempts': f2_sub_att
        }
    }
    
    return win_prob, analysis

def predict_fight(model, fighter1, fighter2, fighter_stats_df, fight_stats_df, fights_df=None):
    """
    Predict the outcome of a fight
    
    Args:
        model: Trained prediction model
        fighter1: Name of first fighter
        fighter2: Name of second fighter
        fighter_stats_df: DataFrame with fighter statistics
        fight_stats_df: DataFrame with fight statistics
        fights_df: DataFrame with fight results (optional)
    
    Returns:
        Tuple of (win probability for fighter1, analysis dictionary)
    """
    # Get advanced analysis
    analysis = advanced_fight_analysis(fighter1, fighter2, fighter_stats_df, fight_stats_df, fights_df)
    
    # If we can't get advanced analysis, return a default probability
    if analysis is None:
        # Try to get basic physical stats
        phys_1 = find_best_fighter_match(fighter1, fighter_stats_df, debug=False)
        phys_2 = find_best_fighter_match(fighter2, fighter_stats_df, debug=False)
        
        if phys_1 is None or phys_2 is None:
            # If we can't even get physical stats, return 0.9 probability (heavily favor fighter1)
            # This is just a placeholder - in a real system, we'd want to handle this better
            return 0.9, None
        
        # Set default method of victory probabilities
        analysis = {
            'ko_probability': 0.45,  # 45% KO/TKO rate in UFC
            'sub_probability': 0.25,  # 25% submission rate
            'dec_probability': 0.30,  # 30% decision rate
            'fighter1_stats': {
                'height': phys_1['HEIGHT'],
                'reach': phys_1['REACH'],
                'recent_wins': 0,
                'losing_streak': 0,
                'sig_strikes': 0,
                'takedowns': 0
            },
            'fighter2_stats': {
                'height': phys_2['HEIGHT'],
                'reach': phys_2['REACH'],
                'recent_wins': 0,
                'losing_streak': 0,
                'sig_strikes': 0,
                'takedowns': 0
            }
        }
    
    # Get fighter stats
    phys_1 = find_best_fighter_match(fighter1, fighter_stats_df, debug=False)
    phys_2 = find_best_fighter_match(fighter2, fighter_stats_df, debug=False)
    
    if phys_1 is None or phys_2 is None:
        return 0.5, analysis
    
    # Get average fight stats
    fighter1_fights = fight_stats_df[fight_stats_df['FIGHTER'].str.contains(fighter1, case=False, na=False)]
    fighter2_fights = fight_stats_df[fight_stats_df['FIGHTER'].str.contains(fighter2, case=False, na=False)]
    
    fighter1_avg = fighter1_fights.mean(numeric_only=False)
    fighter2_avg = fighter2_fights.mean(numeric_only=False)
    
    def extract_numeric(val, default=0):
        if pd.isna(val) or val == '--':
            return default
        
        if isinstance(val, (int, float)):
            return val
            
        # Handle string values like "4 of 9"
        if ' of ' in str(val):
            return float(str(val).split(' of ')[0])
        
        try:
            return float(val)
        except:
            return default
    
    # Create feature set for prediction
    feature_set = {
        'sig_str_diff': extract_numeric(fighter1_avg.get('SIG.STR.', 0)) - 
                        extract_numeric(fighter2_avg.get('SIG.STR.', 0)),
        'td_diff': extract_numeric(fighter1_avg.get('TD', 0)) - 
                  extract_numeric(fighter2_avg.get('TD', 0)),
        'sub_att_diff': extract_numeric(fighter1_avg.get('SUB.ATT', 0)) - 
                       extract_numeric(fighter2_avg.get('SUB.ATT', 0)),
        'height_diff': convert_height_to_inches(phys_1['HEIGHT']) - 
                     convert_height_to_inches(phys_2['HEIGHT']),
        'reach_diff': convert_reach_to_inches(phys_1['REACH']) - 
                    convert_reach_to_inches(phys_2['REACH'])
    }
    
    # Make prediction
    features_df = pd.DataFrame([feature_set])
    
    try:
        if model is not None and isinstance(model, dict) and 'model' in model and 'scaler' in model:
            # Scale features using the same scaler used during training
            features_scaled = model['scaler'].transform(features_df)
            # Get prediction probability
            probability = model['model'].predict_proba(features_scaled)[0][1]
            return probability, analysis
        else:
            # If model is not available or not in the expected format
            print(f"Model not available for prediction. Using default probability.")
            return 0.5, analysis
    except Exception as e:
        # If model prediction fails, return 0.5 (even odds)
        print(f"Error making prediction for {fighter1} vs {fighter2}: {str(e)}")
        return 0.5, analysis

def incorporate_external_data(fighter1, fighter2, prediction_prob, event_id=None):
    """
    Incorporate external data from Fight Matrix and Tapology
    
    Args:
        fighter1: First fighter name
        fighter2: Second fighter name
        prediction_prob: Probability from our model
        event_id: Optional event ID to scrape fresh data
        
    Returns:
        Adjusted probability and confidence level
    """
    # If event_id is provided, scrape fresh data
    if event_id:
        try:
            # Load cached data first
            fight_matrix_data, tapology_data = load_cached_data(event_id)
            
            # If no cached data, scrape it
            if fight_matrix_data is None:
                # This would call the scraping functions
                # fight_matrix_data = scrape_fight_matrix_data(f"https://www.fightmatrix.com/event/{event_id}/")
                # tapology_data = scrape_tapology_data(f"https://www.tapology.com/fightcenter/events/{event_id}")
                
                # Save to cache
                # save_cached_data(event_id, fight_matrix_data, tapology_data)
                pass
        except Exception as e:
            print(f"Error loading external data: {str(e)}")
            # Fall back to default data
            fight_matrix_data = None
    else:
        # Use hardcoded data
        fight_matrix_data = {
            ("Alex Pereira", "Magomed Ankalaev"): {
                "betting_odds": {"favorite": "Alex Pereira", "implied_prob": 0.5183},
                "elo_k170": {"favorite": "Alex Pereira", "win_pct": 0.6167},
                "elo_modified": {"favorite": "Alex Pereira", "win_pct": 0.5771},
                "glicko": {"favorite": "Alex Pereira", "win_pct": 0.6140},
                "whr": {"favorite": "Alex Pereira", "win_pct": 0.6700}
            },
            ("Justin Gaethje", "Rafael Fiziev"): {
                "betting_odds": {"favorite": "Rafael Fiziev", "implied_prob": 0.5612},
                "elo_k170": {"favorite": "Justin Gaethje", "win_pct": 0.5300},
                "elo_modified": {"favorite": "Justin Gaethje", "win_pct": 0.5300},
                "glicko": {"favorite": "Justin Gaethje", "win_pct": 0.5300},
                "whr": {"favorite": "Justin Gaethje", "win_pct": 0.5300}
            },
            ("Curtis Blaydes", "Rizvan Kuniev"): {
                "betting_odds": {"favorite": "Curtis Blaydes", "implied_prob": 0.7221},
                "elo_k170": {"favorite": "Curtis Blaydes", "win_pct": 0.8902},
                "elo_modified": {"favorite": "Curtis Blaydes", "win_pct": 0.8788},
                "glicko": {"favorite": "Curtis Blaydes", "win_pct": 0.8737},
                "whr": {"favorite": "Curtis Blaydes", "win_pct": 0.7479}
            },
            ("Joshua Van", "Rei Tsuruya"): {
                "betting_odds": {"favorite": "Joshua Van", "implied_prob": 0.6326},
                "elo_k170": {"favorite": "Rei Tsuruya", "win_pct": 0.5825},
                "elo_modified": {"favorite": "Rei Tsuruya", "win_pct": 0.5991},
                "glicko": {"favorite": "Rei Tsuruya", "win_pct": 0.5437},
                "whr": {"favorite": "Rei Tsuruya", "win_pct": 0.5902}
            },
            ("Francis Marshall", "Mairon Santos"): {
                "betting_odds": {"favorite": "Mairon Santos", "implied_prob": 0.7300},
                "elo_k170": {"favorite": "Mairon Santos", "win_pct": 0.8217},
                "elo_modified": {"favorite": "Mairon Santos", "win_pct": 0.8217},
                "glicko": {"favorite": "Mairon Santos", "win_pct": 0.8244},
                "whr": {"favorite": "Mairon Santos", "win_pct": 0.7065}
            }
        }
    
    # Check if we have Fight Matrix data for this matchup
    matchup = (fighter1, fighter2)
    reverse_matchup = (fighter2, fighter1)
    
    fight_matrix_prob = None
    is_reversed = False
    
    if fight_matrix_data is not None:
        if matchup in fight_matrix_data:
            data = fight_matrix_data[matchup]
        elif reverse_matchup in fight_matrix_data:
            data = fight_matrix_data[reverse_matchup]
            is_reversed = True
        else:
            # No Fight Matrix data available for this matchup
            print(f"No external data available for {fighter1} vs {fighter2}")
            return prediction_prob, "Low"  # Return original probability with low confidence
    else:
        # No Fight Matrix data available at all
        print(f"No external data available")
        return prediction_prob, "Low"  # Return original probability with low confidence
    
    # Calculate average probability from Fight Matrix models
    models = ["betting_odds", "elo_k170", "elo_modified", "glicko", "whr"]
    total_prob = 0
    count = 0
    
    for model in models:
        if model in data:
            model_data = data[model]
            favorite = model_data["favorite"]
            
            # Get win percentage or implied probability
            if "win_pct" in model_data:
                win_pct = model_data["win_pct"]
            elif "implied_prob" in model_data:
                win_pct = model_data["implied_prob"]
            else:
                # Skip this model if no probability data
                continue
            
            # Adjust probability based on favorite
            if (not is_reversed and favorite == fighter1) or (is_reversed and favorite == fighter2):
                total_prob += win_pct
            else:
                total_prob += (1 - win_pct)
            
            count += 1
    
    if count > 0:
        fight_matrix_prob = total_prob / count
    else:
        # No valid models found
        return prediction_prob, "Low"  # Return original probability with low confidence
    
    # Combine our model with Fight Matrix data (60% our model, 40% Fight Matrix)
    combined_prob = (0.6 * prediction_prob) + (0.4 * fight_matrix_prob)
    confidence = "High" if abs(combined_prob - 0.5) > 0.15 else "Medium"
    
    return combined_prob, confidence

def load_cached_data(event_id):
    """Load cached data for an event"""
    cache_file = f"cache_{event_id}.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
            # Check if cache is fresh (less than 1 day old)
            if datetime.now().timestamp() - data.get('timestamp', 0) < 86400:
                return data.get('fight_matrix', {}), data.get('tapology', {})
    return None, None

def save_cached_data(event_id, fight_matrix_data, tapology_data):
    """Save data to cache"""
    cache_file = f"cache_{event_id}.json"
    with open(cache_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().timestamp(),
            'fight_matrix': fight_matrix_data,
            'tapology': tapology_data
        }, f)

def export_predictions_to_csv(predictions, filename="ufc_predictions.csv"):
    """Export predictions to CSV file"""
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fighter 1', 'Fighter 2', 'Predicted Winner', 
                        'Win Probability', 'Confidence', 'KO Probability', 
                        'Submission Probability', 'Decision Probability'])
        
        for pred in predictions:
            writer.writerow([
                pred['fighter1'], 
                pred['fighter2'],
                pred['winner'],
                f"{pred['win_prob']:.2f}",
                pred['confidence'],
                f"{pred.get('ko_prob', 0):.2f}",
                f"{pred.get('sub_prob', 0):.2f}",
                f"{pred.get('dec_prob', 0):.2f}"
            ])
    
    print(f"Predictions exported to {filename}")

def evaluate_past_event(event_name, event_date, model, fighter_stats_df, fight_stats_df, fights_df):
    """
    Evaluate model accuracy on a past event
    
    Args:
        event_name: Name of the event (e.g., "UFC 300")
        event_date: Date of the event in YYYY-MM-DD format
        model: Trained prediction model
        fighter_stats_df: DataFrame with fighter statistics
        fight_stats_df: DataFrame with fight statistics
        fights_df: DataFrame with fight results
    
    Returns:
        Dictionary with accuracy metrics
    """
    print(f"\n===== Evaluating Model on {event_name} ({event_date}) =====")
    
    # Print column names for debugging
    print(f"Available columns in fights dataframe: {list(fights_df.columns)}")
    
    # Filter fights for this event - adjust column names as needed
    if 'EVENT_NAME' in fights_df.columns:
        event_column = 'EVENT_NAME'
    elif 'EVENT' in fights_df.columns:
        event_column = 'EVENT'
    else:
        # Try to find a column that might contain event names
        potential_columns = [col for col in fights_df.columns if 'EVENT' in col.upper()]
        if potential_columns:
            event_column = potential_columns[0]
            print(f"Using column '{event_column}' for event names")
        else:
            print("Could not find event name column in fights dataframe")
            return None
    
    # Filter by event name
    event_fights = fights_df[
        fights_df[event_column].str.contains(event_name, case=False, na=False)
    ]
    
    if len(event_fights) == 0:
        print(f"No fights found for {event_name}")
        return None
    
    print(f"Found {len(event_fights)} fights for this event")
    
    # Extract fighter names from BOUT column
    if 'BOUT' in event_fights.columns:
        print("Extracting fighter names from BOUT column")
        
        correct_predictions = 0
        total_predictions = 0
        predictions = []
        
        for _, fight in event_fights.iterrows():
            try:
                # Extract fighter names from BOUT column (format: "Fighter 1 vs. Fighter 2")
                bout = fight['BOUT']
                print(f"Processing bout: {bout}")
                
                if 'vs.' in bout:
                    separator = 'vs.'
                elif 'vs' in bout:
                    separator = 'vs'
                else:
                    print(f"  Skipping - no 'vs' found in bout: {bout}")
                    continue
                    
                fighters = bout.split(separator)
                if len(fighters) == 2:
                    fighter1 = fighters[0].strip()
                    fighter2 = fighters[1].strip()
                    
                    print(f"  Extracted fighters: '{fighter1}' vs '{fighter2}'")
                    
                    # Determine winner from METHOD column
                    method = fight.get('METHOD', '')
                    print(f"  Method: {method}")
                    
                    # Look for fighter names in the method description
                    fighter1_first_name = fighter1.split()[0]
                    fighter2_first_name = fighter2.split()[0]
                    fighter1_last_name = fighter1.split()[-1]
                    fighter2_last_name = fighter2.split()[-1]
                    
                    if fighter1_first_name in method or fighter1_last_name in method:
                        winner = fighter1
                        print(f"  Winner determined from method: {fighter1}")
                    elif fighter2_first_name in method or fighter2_last_name in method:
                        winner = fighter2
                        print(f"  Winner determined from method: {fighter2}")
                    else:
                        # If we can't determine the winner from METHOD, use OUTCOME
                        if 'OUTCOME' in fight and not pd.isna(fight['OUTCOME']):
                            outcome = fight['OUTCOME']
                            print(f"  Using outcome: {outcome}")
                            
                            # Handle W/L format
                            if outcome == 'W/L':
                                winner = fighter1
                                print(f"  Winner from W/L: {fighter1}")
                            elif outcome == 'L/W':
                                winner = fighter2
                                print(f"  Winner from L/W: {fighter2}")
                            elif outcome.lower() == 'win':
                                winner = fighter1
                            elif outcome.lower() == 'loss':
                                winner = fighter2
                            else:
                                print(f"  Skipping - unknown outcome format: {outcome}")
                                continue
                        else:
                            # If we still can't determine the winner, skip this fight
                            print(f"  Skipping - can't determine winner")
                            continue
                    
                    # Make prediction
                    prob, analysis = predict_fight(model, fighter1, fighter2, fighter_stats_df, fight_stats_df, fights_df)
                    
                    # Incorporate external data (without scraping)
                    adjusted_prob, confidence = incorporate_external_data(fighter1, fighter2, prob)
                    
                    # Determine predicted winner
                    predicted_winner = fighter1 if adjusted_prob > 0.5 else fighter2
                    win_prob = adjusted_prob if adjusted_prob > 0.5 else 1 - adjusted_prob
                    
                    # Check if prediction was correct
                    correct = predicted_winner == winner
                    if correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    # Store prediction details
                    prediction = {
                        'event': event_name,
                        'date': event_date,
                        'fighter1': fighter1,
                        'fighter2': fighter2,
                        'actual_winner': winner,
                        'predicted_winner': predicted_winner,
                        'win_probability': win_prob,
                        'confidence': confidence,
                        'correct': correct
                    }
                    
                    if analysis:
                        prediction['ko_probability'] = analysis['ko_probability']
                        prediction['sub_probability'] = analysis['sub_probability']
                        prediction['dec_probability'] = analysis['dec_probability']
                    
                    predictions.append(prediction)
                    
                    # Print prediction vs actual
                    print(f"\n{fighter1} vs {fighter2}:")
                    print(f"  Actual winner: {winner}")
                    print(f"  Predicted winner: {predicted_winner} ({win_prob:.1%} probability)")
                    print(f"  Correct: {'✓' if correct else '✗'}")
                else:
                    print(f"  Skipping - couldn't split fighters from bout: {bout}")
            except Exception as e:
                print(f"  Error processing bout: {str(e)}")
                continue
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\nOverall accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
        
        # Export predictions to CSV
        filename = f"{event_name.replace(' ', '_').lower()}_evaluation.csv"
        export_predictions_to_csv(predictions, filename=filename)
        print(f"Evaluation results exported to {filename}")
        
        return {
            'event': event_name,
            'date': event_date,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'predictions': predictions
        }
    else:
        print("Could not find BOUT column to extract fighter names")
        return None

def main(event_id=None, use_scraping=True, evaluate_past=False, past_event=None, past_date=None):
    # Debug prints
    print("=" * 50)
    print("STARTING UFC PREDICTION SCRIPT")
    print("=" * 50)
    
    # Load data
    print("Loading fight data...")
    fighters, fighter_stats, fights, fight_stats = load_fight_data()
    
    # Print data sizes for debugging
    print(f"Loaded {len(fighters)} fighters")
    print(f"Loaded {len(fighter_stats)} fighter stats")
    print(f"Loaded {len(fights)} fights")
    print(f"Loaded {len(fight_stats)} fight stats")
    
    # Create features and train model
    print("\nCreating features and training model...")
    try:
        features, labels = create_fight_features(fight_stats, fighter_stats, fights)
        
        if len(features) < 10:
            print("Not enough features created for reliable model training.")
            print("Using a simple baseline model instead.")
            model = None
        else:
            model = train_prediction_model(features, labels)
    except Exception as e:
        print(f"Error creating features or training model: {str(e)}")
        print("Using a simple baseline model instead.")
        model = None
    
    # If evaluating past event
    if evaluate_past and past_event and past_date:
        evaluate_past_event(past_event, past_date, model, fighter_stats, fight_stats, fights)
        return
    
    # March 8, 2025 fight card
    upcoming_fights = [
        ("Magomed Ankalaev", "Alex Pereira"),
        ("Ignacio Bahamondes", "Jalin Turner"),
        ("Iasmin Lucindo", "Amanda Lemos"),
        ("Mauricio Ruffy", "Bobby Green"),
        ("Rizvan Kuniev", "Curtis Blaydes"),
        ("Armen Petrosyan", "Brunno Ferreira"),
        ("Carlos Leal", "Alex Morono"),
        ("Francis Marshall", "Mairon Santos"),
        ("Ozzy Diaz", "Djorden Santos"),
        ("Rei Tsuruya", "Joshua Van"),
        ("Rafael Fiziev", "Justin Gaethje"),
        ("John Castaneda", "Chris Gutierrez")
    ]
    
    # Store all predictions for CSV export
    all_predictions = []
    
    # Predict upcoming fights
    print("\n===== March 8, 2025 Fight Predictions =====")
    for fighter1, fighter2 in upcoming_fights:
        print(f"\n{fighter1} vs {fighter2}:")
        
        # Use advanced analysis
        prob, analysis = predict_fight(model, fighter1, fighter2, fighter_stats, fight_stats, fights)
        
        # Incorporate external data
        if use_scraping and event_id:
            adjusted_prob, confidence = incorporate_external_data(fighter1, fighter2, prob, event_id)
        else:
            adjusted_prob, confidence = incorporate_external_data(fighter1, fighter2, prob)
        
        # Create prediction record for CSV export
        prediction = {
            'fighter1': fighter1,
            'fighter2': fighter2,
            'winner': fighter1 if adjusted_prob > 0.5 else fighter2,
            'win_prob': adjusted_prob if adjusted_prob > 0.5 else 1 - adjusted_prob,
            'confidence': confidence
        }
        
        if analysis:
            # Print detailed analysis
            winner = fighter1 if adjusted_prob > 0.5 else fighter2
            win_prob = adjusted_prob if adjusted_prob > 0.5 else 1 - adjusted_prob
            
            print(f"Prediction: {winner} wins ({win_prob:.1%} probability) - {confidence} confidence")
            print(f"Method: KO/TKO ({analysis['ko_probability']:.1%}), Submission ({analysis['sub_probability']:.1%}), Decision ({analysis['dec_probability']:.1%})")
            
            # Add method probabilities to prediction record
            prediction['ko_prob'] = analysis['ko_probability']
            prediction['sub_prob'] = analysis['sub_probability']
            prediction['dec_prob'] = analysis['dec_probability']
            
            # Print fighter stats
            print(f"\n{fighter1} stats:")
            print(f"  Height: {analysis['fighter1_stats']['height']}, Reach: {analysis['fighter1_stats']['reach']}")
            print(f"  Recent wins: {analysis['fighter1_stats']['recent_wins']}, Losing streak: {analysis['fighter1_stats']['losing_streak']}")
            print(f"  Sig. strikes: {analysis['fighter1_stats']['sig_strikes']:.1f}, Takedowns: {analysis['fighter1_stats']['takedowns']:.1f}")
            
            print(f"\n{fighter2} stats:")
            print(f"  Height: {analysis['fighter2_stats']['height']}, Reach: {analysis['fighter2_stats']['reach']}")
            print(f"  Recent wins: {analysis['fighter2_stats']['recent_wins']}, Losing streak: {analysis['fighter2_stats']['losing_streak']}")
            print(f"  Sig. strikes: {analysis['fighter2_stats']['sig_strikes']:.1f}, Takedowns: {analysis['fighter2_stats']['takedowns']:.1f}")
            
            # Flag potential upsets
            if (adjusted_prob > 0.5 and adjusted_prob < 0.6) or (adjusted_prob < 0.5 and adjusted_prob > 0.4):
                print("\n⚠️ POTENTIAL UPSET ALERT: Close matchup with upset potential")
        else:
            # Fall back to simple prediction
            winner = fighter1 if adjusted_prob > 0.5 else fighter2
            win_prob = adjusted_prob if adjusted_prob > 0.5 else 1 - adjusted_prob
            
            print(f"Prediction: {winner} wins ({win_prob:.1%} probability) - {confidence} confidence")
                
            # Simple baseline prediction based on physical stats
            phys_1 = find_best_fighter_match(fighter1, fighter_stats, debug=False)
            phys_2 = find_best_fighter_match(fighter2, fighter_stats, debug=False)
            
            if phys_1 is not None and phys_2 is not None:
                print(f"{fighter1}: Height {phys_1['HEIGHT']}, Reach {phys_1['REACH']}")
                print(f"{fighter2}: Height {phys_2['HEIGHT']}, Reach {phys_2['REACH']}")
        
        # Add prediction to list
        all_predictions.append(prediction)
    
    # Export predictions to CSV
    export_predictions_to_csv(all_predictions, filename="ufc_313_predictions.csv")
    print("\nPredictions exported to ufc_313_predictions.csv")

# Add debug print before if __name__ == "__main__"
print("About to check if __name__ == '__main__'")

if __name__ == "__main__":
    print("In __name__ == '__main__' block")
    
    parser = argparse.ArgumentParser(description='UFC Fight Prediction Tool')
    parser.add_argument('--event', type=str, help='Event ID for scraping fresh data')
    parser.add_argument('--no-scrape', action='store_true', help='Disable web scraping')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on past event')
    parser.add_argument('--past-event', type=str, help='Name of past event to evaluate (e.g., "UFC 300")')
    parser.add_argument('--past-date', type=str, help='Date of past event in YYYY-MM-DD format')
    args = parser.parse_args()
    
    event_id = args.event if args.event else None
    use_scraping = not args.no_scrape
    
    if args.evaluate:
        if not args.past_event or not args.past_date:
            print("Error: --past-event and --past-date are required when using --evaluate")
            sys.exit(1)
        main(event_id=event_id, use_scraping=use_scraping, evaluate_past=True, 
             past_event=args.past_event, past_date=args.past_date)
    else:
        main(event_id=event_id, use_scraping=use_scraping)
