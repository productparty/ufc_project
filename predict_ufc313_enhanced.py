import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse

# Load data
print("Loading fight data...")
fighters_df = pd.read_csv("ufc_stats/data/ufc_fighter_details.csv")
fighter_stats_df = pd.read_csv("ufc_stats/data/ufc_fighter_tott.csv")
fights_df = pd.read_csv("ufc_stats/data/ufc_fight_results.csv")
fight_stats_df = pd.read_csv("ufc_stats/data/ufc_fight_stats.csv")

print(f"Loaded {len(fighters_df)} fighters")
print(f"Loaded {len(fighter_stats_df)} fighter stats")
print(f"Loaded {len(fights_df)} fights")
print(f"Loaded {len(fight_stats_df)} fight stats")

# Helper functions
def clean_fighter_name(name):
    """Clean fighter name for better matching"""
    if pd.isna(name):
        return ""
    # Remove extra spaces, convert to lowercase
    name = re.sub(r'\s+', ' ', name).strip().lower()
    return name

def find_fighter_match(search_name, df, column='FIGHTER', debug=False):
    """Find the best match for a fighter name in the dataframe"""
    search_name = clean_fighter_name(search_name)
    
    # Try exact match first
    exact_match = df[df[column].apply(clean_fighter_name) == search_name]
    if len(exact_match) > 0:
        if debug:
            print(f"Exact match found for {search_name}")
        return exact_match.iloc[0]
    
    # Try contains match
    contains_match = df[df[column].apply(clean_fighter_name).str.contains(search_name)]
    if len(contains_match) > 0:
        if debug:
            print(f"Contains match found for {search_name}: {contains_match.iloc[0][column]}")
        return contains_match.iloc[0]
    
    # Try matching parts of the name
    name_parts = search_name.split()
    if len(name_parts) > 1:
        for part in name_parts:
            if len(part) > 2:  # Avoid matching short parts like "de" or "da"
                part_match = df[df[column].apply(clean_fighter_name).str.contains(part)]
                if len(part_match) > 0:
                    if debug:
                        print(f"Partial match found for {search_name} using '{part}': {part_match.iloc[0][column]}")
                    return part_match.iloc[0]
    
    if debug:
        print(f"No match found for {search_name}")
    return None

def extract_numeric(val, default=0):
    """Extract numeric value from string"""
    if pd.isna(val) or val == '--':
        return default
    try:
        # Remove any non-numeric characters except decimal point
        clean_val = re.sub(r'[^0-9.]', '', str(val))
        return float(clean_val) if clean_val else default
    except:
        return default

def convert_height_to_inches(height_str):
    """Convert height string to inches"""
    if pd.isna(height_str) or height_str == '--':
        return 0
    try:
        feet, inches = height_str.split("' ")
        inches = inches.replace('"', '')
        return int(feet) * 12 + int(inches)
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

def calculate_win_streak(fighter_name, fights_df, max_fights=10):
    """Calculate current win streak for a fighter"""
    # Get fighter's recent fights
    fighter_fights = fights_df[
        fights_df['BOUT'].apply(lambda x: 
            clean_fighter_name(fighter_name) in clean_fighter_name(x))
    ].sort_values('EVENT', ascending=False).head(max_fights)
    
    if len(fighter_fights) == 0:
        return 0
    
    # Calculate win streak
    streak = 0
    for _, fight in fighter_fights.iterrows():
        bout = fight['BOUT']
        outcome = fight['OUTCOME']
        
        fighters = bout.split('vs')
        if len(fighters) != 2:
            continue
            
        fighter1 = fighters[0].strip()
        fighter2 = fighters[1].strip()
        
        is_fighter1 = clean_fighter_name(fighter_name) in clean_fighter_name(fighter1)
        
        if (is_fighter1 and outcome == 'W/L') or (not is_fighter1 and outcome == 'L/W'):
            streak += 1
        else:
            break
    
    return streak

def calculate_losing_streak(fighter_name, fights_df, max_fights=10):
    """Calculate current losing streak for a fighter"""
    # Get fighter's recent fights
    fighter_fights = fights_df[
        fights_df['BOUT'].apply(lambda x: 
            clean_fighter_name(fighter_name) in clean_fighter_name(x))
    ].sort_values('EVENT', ascending=False).head(max_fights)
    
    if len(fighter_fights) == 0:
        return 0
    
    # Calculate losing streak
    streak = 0
    for _, fight in fighter_fights.iterrows():
        bout = fight['BOUT']
        outcome = fight['OUTCOME']
        
        fighters = bout.split('vs')
        if len(fighters) != 2:
            continue
            
        fighter1 = fighters[0].strip()
        fighter2 = fighters[1].strip()
        
        is_fighter1 = clean_fighter_name(fighter_name) in clean_fighter_name(fighter1)
        
        if (is_fighter1 and outcome == 'L/W') or (not is_fighter1 and outcome == 'W/L'):
            streak += 1
        else:
            break
    
    return streak

def calculate_ko_rate(fighter_name, fights_df, max_fights=10):
    """Calculate KO/TKO win rate for a fighter"""
    # Get fighter's recent fights
    fighter_fights = fights_df[
        fights_df['BOUT'].apply(lambda x: 
            clean_fighter_name(fighter_name) in clean_fighter_name(x))
    ].sort_values('EVENT', ascending=False).head(max_fights)
    
    if len(fighter_fights) == 0:
        return 0
    
    # Calculate KO rate
    ko_wins = 0
    total_wins = 0
    
    for _, fight in fighter_fights.iterrows():
        bout = fight['BOUT']
        outcome = fight['OUTCOME']
        method = fight['METHOD']
        
        fighters = bout.split('vs')
        if len(fighters) != 2:
            continue
            
        fighter1 = fighters[0].strip()
        fighter2 = fighters[1].strip()
        
        is_fighter1 = clean_fighter_name(fighter_name) in clean_fighter_name(fighter1)
        
        # Check if fighter won
        if (is_fighter1 and outcome == 'W/L') or (not is_fighter1 and outcome == 'L/W'):
            total_wins += 1
            # Check if win was by KO/TKO
            if 'KO/TKO' in method:
                ko_wins += 1
    
    return ko_wins / total_wins if total_wins > 0 else 0

def calculate_sub_rate(fighter_name, fights_df, max_fights=10):
    """Calculate submission win rate for a fighter"""
    # Get fighter's recent fights
    fighter_fights = fights_df[
        fights_df['BOUT'].apply(lambda x: 
            clean_fighter_name(fighter_name) in clean_fighter_name(x))
    ].sort_values('EVENT', ascending=False).head(max_fights)
    
    if len(fighter_fights) == 0:
        return 0
    
    # Calculate submission rate
    sub_wins = 0
    total_wins = 0
    
    for _, fight in fighter_fights.iterrows():
        bout = fight['BOUT']
        outcome = fight['OUTCOME']
        method = fight['METHOD']
        
        fighters = bout.split('vs')
        if len(fighters) != 2:
            continue
            
        fighter1 = fighters[0].strip()
        fighter2 = fighters[1].strip()
        
        is_fighter1 = clean_fighter_name(fighter_name) in clean_fighter_name(fighter1)
        
        # Check if fighter won
        if (is_fighter1 and outcome == 'W/L') or (not is_fighter1 and outcome == 'L/W'):
            total_wins += 1
            # Check if win was by submission
            if 'Submission' in method:
                sub_wins += 1
    
    return sub_wins / total_wins if total_wins > 0 else 0

def calculate_decision_rate(fighter_name, fights_df, max_fights=10):
    """Calculate decision win rate for a fighter"""
    # Get fighter's recent fights
    fighter_fights = fights_df[
        fights_df['BOUT'].apply(lambda x: 
            clean_fighter_name(fighter_name) in clean_fighter_name(x))
    ].sort_values('EVENT', ascending=False).head(max_fights)
    
    if len(fighter_fights) == 0:
        return 0
    
    # Calculate decision rate
    dec_wins = 0
    total_wins = 0
    
    for _, fight in fighter_fights.iterrows():
        bout = fight['BOUT']
        outcome = fight['OUTCOME']
        method = fight['METHOD']
        
        fighters = bout.split('vs')
        if len(fighters) != 2:
            continue
            
        fighter1 = fighters[0].strip()
        fighter2 = fighters[1].strip()
        
        is_fighter1 = clean_fighter_name(fighter_name) in clean_fighter_name(fighter1)
        
        # Check if fighter won
        if (is_fighter1 and outcome == 'W/L') or (not is_fighter1 and outcome == 'L/W'):
            total_wins += 1
            # Check if win was by decision
            if 'Decision' in method:
                dec_wins += 1
    
    return dec_wins / total_wins if total_wins > 0 else 0

def calculate_fighter_age(fighter_stats):
    """Calculate fighter age from DOB"""
    if pd.isna(fighter_stats.get('DOB')) or fighter_stats.get('DOB') == '--':
        return 30  # Default age if not available
    
    try:
        dob_str = fighter_stats['DOB']
        dob = datetime.strptime(dob_str, "%b %d, %Y")
        age = (datetime.now() - dob).days / 365.25
        return age
    except:
        return 30  # Default age if parsing fails

def calculate_total_fights(fighter_name, fights_df):
    """Calculate total number of fights for a fighter"""
    fighter_fights = fights_df[
        fights_df['BOUT'].apply(lambda x: 
            clean_fighter_name(fighter_name) in clean_fighter_name(x))
    ]
    return len(fighter_fights)

def extract_fight_features(fighter1, fighter2, fighter_stats_df, fights_df):
    """Extract comprehensive features for a fight"""
    # Find fighter stats
    fighter1_stats = find_fighter_match(fighter1, fighter_stats_df)
    fighter2_stats = find_fighter_match(fighter2, fighter_stats_df)
    
    if fighter1_stats is None or fighter2_stats is None:
        print(f"Could not find stats for {fighter1} or {fighter2}")
        return None
    
    # Physical attributes
    height1 = convert_height_to_inches(fighter1_stats.get('HEIGHT', 0))
    height2 = convert_height_to_inches(fighter2_stats.get('HEIGHT', 0))
    height_diff = height1 - height2
    
    reach1 = convert_reach_to_inches(fighter1_stats.get('REACH', 0))
    reach2 = convert_reach_to_inches(fighter2_stats.get('REACH', 0))
    reach_diff = reach1 - reach2
    
    # Age
    age1 = calculate_fighter_age(fighter1_stats)
    age2 = calculate_fighter_age(fighter2_stats)
    age_diff = age1 - age2
    
    # Experience
    total_fights1 = calculate_total_fights(fighter1, fights_df)
    total_fights2 = calculate_total_fights(fighter2, fights_df)
    experience_diff = total_fights1 - total_fights2
    
    # Win/loss streaks
    win_streak1 = calculate_win_streak(fighter1, fights_df)
    win_streak2 = calculate_win_streak(fighter2, fights_df)
    win_streak_diff = win_streak1 - win_streak2
    
    losing_streak1 = calculate_losing_streak(fighter1, fights_df)
    losing_streak2 = calculate_losing_streak(fighter2, fights_df)
    losing_streak_diff = losing_streak1 - losing_streak2
    
    # Fighting style metrics
    ko_rate1 = calculate_ko_rate(fighter1, fights_df)
    ko_rate2 = calculate_ko_rate(fighter2, fights_df)
    ko_rate_diff = ko_rate1 - ko_rate2
    
    sub_rate1 = calculate_sub_rate(fighter1, fights_df)
    sub_rate2 = calculate_sub_rate(fighter2, fights_df)
    sub_rate_diff = sub_rate1 - sub_rate2
    
    dec_rate1 = calculate_decision_rate(fighter1, fights_df)
    dec_rate2 = calculate_decision_rate(fighter2, fights_df)
    dec_rate_diff = dec_rate1 - dec_rate2
    
    # Striking stats
    sig_str_acc1 = extract_numeric(fighter1_stats.get('SIG. STR. ACC.', 0))
    sig_str_acc2 = extract_numeric(fighter2_stats.get('SIG. STR. ACC.', 0))
    sig_str_diff = sig_str_acc1 - sig_str_acc2
    
    # Takedown stats
    td_acc1 = extract_numeric(fighter1_stats.get('TD ACC.', 0))
    td_acc2 = extract_numeric(fighter2_stats.get('TD ACC.', 0))
    td_diff = td_acc1 - td_acc2
    
    # Submission stats
    sub_avg1 = extract_numeric(fighter1_stats.get('SUB. ATT', 0))
    sub_avg2 = extract_numeric(fighter2_stats.get('SUB. ATT', 0))
    sub_diff = sub_avg1 - sub_avg2
    
    # Create feature vector
    features = [
        height_diff, 
        reach_diff, 
        age_diff,
        experience_diff,
        win_streak_diff,
        losing_streak_diff,
        ko_rate_diff,
        sub_rate_diff,
        dec_rate_diff,
        sig_str_diff, 
        td_diff, 
        sub_diff,
        # Raw values (not just differences)
        height1, height2,
        reach1, reach2,
        win_streak1, win_streak2,
        losing_streak1, losing_streak2,
        ko_rate1, ko_rate2,
        sub_rate1, sub_rate2,
        dec_rate1, dec_rate2
    ]
    
    return features, {
        'height': f"{int(height1/12)}'{height1%12}\"" if height1 > 0 else "N/A",
        'reach': f"{reach1}\"" if reach1 > 0 else "N/A",
        'age': f"{age1:.1f}" if age1 != 30 else "N/A",
        'total_fights': total_fights1,
        'win_streak': win_streak1,
        'losing_streak': losing_streak1,
        'ko_rate': f"{ko_rate1:.0%}",
        'sub_rate': f"{sub_rate1:.0%}",
        'dec_rate': f"{dec_rate1:.0%}"
    }, {
        'height': f"{int(height2/12)}'{height2%12}\"" if height2 > 0 else "N/A",
        'reach': f"{reach2}\"" if reach2 > 0 else "N/A",
        'age': f"{age2:.1f}" if age2 != 30 else "N/A",
        'total_fights': total_fights2,
        'win_streak': win_streak2,
        'losing_streak': losing_streak2,
        'ko_rate': f"{ko_rate2:.0%}",
        'sub_rate': f"{sub_rate2:.0%}",
        'dec_rate': f"{dec_rate2:.0%}"
    }

def advanced_fight_analysis(fighter1, fighter2, fighter_stats_df, fight_stats_df, fights_df):
    """Analyze fight with advanced metrics"""
    # Extract features
    features_result = extract_fight_features(fighter1, fighter2, fighter_stats_df, fights_df)
    if features_result is None:
        return None
    
    features, fighter1_stats, fighter2_stats = features_result
    
    # Analyze fighting styles
    ko_rate1 = float(fighter1_stats['ko_rate'].strip('%')) / 100
    ko_rate2 = float(fighter2_stats['ko_rate'].strip('%')) / 100
    sub_rate1 = float(fighter1_stats['sub_rate'].strip('%')) / 100
    sub_rate2 = float(fighter2_stats['sub_rate'].strip('%')) / 100
    dec_rate1 = float(fighter1_stats['dec_rate'].strip('%')) / 100
    dec_rate2 = float(fighter2_stats['dec_rate'].strip('%')) / 100
    
    # Calculate method of victory probabilities
    # These are simplified calculations - a more sophisticated model would be better
    ko_probability = (ko_rate1 + ko_rate2) / 2
    sub_probability = (sub_rate1 + sub_rate2) / 2
    dec_probability = (dec_rate1 + dec_rate2) / 2
    
    # Normalize probabilities
    total = ko_probability + sub_probability + dec_probability
    if total > 0:
        ko_probability /= total
        sub_probability /= total
        dec_probability /= total
    else:
        # Default probabilities if no data
        ko_probability = 0.3
        sub_probability = 0.2
        dec_probability = 0.5
    
    return {
        'fighter1_stats': fighter1_stats,
        'fighter2_stats': fighter2_stats,
        'ko_probability': ko_probability,
        'sub_probability': sub_probability,
        'dec_probability': dec_probability
    }

def train_prediction_model(features, labels):
    """Train a prediction model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)
        print(f"{name} accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    print(f"Best model: {best_model.__class__.__name__} with accuracy: {best_accuracy:.4f}")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_names = [
            'height_diff', 'reach_diff', 'age_diff', 'experience_diff',
            'win_streak_diff', 'losing_streak_diff', 'ko_rate_diff',
            'sub_rate_diff', 'dec_rate_diff', 'sig_str_diff', 'td_diff', 'sub_diff',
            'height1', 'height2', 'reach1', 'reach2', 'win_streak1', 'win_streak2',
            'losing_streak1', 'losing_streak2', 'ko_rate1', 'ko_rate2',
            'sub_rate1', 'sub_rate2', 'dec_rate1', 'dec_rate2'
        ]
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 10 feature importance:")
        for i in range(min(10, len(feature_names))):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return best_model, scaler

def predict_fight(model, scaler, fighter1, fighter2, fighter_stats_df, fight_stats_df, fights_df):
    """Predict fight outcome"""
    # Extract features
    features_result = extract_fight_features(fighter1, fighter2, fighter_stats_df, fights_df)
    if features_result is None:
        return 0.5, None
    
    features, _, _ = features_result
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prob = model.predict_proba(features_scaled)[0][1]
    
    # Get advanced analysis
    analysis = advanced_fight_analysis(fighter1, fighter2, fighter_stats_df, fight_stats_df, fights_df)
    
    return prob, analysis

def export_predictions_to_csv(predictions, filename="ufc_313_enhanced_predictions.csv"):
    """Export predictions to CSV file"""
    df = pd.DataFrame(predictions)
    df.to_csv(filename, index=False)
    print(f"Predictions exported to {filename}")

def main():
    # Define upcoming fights
    upcoming_fights = [
        # UFC 313 - March 8, 2025
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
    
    # Create training data
    print("\nCreating training data...")
    X = []
    y = []
    
    # Process fights for training data
    for _, fight in fights_df.head(3000).iterrows():
        try:
            # Extract fighter names
            bout = fight['BOUT']
            if 'vs.' in bout:
                separator = 'vs.'
            elif 'vs' in bout:
                separator = 'vs'
            else:
                continue
                
            fighters = bout.split(separator)
            if len(fighters) != 2:
                continue
                
            fighter1 = fighters[0].strip()
            fighter2 = fighters[1].strip()
            
            # Determine winner
            outcome = fight['OUTCOME']
            if outcome == 'W/L':
                winner = fighter1
            elif outcome == 'L/W':
                winner = fighter2
            else:
                continue
            
            # Extract features
            features_result = extract_fight_features(fighter1, fighter2, fighter_stats_df, fights_df)
            if features_result is None:
                continue
            
            features, _, _ = features_result
            
            # Add to training data
            X.append(features)
            y.append(1 if winner == fighter1 else 0)
        except Exception as e:
            continue
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} training examples")
    
    # Train model
    print("\nTraining model...")
    model, scaler = train_prediction_model(X, y)
    
    # Make predictions
    print("\nMaking predictions for UFC 313...")
    predictions = []
    
    for fighter1, fighter2 in upcoming_fights:
        print(f"\n{fighter1} vs {fighter2}:")
        
        try:
            # Make prediction
            prob, analysis = predict_fight(model, scaler, fighter1, fighter2, fighter_stats_df, fight_stats_df, fights_df)
            
            # Determine predicted winner
            predicted_winner = fighter1 if prob > 0.5 else fighter2
            win_prob = prob if prob > 0.5 else 1 - prob
            
            # Determine confidence level
            if win_prob >= 0.8:
                confidence = "High"
            elif win_prob >= 0.65:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Store prediction details
            prediction = {
                'fighter1': fighter1,
                'fighter2': fighter2,
                'predicted_winner': predicted_winner,
                'win_probability': win_prob,
                'confidence': confidence
            }
            
            if analysis:
                prediction['ko_probability'] = analysis['ko_probability']
                prediction['sub_probability'] = analysis['sub_probability']
                prediction['dec_probability'] = analysis['dec_probability']
                
                # Print prediction details
                print(f"Predicted winner: {predicted_winner} ({win_prob:.1%} probability)")
                print(f"Confidence: {confidence}")
                print(f"Method of victory probabilities:")
                print(f"  KO/TKO: {analysis['ko_probability']:.1%}")
                print(f"  Submission: {analysis['sub_probability']:.1%}")
                print(f"  Decision: {analysis['dec_probability']:.1%}")
                
                # Print fighter stats
                print(f"\n{fighter1} stats:")
                for stat, value in analysis['fighter1_stats'].items():
                    print(f"  {stat}: {value}")
                
                print(f"\n{fighter2} stats:")
                for stat, value in analysis['fighter2_stats'].items():
                    print(f"  {stat}: {value}")
                
                # Flag potential upsets
                if (win_prob >= 0.5 and win_prob < 0.6) or (win_prob < 0.5 and win_prob > 0.4):
                    print("\n⚠️ POTENTIAL UPSET ALERT: Close matchup with upset potential")
            
            predictions.append(prediction)
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            continue
    
    # Export predictions to CSV
    export_predictions_to_csv(predictions)

if __name__ == "__main__":
    main() 