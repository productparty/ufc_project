import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import os
import re

# Load data
print("Loading data...")
fighters_df = pd.read_csv("ufc_stats/data/ufc_fighter_details.csv")
fighter_stats_df = pd.read_csv("ufc_stats/data/ufc_fighter_tott.csv")
fights_df = pd.read_csv("ufc_stats/data/ufc_fight_results.csv")
fight_stats_df = pd.read_csv("ufc_stats/data/ufc_fight_stats.csv")

print(f"Loaded {len(fighters_df)} fighters")
print(f"Loaded {len(fighter_stats_df)} fighter stats")
print(f"Loaded {len(fights_df)} fights")
print(f"Loaded {len(fight_stats_df)} fight stats")

# Define the fights for UFC Fight Night: Kape vs. Almabayev
kape_fights = [
    {"fighter1": "Manel Kape", "fighter2": "Asu Almabayev", "outcome": "W/L"},
    {"fighter1": "Cody Brundage", "fighter2": "Julian Marquez", "outcome": "W/L"},
    {"fighter1": "Nasrat Haqparast", "fighter2": "Esteban Ribovics", "outcome": "W/L"},
    {"fighter1": "Hyder Amil", "fighter2": "William Gomis", "outcome": "W/L"},
    {"fighter1": "Danny Barlow", "fighter2": "Sam Patterson", "outcome": "L/W"},
    {"fighter1": "Austen Lane", "fighter2": "Mario Pinto", "outcome": "L/W"},
    {"fighter1": "Ricardo Ramos", "fighter2": "Chepe Mariscal", "outcome": "L/W"},
    {"fighter1": "Danny Silva", "fighter2": "Lucas Almeida", "outcome": "W/L"},
    {"fighter1": "Andrea Lee", "fighter2": "JJ Aldrich", "outcome": "L/W"},
    {"fighter1": "Charles Johnson", "fighter2": "Ramazan Temirov", "outcome": "L/W"}
]

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
        from datetime import datetime
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
    
    return features

# Create training data from historical fights
print("Creating training data...")
X = []
y = []

# Process fights for training data (use more than 1000 for better results)
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
        features = extract_fight_features(fighter1, fighter2, fighter_stats_df, fights_df)
        if features is None:
            continue
        
        # Add to training data
        X.append(features)
        y.append(1 if winner == fighter1 else 0)
    except Exception as e:
        continue

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"Created {len(X)} training examples")

# Train a more sophisticated model
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models and compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} accuracy: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f"\nBest model: {best_model.__class__.__name__} with accuracy: {best_accuracy:.4f}")

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
    
    print("\nFeature importance:")
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Evaluate on Kape vs. Almabayev fights
print("\nEvaluating UFC Fight Night: Kape vs. Almabayev")
correct = 0
total = 0
results = []

for fight in kape_fights:
    fighter1 = fight["fighter1"]
    fighter2 = fight["fighter2"]
    outcome = fight["outcome"]
    
    print(f"\n{fighter1} vs {fighter2}")
    
    # Determine actual winner
    if outcome == "W/L":
        actual_winner = fighter1
    elif outcome == "L/W":
        actual_winner = fighter2
    else:
        print(f"Unknown outcome: {outcome}")
        continue
    
    print(f"Actual winner: {actual_winner}")
    
    # Extract features
    features = extract_fight_features(fighter1, fighter2, fighter_stats_df, fights_df)
    if features is None:
        print(f"Could not extract features for {fighter1} vs {fighter2}")
        continue
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prob = best_model.predict_proba(features_scaled)[0][1]
    predicted_winner = fighter1 if prob > 0.5 else fighter2
    win_prob = prob if prob > 0.5 else 1 - prob
    
    print(f"Predicted winner: {predicted_winner} ({win_prob:.2%})")
    
    # Check if prediction was correct
    is_correct = predicted_winner == actual_winner
    if is_correct:
        correct += 1
    total += 1
    
    print(f"Correct: {'✓' if is_correct else '✗'}")
    
    # Store result
    results.append({
        "fighter1": fighter1,
        "fighter2": fighter2,
        "actual_winner": actual_winner,
        "predicted_winner": predicted_winner,
        "win_probability": win_prob,
        "correct": is_correct
    })

# Calculate accuracy
accuracy = correct / total if total > 0 else 0
print(f"\nOverall accuracy: {accuracy:.2%} ({correct}/{total})")

# Export results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("kape_almabayev_enhanced_evaluation.csv", index=False)
print("Results exported to kape_almabayev_enhanced_evaluation.csv") 