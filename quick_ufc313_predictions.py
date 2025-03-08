import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import re
from datetime import datetime

# Define UFC 313 fights
ufc_313_fights = [
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

# Load data
print("Loading fight data...")
fighters_df = pd.read_csv("ufc_stats/data/ufc_fighter_details.csv")
fighter_stats_df = pd.read_csv("ufc_stats/data/ufc_fighter_tott.csv")
fights_df = pd.read_csv("ufc_stats/data/ufc_fight_results.csv")

print(f"Loaded {len(fighters_df)} fighters")
print(f"Loaded {len(fighter_stats_df)} fighter stats")
print(f"Loaded {len(fights_df)} fights")

# Helper functions
def clean_fighter_name(name):
    """Clean fighter name for better matching"""
    if pd.isna(name):
        return ""
    # Remove extra spaces, convert to lowercase
    name = re.sub(r'\s+', ' ', name).strip().lower()
    return name

def find_fighter_match(search_name, df, column='FIGHTER'):
    """Find the best match for a fighter name in the dataframe"""
    search_name = clean_fighter_name(search_name)
    
    # Try exact match first
    exact_match = df[df[column].apply(clean_fighter_name) == search_name]
    if len(exact_match) > 0:
        return exact_match.iloc[0]
    
    # Try contains match
    contains_match = df[df[column].apply(clean_fighter_name).str.contains(search_name)]
    if len(contains_match) > 0:
        return contains_match.iloc[0]
    
    # Try matching parts of the name
    name_parts = search_name.split()
    if len(name_parts) > 1:
        for part in name_parts:
            if len(part) > 2:  # Avoid matching short parts like "de" or "da"
                part_match = df[df[column].apply(clean_fighter_name).str.contains(part)]
                if len(part_match) > 0:
                    return part_match.iloc[0]
    
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

def calculate_win_streak(fighter_name, fights_df, max_fights=5):
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

def calculate_losing_streak(fighter_name, fights_df, max_fights=5):
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

def extract_fight_features(fighter1, fighter2, fighter_stats_df, fights_df):
    """Extract features for a fight"""
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
    
    # Win/loss streaks
    win_streak1 = calculate_win_streak(fighter1, fights_df)
    win_streak2 = calculate_win_streak(fighter2, fights_df)
    win_streak_diff = win_streak1 - win_streak2
    
    losing_streak1 = calculate_losing_streak(fighter1, fights_df)
    losing_streak2 = calculate_losing_streak(fighter2, fights_df)
    losing_streak_diff = losing_streak1 - losing_streak2
    
    # Create feature vector
    features = [
        height_diff, 
        reach_diff, 
        win_streak_diff,
        losing_streak_diff
    ]
    
    return features, {
        'height': f"{int(height1/12)}'{height1%12}\"" if height1 > 0 else "N/A",
        'reach': f"{reach1}\"" if reach1 > 0 else "N/A",
        'win_streak': win_streak1,
        'losing_streak': losing_streak1
    }, {
        'height': f"{int(height2/12)}'{height2%12}\"" if height2 > 0 else "N/A",
        'reach': f"{reach2}\"" if reach2 > 0 else "N/A",
        'win_streak': win_streak2,
        'losing_streak': losing_streak2
    }

# Create training data
print("\nCreating training data...")
X = []
y = []

# Process fights for training data
for _, fight in fights_df.head(1000).iterrows():
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
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Make predictions
print("\nMaking predictions for UFC 313...")
predictions = []

for fighter1, fighter2 in ufc_313_fights:
    print(f"\n{fighter1} vs {fighter2}:")
    
    try:
        # Extract features
        features_result = extract_fight_features(fighter1, fighter2, fighter_stats_df, fights_df)
        if features_result is None:
            print(f"Could not extract features for {fighter1} vs {fighter2}")
            continue
        
        features, fighter1_stats, fighter2_stats = features_result
        
        # Make prediction
        prob = model.predict_proba([features])[0][1]
        
        # Determine predicted winner
        predicted_winner = fighter1 if prob > 0.5 else fighter2
        win_prob = prob if prob > 0.5 else 1 - prob
        
        # Determine confidence level
        if win_prob >= 0.7:
            confidence = "High"
        elif win_prob >= 0.6:
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
        
        # Print prediction details
        print(f"Predicted winner: {predicted_winner} ({win_prob:.1%} probability)")
        print(f"Confidence: {confidence}")
        
        # Print fighter stats
        print(f"\n{fighter1} stats:")
        for stat, value in fighter1_stats.items():
            print(f"  {stat}: {value}")
        
        print(f"\n{fighter2} stats:")
        for stat, value in fighter2_stats.items():
            print(f"  {stat}: {value}")
        
        predictions.append(prediction)
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        continue

# Export predictions to CSV
df = pd.DataFrame(predictions)
df.to_csv("ufc_313_quick_predictions.csv", index=False)
print(f"\nPredictions exported to ufc_313_quick_predictions.csv") 