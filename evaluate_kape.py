import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Function to find the best match for a fighter name
def find_fighter_match(name, df):
    # Try exact match first
    exact_match = df[df['FIGHTER'].str.lower() == name.lower()]
    if len(exact_match) > 0:
        return exact_match.iloc[0]
    
    # Try contains match
    contains_match = df[df['FIGHTER'].str.lower().str.contains(name.lower())]
    if len(contains_match) > 0:
        return contains_match.iloc[0]
    
    # No match found
    return None

# Function to extract features for a fight
def extract_fight_features(fighter1, fighter2):
    # Find fighter stats
    fighter1_stats = find_fighter_match(fighter1, fighter_stats_df)
    fighter2_stats = find_fighter_match(fighter2, fighter_stats_df)
    
    if fighter1_stats is None or fighter2_stats is None:
        print(f"Could not find stats for {fighter1} or {fighter2}")
        return None
    
    # Extract numeric features
    def extract_numeric(val, default=0):
        if pd.isna(val) or val == '--':
            return default
        try:
            return float(val)
        except:
            return default
    
    # Calculate feature differences
    features = []
    
    # Height difference (convert to inches)
    def convert_height(height_str):
        if pd.isna(height_str) or height_str == '--':
            return 0
        try:
            feet, inches = height_str.split("' ")
            inches = inches.replace('"', '')
            return int(feet) * 12 + int(inches)
        except:
            return 0
    
    height1 = convert_height(fighter1_stats.get('HEIGHT', 0))
    height2 = convert_height(fighter2_stats.get('HEIGHT', 0))
    height_diff = height1 - height2
    
    # Reach difference (convert to inches)
    def convert_reach(reach_str):
        if pd.isna(reach_str) or reach_str == '--':
            return 0
        try:
            return float(reach_str.replace('"', ''))
        except:
            return 0
    
    reach1 = convert_reach(fighter1_stats.get('REACH', 0))
    reach2 = convert_reach(fighter2_stats.get('REACH', 0))
    reach_diff = reach1 - reach2
    
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
    features = [height_diff, reach_diff, sig_str_diff, td_diff, sub_diff]
    
    return features

# Create training data from historical fights
print("Creating training data...")
X = []
y = []

# Process first 1000 fights for training data
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
        features = extract_fight_features(fighter1, fighter2)
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

# Train a simple model
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate on test set
test_acc = model.score(X_test_scaled, y_test)
print(f"Test accuracy: {test_acc:.2f}")

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
    features = extract_fight_features(fighter1, fighter2)
    if features is None:
        print(f"Could not extract features for {fighter1} vs {fighter2}")
        continue
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prob = model.predict_proba(features_scaled)[0][1]
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
results_df.to_csv("kape_almabayev_evaluation.csv", index=False)
print("Results exported to kape_almabayev_evaluation.csv") 