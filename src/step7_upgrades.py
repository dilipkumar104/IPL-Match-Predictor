import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# --- rebuild dataset (same as Step 4-6) ---
df = pd.read_csv(r'c:/Users/dilip/OneDrive/Desktop/IPL matches dataset/data/matches.csv')
df = df.drop(columns=['method', 'umpire1', 'umpire2'], errors='ignore')
df = df.dropna(subset=['winner'])
df = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'winner']]

le = LabelEncoder()
df['winner_encoded'] = le.fit_transform(df['winner'])

team_win_counts = df['winner'].value_counts()
df['team1_strength'] = df['team1'].map(team_win_counts)
df['team2_strength'] = df['team2'].map(team_win_counts)
df['toss_win'] = (df['toss_winner'] == df['team1']).astype(int)
df['toss_decision'] = df['toss_decision'].map({'bat': 1, 'field': 0})

X = df[['team1_strength', 'team2_strength', 'toss_win', 'toss_decision']]
y = df['winner_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================
# UPGRADE 1: Logistic Regression + classification report
# =============================================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

print("=== LOGISTIC REGRESSION ===")
print("Accuracy:", round(accuracy_score(y_test, preds), 4))
print()
# Only show macro/weighted averages - per-class is 19 rows, too noisy
cr_lr = classification_report(y_test, preds, zero_division=0, output_dict=True)
print(f"  precision (weighted) : {cr_lr['weighted avg']['precision']:.4f}")
print(f"  recall    (weighted) : {cr_lr['weighted avg']['recall']:.4f}")
print(f"  f1-score  (weighted) : {cr_lr['weighted avg']['f1-score']:.4f}")

# =============================================
# UPGRADE 2: Random Forest
# =============================================
print()
print("=== RANDOM FOREST ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("RF Accuracy:", round(accuracy_score(y_test, rf_preds), 4))
cr_rf = classification_report(y_test, rf_preds, zero_division=0, output_dict=True)
print(f"  precision (weighted) : {cr_rf['weighted avg']['precision']:.4f}")
print(f"  recall    (weighted) : {cr_rf['weighted avg']['recall']:.4f}")
print(f"  f1-score  (weighted) : {cr_rf['weighted avg']['f1-score']:.4f}")

# =============================================
# UPGRADE 3: Feature Importance
# =============================================
print()
print("=== FEATURE IMPORTANCE (Random Forest) ===")
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
for feat, score in fi.items():
    bar = '#' * int(score * 50)
    print(f"  {feat:<22} {score:.4f}  {bar}")

print()
print("Winner: LR" if accuracy_score(y_test, preds) > accuracy_score(y_test, rf_preds) else "Winner: RF")
