import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# Load the real traffic accident dataset
df = pd.read_csv("data/RTA Dataset.csv")
# Create Hour feature from Time
df["Hour"] = pd.to_datetime(
    df["Time"],
    format="%H:%M:%S"
).dt.hour

print("\nTime and Hour:")
print(df[["Time", "Hour"]].head())

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# ==========================================
# MISSING VALUE PERCENTAGE
# ==========================================

missing = df.isnull().sum()

missing_percentage = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Missing Percentage": missing_percentage
})

print("\nMissing Value Analysis:")
print(missing_table[missing_table["Missing Values"] > 0])


# ==========================================
# DUPLICATE VALUE CHECK
# ==========================================

duplicates = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicates)

# ==========================================
# ACCIDENT SEVERITY DISTRIBUTION
# ==========================================

print("\nAccident Severity Distribution:")
print(df["Accident_severity"].value_counts())

print("\nAccident Severity Percentage:")
print(df["Accident_severity"].value_counts(normalize=True) * 100)

# ==========================================
# ACCIDENT SEVERITY GRAPH
# ==========================================

sns.countplot(
    x="Accident_severity",
    data=df
)

plt.title("Accident Severity Distribution")
plt.xlabel("Accident Severity")
plt.ylabel("Number of Accidents")

plt.xticks(rotation=15)

plt.show()

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

# Categorical columns
categorical_columns = df.select_dtypes(include="object").columns

# Fill missing categorical values with Unknown
df[categorical_columns] = df[categorical_columns].fillna("Unknown")

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())


# ==========================================
# CHECK NUMERIC MISSING VALUES
# ==========================================

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

print("\nNumeric columns:")
print(numeric_columns.tolist())

print("\nMissing values in numeric columns:")
print(df[numeric_columns].isnull().sum())

df = df.drop_duplicates()

print("Rows after removing duplicates:", len(df))

print("\nFinal dataset shape:")
print(df.shape)

print("\nTotal missing values:")
print(df.isnull().sum().sum())


# ==========================================
# WEATHER VS ACCIDENT SEVERITY
# ==========================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Weather_conditions",
    hue="Accident_severity"
)

plt.title("Weather Conditions vs Accident Severity")
plt.xlabel("Weather Conditions")
plt.ylabel("Number of Accidents")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# ROAD SURFACE VS ACCIDENT SEVERITY
# ==========================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Road_surface_conditions",
    hue="Accident_severity"
)

plt.title("Road Surface Conditions vs Accident Severity")
plt.xlabel("Road Surface Conditions")
plt.ylabel("Number of Accidents")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# LIGHT CONDITIONS VS ACCIDENT SEVERITY
# ==========================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Light_conditions",
    hue="Accident_severity"
)

plt.title("Light Conditions vs Accident Severity")
plt.xlabel("Light Conditions")
plt.ylabel("Number of Accidents")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ==========================================
# TOP ACCIDENT CAUSES
# ==========================================

top_causes = df["Cause_of_accident"].value_counts().head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=top_causes.values,
    y=top_causes.index
)

plt.title("Top 10 Causes of Accidents")
plt.xlabel("Number of Accidents")
plt.ylabel("Cause of Accident")

plt.tight_layout()
plt.show()

# ==========================================
# VEHICLES INVOLVED VS ACCIDENT SEVERITY
# ==========================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="Number_of_vehicles_involved",
    hue="Accident_severity"
)

plt.title("Number of Vehicles vs Accident Severity")
plt.xlabel("Number of Vehicles Involved")
plt.ylabel("Number of Accidents")

plt.tight_layout()
plt.show()

# ==========================================
# WEATHER VS ACCIDENT SEVERITY %
# ==========================================

weather_severity = pd.crosstab(
    df["Weather_conditions"],
    df["Accident_severity"],
    normalize="index"
) * 100

print("\nWeather vs Accident Severity (%):")
print(weather_severity.round(2))

# ==========================================
# ROAD CONDITION VS ACCIDENT SEVERITY %
# ==========================================

road_severity = pd.crosstab(
    df["Road_surface_conditions"],
    df["Accident_severity"],
    normalize="index"
) * 100

print("\nRoad Surface Conditions vs Accident Severity (%):")
print(road_severity.round(2))

# ==========================================
# LIGHT CONDITIONS VS ACCIDENT SEVERITY %
# ==========================================

light_severity = pd.crosstab(
    df["Light_conditions"],
    df["Accident_severity"],
    normalize="index"
) * 100

print("\nLight Conditions vs Accident Severity (%):")
print(light_severity.round(2))

# ==========================================
# RISK SCORE FEATURE
# ==========================================

df["Risk_Score"] = (
    df["Number_of_vehicles_involved"]
    + df["Number_of_casualties"]
)

print("\nRisk Score:")
print(
    df[
        [
            "Number_of_vehicles_involved",
            "Number_of_casualties",
            "Risk_Score"
        ]
    ].head()
)


# ==========================================
# SELECT FEATURES
# ==========================================

features = [
    "Hour",
    "Day_of_week",
    "Age_band_of_driver",
    "Sex_of_driver",
    "Driving_experience",
    "Type_of_vehicle",
    "Area_accident_occured",
    "Road_surface_type",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_collision",
    "Number_of_vehicles_involved",
    "Number_of_casualties",
    "Risk_Score",
    "Vehicle_movement",
    "Cause_of_accident"
]

X = df[features]

# Target
y = df["Accident_severity"]

print("\nSelected Features:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

# ==========================================
# CATEGORICAL ENCODING
# ==========================================

X_encoded = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

print("\nEncoded Dataset:")
print(X_encoded.head())

print("\nEncoded Dataset Shape:")
print(X_encoded.shape)

# Target
y = df["Accident_severity"]

print("\nSelected Features:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# # ==========================================
# # LOGISTIC REGRESSION MODEL
# # ==========================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed!")

# ==========================================
# MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:20])

# ==========================================
# MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nModel Accuracy (%):")
print(accuracy * 100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("Accident Severity - Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================
# BALANCED LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("\nTraining Balanced Logistic Regression model...")

model.fit(X_train, y_train)

print("Balanced model training completed!")

# Predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:20])

accuracy = accuracy_score(y_test, y_pred)

print("\nBalanced Model Accuracy:")
print(accuracy * 100)

print("\nBalanced Model Classification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest model...")

rf_model.fit(X_train, y_train)

print("Random Forest training completed!")

# Predictions
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Predictions:")
print(rf_pred[:20])

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:")
print(rf_accuracy * 100)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# # ==========================================
# # GRADIENT BOOSTING MODEL
# # ==========================================

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

print("\nTraining Gradient Boosting model...")

gb_model.fit(X_train, y_train)

print("Gradient Boosting training completed!")

# Predictions
gb_pred = gb_model.predict(X_test)

print("\nGradient Boosting Predictions:")
print(gb_pred[:20])
gb_accuracy = accuracy_score(y_test, gb_pred)

print("\nGradient Boosting Accuracy:")
print(gb_accuracy * 100)

print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, gb_pred))

# ==========================================
# SAVE RANDOM FOREST MODEL
# ==========================================

import joblib

joblib.dump(rf_model, "accident_severity_model.pkl")

# Save the feature names used by the model
joblib.dump(X_encoded.columns.tolist(), "model_features.pkl")

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY!")
print("accident_severity_model.pkl")
print("model_features.pkl")
print("===================================")