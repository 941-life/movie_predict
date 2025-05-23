import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

df = pd.read_csv('./movies_metadata.csv', low_memory=False)


df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df = df[df['budget'] > 0].copy()

# removing missing value
df['runtime'] = df['runtime'].fillna(df['runtime'].median())
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df = df.dropna(subset=['release_date'])

# original_language - 최빈값 대체
most_common_lang = df['original_language'].mode()[0]
df['original_language'] = df['original_language'].fillna(most_common_lang)

# genres parsing
def parse_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres_list]
    except:
        return []
df['genres_list'] = df['genres'].apply(parse_genres)


# MultiLabelBinarizer: genres_list
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(df['genres_list']),
                              columns=mlb.classes_, index=df.index)
df = pd.concat([df, genres_encoded], axis=1)

# OneHotEncoder: original_language
lang_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
lang_encoded = pd.DataFrame(lang_encoder.fit_transform(df[['original_language']]),
                            columns=lang_encoder.get_feature_names_out(['original_language']),
                            index=df.index)
df = pd.concat([df, lang_encoded], axis=1)

# Classification Model
print("\n--- Classification Model ---")
classification_df = df.copy()

# labeling (1: success / 0: failed)
classification_df['success'] = (classification_df['revenue'] / classification_df['budget'] >= 2).astype(int)

# feature selection
numerical_features_cls = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
categorical_features_cls_genres = 'genres_list'
categorical_features_cls_lang = ['original_language']

final_features_cls = numerical_features_cls + list(mlb.classes_) + list(lang_encoder.get_feature_names_out(categorical_features_cls_lang))

X_cls = classification_df[final_features_cls]
y_cls = classification_df['success']

# train/test split
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls if y_cls.nunique() > 1 else None) # stratify 추가

# numerical feature scaling
scaler_cls = StandardScaler()
X_train_cls[numerical_features_cls] = scaler_cls.fit_transform(X_train_cls[numerical_features_cls])
X_test_cls[numerical_features_cls] = scaler_cls.transform(X_test_cls[numerical_features_cls])


# train
cls_model = RandomForestClassifier(random_state=42)
if len(X_train_cls) > 0 and len(X_test_cls) > 0:
    cls_model.fit(X_train_cls, y_train_cls)
    y_pred_cls = cls_model.predict(X_test_cls)
    y_pred_proba_cls = cls_model.predict_proba(X_test_cls)[:, 1]

    # evaluation
    print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
    print("Classification Report:\n", classification_report(y_test_cls, y_pred_cls, zero_division=0))
    if len(np.unique(y_test_cls)) > 1 :
         print("ROC AUC Score:", roc_auc_score(y_test_cls, y_pred_proba_cls))
else:
    print("Not enough data for classification model training/testing after preprocessing.")



# Regression Model
print("\n--- Regression Model ---")
regression_df = df.copy()

# feature selection
features_reg = ['vote_count', 'budget', 'popularity', 'runtime', 'vote_average']
target_reg = 'revenue'

X_reg = regression_df[features_reg]
y_reg = regression_df[target_reg]

# train/test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# feature scaling
scaler_reg = StandardScaler()
X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

# train
reg_model = RandomForestRegressor(random_state=42)
if len(X_train_reg) > 0 and len(X_test_reg) > 0:
    reg_model.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg_model.predict(X_test_reg)

    # evaluation
    print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
    print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))
    print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
    print("R-squared:", r2_score(y_test_reg, y_pred_reg))
else:
    print("Not enough data for regression model training/testing after preprocessing.")


# ---- Testing ----

# classification model test
def test_classification_model(model, X_test, y_test):
    print("\n--- Classification Test Results ---")

    # prediction
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # probabilities for success class

    # evaluation
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    return y_pred, y_proba

# regression model test
def test_regression_model(model, X_test, y_test):
    print("\n--- Regression Test Results ---")

    # prediction
    y_pred = model.predict(X_test)

    # evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:,.2f}")
    print(f"MSE: {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R-squared: {r2:.4f}")

    return y_pred

y_pred_cls, y_proba_cls = test_classification_model(cls_model, X_test_cls, y_test_cls)
y_pred_reg = test_regression_model(reg_model, X_test_reg, y_test_reg)

print("\n--- Predictions ---")
print("Classification Predictions:\n", y_pred_cls)
print("Classification Probabilities:\n", y_proba_cls)
print("Regression Predictions:\n", y_pred_reg)