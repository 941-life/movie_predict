import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 예시 데이터 생성 (실제 사용시 영화 데이터셋으로 대체)
np.random.seed(42)
sample_size = 500
df = pd.DataFrame({
    "budget": np.random.randint(1000000, 200000000, sample_size),
    "popularity": np.random.rand(sample_size) * 100,
    "runtime": np.random.randint(80, 180, sample_size),
    "vote_average": np.random.rand(sample_size) * 10,
    "vote_count": np.random.randint(10, 10000, sample_size),
    "revenue": np.random.randint(1000000, 1000000000, sample_size),
    "original_language": np.random.choice(["en", "fr", "es", "ko"], sample_size),
    "genres": np.random.choice(["Action|Adventure", "Comedy|Romance", "Drama", "Horror|Thriller"], sample_size)
})

# 1. 전처리 - 타겟 변수 생성
df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
df["is_hit"] = (df["revenue"] / df["budget"] >= 2).astype(int)
df["log_revenue"] = np.log1p(df["revenue"])
df["genres"] = df["genres"].apply(lambda x: x.split("|"))

# 2. 장르 MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(df["genres"]), columns=mlb.classes_)
df = pd.concat([df.drop(columns=["genres"]), genres_encoded], axis=1)

# 3. Features & Targets
numerical_features = ["budget", "popularity", "runtime", "vote_average", "vote_count"]
categorical_features = ["original_language"]
genre_features = list(mlb.classes_)
X = df[numerical_features + categorical_features + genre_features]
y_class = df["is_hit"]
y_reg = df["log_revenue"]

# 4. Column Transformer
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")

# 5. Classification Pipeline
clf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 6. Regression Pipeline
reg_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# 7. Train/Test Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# 8. Fit & Predict - Classification 
clf_pipeline.fit(X_train_c, y_train_c)
y_pred_c = clf_pipeline.predict(X_test_c)
print("\n[Classification Report]\n")
print(classification_report(y_test_c, y_pred_c))

# 9. Fit & Predict - Regression
reg_pipeline.fit(X_train_r, y_train_r)
y_pred_r = reg_pipeline.predict(X_test_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)

print("\n[Regression Evaluation]\n")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")
