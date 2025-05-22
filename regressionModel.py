import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 불러오기 또는 생성 (예시)
# df = pd.read_csv('your_movie_data.csv')
# 예제 데이터 생성
df = pd.read_csv('./movies_metadata.csv', low_memory=False)

# 전처리: 로그 스케일
df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
df["log_revenue"] = np.log1p(df["revenue"])

# 피처 및 타겟 설정
X = df[["budget", "vote_count"]]
y = df["log_revenue"]

# 파이프라인 구성
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 학습 및 예측
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 성능 평가
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R^2 Score: {r2:.4f}")
