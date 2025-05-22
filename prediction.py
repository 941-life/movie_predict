import pandas as pd
import numpy as np
import ast # For parsing stringified lists/dicts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

df = pd.read_csv('./movies_metadata.csv', low_memory=False)

df['runtime'] = df['runtime'].fillna(df['runtime'].median())

df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df = df[df['budget'] > 0].copy()

# original_language 결측치 처리 - 최빈값으로 대체
most_common_lang = df['original_language'].mode()[0]
df['original_language'] = df['original_language'].fillna(most_common_lang)

# release_date 파싱 및 결측치 제거
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df = df.dropna(subset=['release_date'])

# genres 처리
def parse_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres_list]
    except:
        return [] # 오류 발생 시 빈 리스트 반환
df['genres_list'] = df['genres'].apply(parse_genres)

# original_language: 이미 문자열이므로 바로 사용 가능

# --- 분류 모델 준비 ---
print("\n--- Classification Model ---")
classification_df = df.copy()

# 타겟 변수 생성
classification_df['success'] = (classification_df['revenue'] / classification_df['budget'] >= 2).astype(int)

# 사용할 피처 선택
numerical_features_cls = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
categorical_features_cls_genres = 'genres_list' # MultiLabelBinarizer는 단일 컬럼을 받음
categorical_features_cls_lang = ['original_language'] # OneHotEncoder는 리스트 형태로 받음

# 전처리기 정의
# 장르: MultiLabelBinarizer (DataFrame에 직접 적용)
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(classification_df[categorical_features_cls_genres]),
                              columns=mlb.classes_, index=classification_df.index)
classification_df = pd.concat([classification_df, genres_encoded], axis=1)

# 언어: OneHotEncoder (ColumnTransformer 내부에서 사용)
# ColumnTransformer를 사용하면 숫자형 스케일링과 범주형 인코딩을 한번에 관리하기 용이
# 여기서는 간소화를 위해 pandas get_dummies를 사용할 수도 있음
lang_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for dense array
lang_encoded_df = pd.DataFrame(lang_encoder.fit_transform(classification_df[categorical_features_cls_lang]),
                               columns=lang_encoder.get_feature_names_out(categorical_features_cls_lang),
                               index=classification_df.index)
classification_df = pd.concat([classification_df, lang_encoded_df], axis=1)


# 최종 피처 목록 (인코딩된 장르 및 언어 포함)
final_features_cls = numerical_features_cls + list(mlb.classes_) + list(lang_encoder.get_feature_names_out(categorical_features_cls_lang))
X_cls = classification_df[final_features_cls]
y_cls = classification_df['success']

# 데이터 분할
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls if y_cls.nunique() > 1 else None) # stratify 추가

# 숫자형 피처 스케일링
scaler_cls = StandardScaler()
X_train_cls[numerical_features_cls] = scaler_cls.fit_transform(X_train_cls[numerical_features_cls])
X_test_cls[numerical_features_cls] = scaler_cls.transform(X_test_cls[numerical_features_cls])


# 모델 학습 (Random Forest 예시)
cls_model = RandomForestClassifier(random_state=42)
# 만약 샘플 수가 너무 적으면 오류가 발생할 수 있습니다.
if len(X_train_cls) > 0 and len(X_test_cls) > 0:
    cls_model.fit(X_train_cls, y_train_cls)
    y_pred_cls = cls_model.predict(X_test_cls)
    y_pred_proba_cls = cls_model.predict_proba(X_test_cls)[:, 1]

    # 모델 평가
    print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
    print("Classification Report:\n", classification_report(y_test_cls, y_pred_cls, zero_division=0))
    if len(np.unique(y_test_cls)) > 1 : # ROC AUC는 두 개 이상의 클래스가 필요
         print("ROC AUC Score:", roc_auc_score(y_test_cls, y_pred_proba_cls))
else:
    print("Not enough data for classification model training/testing after preprocessing.")

def test_classification_model(model, X_test, y_test):
    print("\n--- Classification Test Results ---")
    
    # 예측값
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # 확률값
    
    # 평가 지표
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # 출력
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    return y_pred, y_proba


# --- 회귀 모델 준비 ---
print("\n--- Regression Model ---")
regression_df = df.copy()

# 사용할 피처 및 타겟 선택
features_reg = ['vote_count', 'budget']
target_reg = 'revenue'

X_reg = regression_df[features_reg]
y_reg = regression_df[target_reg]

# 데이터 분할
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# 피처 스케일링 (선택사항이지만, 선형 모델에는 유용)
scaler_reg = StandardScaler()
X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

# 모델 학습 (Random Forest Regressor 예시)
reg_model = RandomForestRegressor(random_state=42)
if len(X_train_reg) > 0 and len(X_test_reg) > 0:
    reg_model.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg_model.predict(X_test_reg)

    # 모델 평가
    print("MAE:", mean_absolute_error(y_test_reg, y_pred_reg))
    print("MSE:", mean_squared_error(y_test_reg, y_pred_reg))
    print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
    print("R-squared:", r2_score(y_test_reg, y_pred_reg))
else:
    print("Not enough data for regression model training/testing after preprocessing.")

def test_regression_model(model, X_test, y_test):
    print("\n--- Regression Test Results ---")

    # 예측값
    y_pred = model.predict(X_test)

    # 평가 지표
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # 출력
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