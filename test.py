# 성공 여부 레이블 생성
df['success'] = (df['revenue'] / df['budget']) >= 2
df['success_label'] = df['success'].map({True: 'Success', False: 'Fail'})

# 결측치 처리
df['runtime'] = df['runtime'].fillna(df['runtime'].mean())
df['release_date'] = df['release_date'].fillna(method='ffill')
df = df.dropna(subset=['original_language'])

# Replace missing values with the mean for numeric columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

df_cleaned = df[(df['budget'] > 0) & (df['revenue'] > 0)].dropna(subset=['budget', 'revenue'])


# 분류 모델 예시
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 스케일링 적용
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['budget', 'revenue']] = scaler.fit_transform(df[['budget', 'revenue']])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_cleaned[['budget_scaled', 'revenue_scaled']] = scaler.fit_transform(df_cleaned[['budget', 'revenue']])

print("\n[6] Scaled Budget and Revenue")
print(df_cleaned[['budget_scaled', 'revenue_scaled']].head())

# ...existing code...

correlation = df_cleaned[numeric_columns].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt

# # 데이터 정의
# labels = ['Success', 'Fail']
# sizes = [2755, 2626]  # 예: 성공 70%, 실패 30%
# colors = ['#4CAF50', '#FF5252']  # 성공은 초록색, 실패는 빨간색
# # explode = (0.1, 0)  # Success를 약간 분리해서 강조

# # 파이 차트 생성
# plt.figure(figsize=(6, 6))
# plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
# plt.title('Success / Fail')
# plt.show()