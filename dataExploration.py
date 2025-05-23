import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

df = pd.read_csv('./movies_metadata.csv', low_memory=False)

columns_to_use = [
    'budget', 'genres', 'original_language', 'popularity',
    'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count'
]
# columns_to_use = [
#     'budget', 'genres', 'original_language',
#     'release_date', 'runtime'
# ]
df = df[columns_to_use]

# Convert columns to numeric
numeric_columns = ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'vote_count']
# numeric_columns = ['budget', 'runtime']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create success label
df['success'] = (df['revenue'] / df['budget']) >= 2
df['success'] = df['success'].map({True: 'Success', False: 'Fail'})

# Clean dataset
df_cleaned = df[(df['budget'] > 0) & (df['revenue'] > 0)].dropna(subset=['budget', 'revenue'])

# Apply StandardScaler to budget and revenue
scaler = StandardScaler()
df_cleaned[['budget_scaled', 'revenue_scaled']] = scaler.fit_transform(df_cleaned[['budget', 'revenue']])

# Numerical summary
print("\n[1] Numerical Feature Description")
print(df_cleaned[numeric_columns].describe())

# Missing values
print("\n[2] Missing Values per Column")
print(df[columns_to_use].isnull().sum())

# Categorical summary
print("\n[3] Top 5 Original Languages")
print(df['original_language'].value_counts().head(5))

print("\n[4] Sample Genres Data")
print(df['genres'].dropna().head(5))

# Success distribution
print("\n[5] Success vs Fail Counts")
print(df_cleaned['success'].value_counts())

# Print scaled budget and revenue
print("\n[6] Scaled Budget and Revenue")
print(df_cleaned[['budget', 'revenue']].head())
print(df_cleaned[['budget_scaled', 'revenue_scaled']].head())

# Correlation heatmap
correlation = df_cleaned[numeric_columns].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig('./correlation_heatmap.png')
plt.show()


# genres preprocessing
df['genres'] = df['genres'].fillna('[]')  # chang missing values -> empty list
df['genres'] = df['genres'].apply(lambda x: eval(x) if isinstance(x, str) else [])
df['main_genre'] = df['genres'].apply(lambda x: x[0]['name'] if len(x) > 0 else 'None')  # main genres
df_encoded_genres = pd.get_dummies(df['main_genre'], prefix='genre')  # One-Hot Encoding

# original_language processing & encoding
df['original_language'] = df['original_language'].fillna('unknown')  # change missing values -> unknown
df_encoded_languages = pd.get_dummies(df['original_language'], prefix='lang')  # One-Hot Encoding

