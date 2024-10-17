# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load datasets
# We start by loading the three datasets containing demographic details, screen time, and well-being data.
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")
dataset3 = pd.read_csv("dataset3.csv")

# Merge datasets on 'ID' to combine demographic, screen time, and well-being data
# This step ensures we have a complete set of information for each respondent.
merged_data = pd.merge(dataset1, dataset2, on='ID', how='inner')
merged_data = pd.merge(merged_data, dataset3, on='ID', how='inner')

# Create total screen time features
# To understand general usage patterns, we calculate the total screen time on weekdays and weekends.
merged_data['Total_We'] = merged_data[['C_we', 'G_we', 'S_we', 'T_we']].sum(axis=1)  # Weekend usage
merged_data['Total_Wk'] = merged_data[['C_wk', 'G_wk', 'S_wk', 'T_wk']].sum(axis=1)  # Weekday usage

# Calculate the average well-being score
# We aggregate multiple well-being indicators into an average score to create a more comprehensive view.
wellbeing_cols = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 
                  'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
merged_data['Avg_Wellbeing'] = merged_data[wellbeing_cols].mean(axis=1)

# Create a composite score from the top well-being indicators most correlated with Avg_Wellbeing
# This composite score captures the most influential aspects of well-being, improving the model's predictive accuracy.
top_wellbeing_features = ['Goodme', 'Cheer', 'Conf', 'Thcklr', 'Usef', 'Dealpr', 'Relx', 
                          'Clsep', 'Loved', 'Intthg', 'Mkmind']
merged_data['Composite_Wellbeing'] = merged_data[top_wellbeing_features].mean(axis=1)

# Select features including the composite score
# We include key demographic data, screen time variables, and the composite well-being score for our model.
features_with_composite = ['Total_We', 'Total_Wk', 'gender', 'minority', 'deprived', 'Composite_Wellbeing']
X = merged_data[features_with_composite]
y = merged_data['Avg_Wellbeing']

# Split the data
# The data is split into training and testing sets to allow for model validation and prevent overfitting.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the linear regression model
# Linear regression is applied to understand the relationship between the selected features and well-being scores.
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and calculate R² score
# The R² score helps measure how well the model explains the variance in the data.
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Output the R² value
print(f"R² score of the linear regression model: {r2:.2f}")

# Visualization: Correlation Heatmap
# A heatmap is generated to visually inspect correlations between features, guiding feature selection.
plt.figure(figsize=(10, 6))
sns.heatmap(merged_data.corr(), annot=False, cmap='Blues')
plt.title("Correlation Heatmap of Features")
plt.show()

# Visualization: Distribution of Avg_Wellbeing
# The histogram displays the distribution of well-being scores to identify any skewness or patterns.
plt.figure(figsize=(8, 6))
sns.histplot(merged_data['Avg_Wellbeing'], bins=20, color='skyblue', kde=True)
plt.title("Distribution of Average Well-being")
plt.xlabel("Average Well-being Score")
plt.ylabel("Frequency")
plt.show()

# Visualization: Scatter plot of Composite Well-being vs. Avg_Wellbeing
# This scatter plot illustrates the relationship between the composite well-being score and the average well-being.
plt.figure(figsize=(8, 6))
sns.scatterplot(x=merged_data['Composite_Wellbeing'], y=merged_data['Avg_Wellbeing'], color='purple', alpha=0.6)
plt.title("Composite Well-being vs. Average Well-being")
plt.xlabel("Composite Well-being Score")
plt.ylabel("Average Well-being Score")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
