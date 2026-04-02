import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")

print("--- Loading Dataset ---")
# Replace 'WA_Fn-UseC_-Telco-Customer-Churn.csv' with your actual file path
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("\nDataset Info:")
print(df.info())

print("\nDataset Description (Numerical):")
print(df.describe())

# ==========================================
# 2. DATA CLEANING
# ==========================================
print("\n--- Data Cleaning ---")

# The 'TotalCharges' column is stored as an object (string) because of blank spaces.
# We need to replace empty spaces with NaN and convert to float.
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan).astype(float)

# Check how many missing values we just exposed
missing_values = df["TotalCharges"].isnull().sum()
print(f"Missing values in TotalCharges after conversion: {missing_values}")

# Drop rows with missing TotalCharges (since it's usually only ~11 rows out of 7000+)
df.dropna(subset=["TotalCharges"], inplace=True)

# Drop the 'customerID' column as it has no predictive power
df.drop("customerID", axis=1, inplace=True)

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
print("\n--- Feature Engineering ---")

# 1. Create TenureGroup (Bucket into years)
labels = [
    "0-12 Months",
    "13-24 Months",
    "25-36 Months",
    "37-48 Months",
    "49-60 Months",
    "61-72 Months",
]
df["TenureGroup"] = pd.cut(
    df["tenure"], bins=[0, 12, 24, 36, 48, 60, 72], labels=labels
)

# 2. Create AvgMonthlySpend
# Avoid division by zero just in case tenure is 0 (though we dropped missing TotalCharges)
df["AvgMonthlySpend"] = np.where(
    df["tenure"] == 0, 0, df["TotalCharges"] / df["tenure"]
)

# 3. Convert binary Yes/No columns to 1/0
binary_columns = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# Map gender to 1/0 (Female=1, Male=0)
df["gender"] = df["gender"].map({"Female": 1, "Male": 0})

# 4. One-Hot Encode categorical columns (Contract, InternetService, etc.)
categorical_cols = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "TenureGroup",
]

# Create dummy variables and drop the first one to avoid dummy variable trap
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Feature engineering complete. Current shape:", df_encoded.shape)

# ==========================================
# 4. EDA & VISUALIZATION
# ==========================================
print("\n--- Generating Visualizations ---")

plt.figure(figsize=(20, 15))

# Plot 1: Countplot for Churn
plt.subplot(2, 2, 1)
sns.countplot(data=df, x="Churn", palette="viridis")
plt.title("Target Variable: Churn Distribution", fontsize=14)
plt.xticks(ticks=[0, 1], labels=["Retained (0)", "Churned (1)"])

# Plot 2: Barplot for Contract vs Churn
# (Using the original df for labels before encoding)
plt.subplot(2, 2, 2)
sns.barplot(data=df, x="Contract", y="Churn", palette="magma", errorbar=None)
plt.title("Churn Rate by Contract Type", fontsize=14)
plt.ylabel("Churn Rate (Percentage)")

# Plot 3: Boxplot for MonthlyCharges by Churn
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette="coolwarm")
plt.title("Monthly Charges vs Churn", fontsize=14)
plt.xticks(ticks=[0, 1], labels=["Retained", "Churned"])

# Plot 4: Heatmap of Correlations (Top 15 features most correlated with Churn)
plt.subplot(2, 2, 4)
# Get correlations with target variable
corr = df_encoded.corr()
top_corr_features = corr.index[abs(corr["Churn"]) > 0.15]
sns.heatmap(
    df_encoded[top_corr_features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
)
plt.title("Correlation Heatmap (Top Features)", fontsize=14)

plt.tight_layout()
plt.show()

# ==========================================
# 5. EXPORT DELIVERABLES
# ==========================================
# Export the fully cleaned and engineered dataset
output_filename = "Cleaned_Customer_Churn.csv"
df_encoded.to_csv(output_filename, index=False)
print(f"\n✅ Cleaned dataset successfully exported as '{output_filename}'")
