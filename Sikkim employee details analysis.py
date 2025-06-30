#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
# Load dataset
df = pd.read_excel(r"C:\Users\kanag\Downloads\Employee_Details_0.xlsx")
# Strip column names
df.columns = df.columns.str.strip()
#  Rename columns (fix typo)
df.rename(columns={'MartitalStatus': 'MaritalStatus'}, inplace=True)
#  Convert Excel serial dates to datetime
df['BirthDate'] = pd.to_datetime(df['BirthDate'], origin='1899-12-30', errors='coerce')
df['Schedule RetirementDate'] = pd.to_datetime(df['Schedule RetirementDate'], origin='1899-12-30', errors='coerce')
#  Calculate Age
df['Age'] = pd.Timestamp.now().year - df['BirthDate'].dt.year
#  Clean string columns: strip + title-case
string_cols = ['Gender', 'IsTeaching', 'Community Name', 'Religion Name', 'Caste Name', 'MaritalStatus']
for col in string_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()
#  Handle missing values
# Option 1: Fill specific missing values
df['SikkimSubjectNo'] = df['SikkimSubjectNo'].fillna(0)
# Option 2: Drop rows with any missing values (creates a cleaned copy)
df_cleaned = df.dropna()
# Drop duplicates if any
df_cleaned = df_cleaned.drop_duplicates()
# ===== Summary Reports =====
print(" DataFrame Info (After Cleaning):")
df_cleaned.info()
print("\n Statistical Summary (Numerical Columns):")
print(df_cleaned.describe())
# Optional: Save cleaned data
# df_cleaned.to_excel("Cleaned_Employee_Data.xlsx", index=False)


# In[44]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load cleaned data
df = pd.read_excel(r"C:\Users\kanag\Downloads\Employee_Details_0.xlsx")
# Fix column names and datatypes again just in case
df.columns = df.columns.str.strip()
df.rename(columns={'MartitalStatus': 'MaritalStatus'}, inplace=True)
df['BirthDate'] = pd.to_datetime(df['BirthDate'], origin='1899-12-30', errors='coerce')
df['Schedule RetirementDate'] = pd.to_datetime(df['Schedule RetirementDate'], origin='1899-12-30', errors='coerce')
df['Age'] = pd.Timestamp.now().year - df['BirthDate'].dt.year
string_cols = ['Gender', 'IsTeaching', 'Community Name', 'Religion Name', 'Caste Name', 'MaritalStatus']
for col in string_cols:
    df[col] = df[col].astype(str).str.strip().str.title()


# In[46]:


# ==========  Basic Info ==========
print(" Dataset Shape:", df.shape)
print(" Column Types & Nulls:")
print(df.info())
print("\n Description:")
print(df.describe(include='all'))


# In[63]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the file path
file_path = r"C:\Users\kanag\Downloads\Employee_Details_0.xlsx"

# Load the Excel file
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Check if 'Gender' column exists
if 'Gender' in df.columns:
    # Count gender values
    gender_counts = df['Gender'].value_counts()

    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        gender_counts,
        labels=gender_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette('pastel')
    )
    plt.title("Gender Distribution", fontsize=14)
    plt.axis('equal')  # Ensures it's a circle
    plt.tight_layout()
    plt.show()
else:
    print("❌ 'Gender' column not found in the dataset.")


# In[64]:


# ==========  Age Distribution ==========
plt.figure(figsize=(8,5))
sns.histplot(df['Age'].dropna(), bins=25, kde=True, color='skyblue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# In[62]:


# ==========  Teaching vs Non-Teaching ==========
plt.figure(figsize=(6,4))
sns.countplot(x='IsTeaching', data=df)
plt.title("Teaching vs Non-Teaching")
plt.show()


# In[60]:


# ==========  Caste / Community / Religion ==========
for col in ['Caste Name', 'Community Name', 'Religion Name']:
    plt.figure(figsize=(10,4))
    top_vals = df[col].value_counts().head(10)
    sns.barplot(x=top_vals.values, y=top_vals.index, palette='pastel')
    plt.title(f"Top 10 {col}")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.show()


# In[58]:


# ==========  Marital Status ==========
plt.figure(figsize=(6,4))
sns.countplot(x='MaritalStatus', data=df)
plt.title("Marital Status")
plt.show()


# In[56]:


# ==========  Top Designations ==========
plt.figure(figsize=(10,5))
top_designations = df['Designation'].value_counts().head(10)
sns.barplot(x=top_designations.values, y=top_designations.index, palette='muted')
plt.title("Top 10 Designations")
plt.xlabel("Count")
plt.ylabel("Designation")
plt.show()


# In[72]:


df_numeric = df.copy()
for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

# Drop columns with all NaNs or only 1 unique value
df_numeric = df_numeric.dropna(axis=1, how='all')
df_numeric = df_numeric.loc[:, df_numeric.nunique() > 1]

# Final correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# In[92]:


# Count values
status_counts = df['Appointment Status'].value_counts()

# Donut chart with clearer labels
fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    status_counts,
    labels=None,  # Hide inner labels
    autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',  # Show only if >2%
    startangle=140,
    pctdistance=0.85,  # Position of autopct
    wedgeprops={'width': 0.4}
)

# Add legend instead of inline labels
ax.legend(
    wedges, status_counts.index,
    title='Appointment Status',
    loc='center left',
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=10
)

# Title & layout
plt.title('Appointment Status Distribution (Donut Chart with Legend)', fontsize=14)
plt.tight_layout()
plt.show()


# In[96]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='IsTeaching', y='Hierarchy', data=df, palette='Set2')
plt.title("Distribution of Hierarchy by Teaching Status")
plt.xlabel("Is Teaching")
plt.ylabel("Hierarchy Level")
plt.tight_layout()
plt.show()


# In[49]:


import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Load the dataset
df = pd.read_excel(r"C:\Users\kanag\Downloads\Employee_Details_0.xlsx")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ------------------------------------------
# 1. Dataset Overview
# ------------------------------------------
print(" Dataset Shape:", df.shape)
print(" Column Names:\n", df.columns.tolist())
print("\n Data Types:\n", df.dtypes)

# ------------------------------------------
# 2. Numerical Summary for ALL Numerical Columns
# ------------------------------------------
numerical = df.select_dtypes(include=[np.number])

print("\n Numerical Summary (Full):")
print(numerical.describe().T)

# ------------------------------------------
# 3. Advanced Stats: Skewness & Kurtosis
# ------------------------------------------
print("\n Skewness & Kurtosis:")
for col in numerical.columns:
    print(f"\n🔸 {col}")
    print(f"Skewness : {skew(df[col].dropna()):.2f}")
    print(f"Kurtosis : {kurtosis(df[col].dropna()):.2f}")


# In[7]:


import pandas as pd
import numpy as np
df = pd.read_excel(r"C:\Users\kanag\Downloads\Employee_Details_0.xlsx")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Select numeric columns
numerical = df.select_dtypes(include=[np.number])

# ------------------------------------------
# 4. Central Tendency + Range
# ------------------------------------------
print("\n📏 Central Tendency + Spread:")
for col in numerical.columns:
    col_data = df[col].dropna()
    print(f"\n🔹 {col}")
    print(f"Mean      : {col_data.mean():.2f}")
    print(f"Median    : {col_data.median():.2f}")
    print(f"Mode      : {col_data.mode().iloc[0] if not col_data.mode().empty else 'N/A'}")
    print(f"Min       : {col_data.min()}")
    print(f"Max       : {col_data.max()}")
    print(f"Range     : {col_data.max() - col_data.min()}")
    print(f"Std Dev   : {col_data.std():.2f}")
    print(f"IQR       : {col_data.quantile(0.75) - col_data.quantile(0.25):.2f}")


# In[39]:


import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ztest
from scipy.stats import ttest_1samp, ttest_ind, chi2_contingency

# Load the dataset
df = pd.read_excel(r"C:\Users\kanag\Downloads\Employee_Details_0.xlsx")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Select only numerical columns
numerical = df.select_dtypes(include=[np.number])

# ------------------------------------------
# 1. Outlier Detection using IQR Method
# ------------------------------------------
print("\n Outlier Detection using IQR Method:")

for col in numerical.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"\n🔹 {col}")
    print(f"IQR Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Outliers Detected: {outliers.shape[0]}")


# In[45]:


# ------------------------------------------
# 2. One-Sample T-Test (using hierarchy as an example)
# ------------------------------------------
if 'hierarchy' in df.columns:
    t_stat, p_value = ttest_1samp(df['hierarchy'].dropna(), 100)
    print("\n One-sample T-Test: Is Hierarchy ≠ 100?")
    print(f"T-statistic: {t_stat:.2f}")
    print(f"P-value: {p_value:.4f}")
else:
    print("\n 'hierarchy' column not found for t-test.")


# In[47]:


from statsmodels.stats.weightstats import ztest

# ------------------------------------------
# One-Sample Z-Test (using hierarchy as an example)
# ------------------------------------------
if 'hierarchy' in df.columns:
    # Drop missing values
    hierarchy_data = df['hierarchy'].dropna()

    # Perform Z-Test (comparing to population mean = 100)
    z_stat, p_value = ztest(hierarchy_data, value=100)

    print("\n One-sample Z-Test: Is Hierarchy ≠ 100?")
    print(f"Z-statistic: {z_stat:.2f}")
    print(f"P-value: {p_value:.4f}")
else:
    print("\n 'hierarchy' column not found for Z-test.")

