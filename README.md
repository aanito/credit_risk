Task 1: Understanding Credit Risk
Research Credit Risk Concepts

Review the provided references to understand:
Key definitions: default, creditworthiness, Basel II compliance.
Risk probability scoring and optimal loan determination.
Techniques for building alternative credit scoring models.
Define the Proxy Variable for Risk

Use FraudResult as a starting point to differentiate "bad" (high-risk) vs. "good" (low-risk) customers.
Augment this with additional derived metrics like overdue transactions or excessively high amounts spent relative to income.
Prepare for Feature Selection

Identify observable features in the dataset that are likely correlated with default behavior, such as transaction frequency, average transaction amount, and product category.
Task 2: Exploratory Data Analysis (EDA)
Data Overview
Inspect structure: Use Python/Pandas to review the dataset (e.g., .info(), .describe()).
Summarize key stats: Mean, median, standard deviation, and null counts for all fields.
Analyze Numerical Features
Distribution plots: Histograms, boxplots, and KDE plots for Amount, Value, and TransactionStartTime.
Correlation analysis: Pearson/Spearman correlation to find relationships between numerical variables.
Analyze Categorical Features
Value counts: Check the frequency of categories in CountryCode, ProductCategory, PricingStrategy, etc.
Cross-tabulations: Analyze relationships between FraudResult and categorical features.
Handle Missing Values
Identify missing values via .isnull().sum() and evaluate strategies:
Impute with mean/median for numerical features.
Replace missing categories with "Unknown" or the mode.
Detect Outliers
Use boxplots for Amount and Value to detect extreme values and decide whether to cap or transform these outliers.
Task 3: Model Development

1. Risk Probability Model
   Algorithm selection: Start with logistic regression, decision trees, or gradient boosting (e.g., XGBoost, LightGBM).
   Target variable: Use the proxy variable for default (FraudResult).
   Input features: Include significant predictors identified during EDA (e.g., transaction frequency, product category, amounts).
2. Credit Scoring
   Convert risk probabilities to credit scores using a scorecard approach:
   Normalize probabilities to a 300–850 range.
   Use thresholds (e.g., low-risk: 700+, moderate: 500–699, high-risk: below 500).
3. Optimal Loan Prediction
   Build a regression model to predict the optimal loan amount and duration.
   Input features: Customer historical spending patterns, income proxies, product categories.
   Target variables: Derived loan size and duration from risk scores.
