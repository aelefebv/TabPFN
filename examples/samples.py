#%%
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor

import matplotlib.pyplot as plt

csv_path = r"C:\Users\austin\Downloads\patient_he.csv"
df = pd.read_csv(csv_path)
# drop rows where hAge is nan
df = df.dropna(subset=['hAge'])
df = df.dropna(subset=['ki67_per_rr'])
df = df.dropna(subset=['ki67_per_unit_length'])

# cols_to_ignore: im_name,duplicate,flag,age,length_for_validation, ...
# y = age column
cols_to_drop = ['im_name', 'length_for_validation', 'hAge']
# drop_containing = ['min', 'max', 'iqr', 'range', 'tortuosity', 'end_end']
# for col in df.columns:
#     if any([word in col for word in drop_containing]):
#         cols_to_drop.append(col)
# analysis_cols = ['ki67_per_rr','ki67_per_unit_length','itgb4_mean','area_per_length','perimeter_per_length','rete_ridges_per_length','thickness_mean','thickness_std','thickness_max','thickness_q25','thickness_q50','thickness_q75',thickness_iqr,rr_length_mean,rr_length_std,rr_length_max,rr_length_min,rr_length_q25,rr_length_q50,rr_length_q75,rr_length_iqr,rr_length_range,rr_end_end_distance_mean,rr_end_end_distance_std,rr_end_end_distance_max,rr_end_end_distance_min,rr_end_end_distance_q25,rr_end_end_distance_q50,rr_end_end_distance_q75,rr_end_end_distance_iqr,rr_end_end_distance_range,rr_tortuosity_mean,rr_tortuosity_std,rr_tortuosity_max,rr_tortuosity_min,rr_tortuosity_q25,rr_tortuosity_q50,rr_tortuosity_q75,rr_tortuosity_iqr,rr_tortuosity_range,rr_tip_thickness_mean,rr_tip_thickness_std,rr_tip_thickness_max,rr_tip_thickness_min,rr_tip_thickness_q25,rr_tip_thickness_q50,rr_tip_thickness_q75,rr_tip_thickness_iqr,rr_tip_thickness_range,rr_base_thickness_mean,rr_base_thickness_std,rr_base_thickness_max,rr_base_thickness_min,rr_base_thickness_q25,rr_base_thickness_q50,rr_base_thickness_q75,rr_base_thickness_iqr,rr_base_thickness_range,rr_tip_base_ratio_mean,rr_tip_base_ratio_std,rr_tip_base_ratio_max,rr_tip_base_ratio_min,rr_tip_base_ratio_q25,rr_tip_base_ratio_q50,rr_tip_base_ratio_q75,rr_tip_base_ratio_iqr,rr_tip_base_ratio_range,rr_aspect_ratio_mean,rr_aspect_ratio_std,rr_aspect_ratio_max,rr_aspect_ratio_min,rr_aspect_ratio_q25,rr_aspect_ratio_q50,rr_aspect_ratio_q75,rr_aspect_ratio_iqr,rr_aspect_ratio_range,rr_thickness_mean,rr_thickness_mean_std,rr_thickness_max,rr_thickness_min_q25,rr_thickness_mean_q50,rr_thickness_max_q75,rr_thickness_mean_iqr,rr_thickness_mean_range]
y = df['hAge']
X = df.drop(columns=cols_to_drop)
# drop any columns with nans
X = X.dropna(axis=1)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
)
# %%

# Initialize a regressor
reg = TabPFNRegressor(
    # n_estimators=50,
)
reg.fit(X_train, y_train)

# Predict a point estimate (using the mean)
predictions = reg.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))
# %%
# Predict quantiles
quantiles = [0.25, 0.5, 0.75]
quantile_predictions = reg.predict(
    X_test,
    output_type="quantiles",
    quantiles=quantiles,
)
for q, q_pred in zip(quantiles, quantile_predictions):
    print(f"Quantile {q} MAE:", mean_absolute_error(y_test, q_pred))

# Predict with mode
mode_predictions = reg.predict(X_test, output_type="mode")
print("Mode MAE:", mean_absolute_error(y_test, mode_predictions))
# %%
# visualize predictions
plt.figure(figsize=(10, 10))
plt.scatter(y_test, predictions, label='Test Predictions')
train_predictions = reg.predict(X_train)
plt.scatter(y_train, train_predictions, color='green', label='Train Predictions')
plt.plot([0, 100], [0, 100], 'r--', label='Ideal')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.legend()
plt.show()

# %%
# Let's use Random Forest regression with polynomial features
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

print(f"Number of features: {X_train.shape[1]}")

# Initialize and train standard Random Forest
rf_reg = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_reg.fit(X_train, y_train)

# Predict and evaluate standard RF
rf_predictions = rf_reg.predict(X_test)
print("\nStandard Random Forest Regression Results:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, rf_predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, rf_predictions))
print("R-squared (R^2):", r2_score(y_test, rf_predictions))

# Create a pipeline with polynomial features and Random Forest
poly_rf_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=3, include_bias=False)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the pipeline
poly_rf_pipeline.fit(X_train, y_train)

# Predict and evaluate poly RF
poly_rf_predictions = poly_rf_pipeline.predict(X_test)
print("\nPolynomial Random Forest Regression Results:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, poly_rf_predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, poly_rf_predictions))
print("R-squared (R^2):", r2_score(y_test, poly_rf_predictions))

# Compare all three models
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([0, 100], [0, 100], 'r--')
plt.title('TabPFN Model')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')

plt.subplot(1, 3, 2)
plt.scatter(y_test, rf_predictions, alpha=0.7)
plt.plot([0, 100], [0, 100], 'r--')
plt.title('Standard Random Forest')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')

plt.subplot(1, 3, 3)
plt.scatter(y_test, poly_rf_predictions, alpha=0.7)
plt.plot([0, 100], [0, 100], 'r--')
plt.title('Polynomial Random Forest')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')

plt.tight_layout()
plt.show()

# Feature importance for standard RF
feature_importances = pd.DataFrame(
    {'feature': X_train.columns, 'importance': rf_reg.feature_importances_}
).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['feature'][:10], feature_importances['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances (Standard RF)')
plt.tight_layout()
plt.show()
# %%
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

age_threshold = 45
new_y_train = y_train > age_threshold
new_y_test = y_test > age_threshold

# Initialize a classifier
clf = TabPFNClassifier()
clf.fit(X_train, new_y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(new_y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(new_y_test, predictions))

# plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(new_y_test, predictions)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



# %%
experimental_df = pd.read_csv(r"C:\Users\austin\Downloads\all_he.csv")
# %%
# drop rows where hAge is nan
experimental_df = experimental_df.dropna(subset=['hAge'])
experimental_df = experimental_df.dropna(subset=['ki67_per_rr'])
experimental_df = experimental_df.dropna(subset=['ki67_per_unit_length'])
experimental_df = experimental_df.dropna(subset=['itgb4_mean'])
# %%

# Prepare data for prediction
cols_to_drop = ['im_name', 'length_for_validation', 'hAge', 'Flag', 'Sample', 'mAge']
experimental_X = experimental_df.drop(columns=[col for col in cols_to_drop if col in experimental_df.columns])
experimental_X = experimental_X.dropna(axis=1)  # Drop columns with NaNs

# Run the model on this data
predicted_ages = reg.predict(experimental_X)

# Add predictions to the dataframe
experimental_df['predicted_age'] = predicted_ages

# Calculate deltaAge (true age - predicted age)
experimental_df['deltaAge'] = experimental_df['predicted_age'] - experimental_df['hAge']

# Group by sample identifier
sample_results = experimental_df.groupby('im_name').agg({
    'hAge': 'first',
    'predicted_age': 'mean',
    'deltaAge': 'mean'
}).reset_index()

# Display results
print("Sample-level age predictions:")
print(sample_results.head())

# %%
# save new experimental_df to csv
experimental_df.to_csv(r"C:\Users\austin\Downloads\all_he_with_predictions.csv", index=False)
# %%
