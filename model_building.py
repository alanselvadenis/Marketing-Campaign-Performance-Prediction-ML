import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

def main():
    # 1. Data Collection
    print("Loading datasets...")
    path = r'Datasets' 
    all_files = glob.glob(os.path.join(path, "*_campaign_data_with_nulls.csv"))
    df = pd.concat([pd.read_csv(file) for file in all_files], ignore_index=True)

    # 2. Data Preprocessing
    print("Preprocessing data...")
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Save cleaned dataset deliverable
    df.to_csv('cleaned_dataset.csv', index=False)

    # 3. Feature Engineering
    print("Engineering features...")
    # Create Profit/Loss flag: 1 if ROI > 0 else 0
    df['Profit_Flag'] = df['ROI'].apply(lambda x: 1 if x > 0 else 0)

    # Multi-label encoding for Channel_Used
    df['Channel_Used'] = df['Channel_Used'].fillna('').str.split(',')
    mlb = MultiLabelBinarizer()
    encoded_channels = pd.DataFrame(mlb.fit_transform(df['Channel_Used']), columns=mlb.classes_, index=df.index)
    df = pd.concat([df.drop('Channel_Used', axis=1), encoded_channels], axis=1)

    # Save feature-engineered dataset deliverable
    df.to_csv('feature_engineered_dataset.csv', index=False)

    # 4. Prepare Data for Modeling
    # One-hot encode remaining categorical variables
    categorical_cols = ['Campaign_Type', 'Target_Audience', 'Language', 'Customer_Segment']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Drop identifiers and date for modeling
    model_data = df_encoded.drop(['Campaign_ID', 'Date'], axis=1)

    # 5. Regression Model (Predict Revenue)
    print("Training Regression Model...")
    X_reg = model_data.drop(['Revenue', 'Profit_Flag'], axis=1) 
    y_reg = model_data['Revenue']
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(Xr_train, yr_train)
    yr_pred = reg_model.predict(Xr_test)
    
    print(f"Regression Metrics -> R2: {r2_score(yr_test, yr_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(yr_test, yr_pred)):.4f}")

    # 6. Classification Model (Predict Profit/Loss)
    print("Training Classification Model...")
    # Prevent data leakage by excluding ROI and Revenue
    X_clf = model_data.drop(['Profit_Flag', 'ROI', 'Revenue'], axis=1) 
    y_clf = model_data['Profit_Flag']
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(Xc_train, yc_train)
    yc_pred = clf_model.predict(Xc_test)
    
    print(f"Classification Metrics -> Accuracy: {accuracy_score(yc_test, yc_pred):.4f}, Precision: {precision_score(yc_test, yc_pred):.4f}, Recall: {recall_score(yc_test, yc_pred):.4f}, F1: {f1_score(yc_test, yc_pred):.4f}")

    # 7. Export Models
    print("Exporting models...")
    joblib.dump(reg_model, 'revenue_model.pkl')
    joblib.dump(clf_model, 'profit_model.pkl')
    
    # Save the expected feature columns to align Streamlit inputs later
    joblib.dump(X_reg.columns.tolist(), 'reg_features.pkl')
    joblib.dump(X_clf.columns.tolist(), 'clf_features.pkl')
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()