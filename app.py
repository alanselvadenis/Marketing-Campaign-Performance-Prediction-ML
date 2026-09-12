import streamlit as st
import pandas as pd
import joblib

def main():
    st.set_page_config(page_title="Campaign Predictor", layout="wide")
    st.title("Marketing Campaign Performance Prediction")
    st.markdown("Predict revenue and profitability for upcoming marketing campaigns.")

    # Load pre-trained models and feature configurations
    try:
        reg_model = joblib.load('revenue_model.pkl')
        clf_model = joblib.load('profit_model.pkl')
        reg_features = joblib.load('reg_features.pkl')
        clf_features = joblib.load('clf_features.pkl')
    except FileNotFoundError:
        st.error("Model files not found. Please run model_building.py first to generate the .pkl files.")
        return

    # Section 1: User Inputs
    st.header("1. Input Campaign Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quantitative Metrics")
        impressions = st.number_input("Impressions", min_value=0, value=15000)
        clicks = st.number_input("Clicks", min_value=0, value=1200)
        cost = st.number_input("Acquisition Cost ($)", min_value=0.0, value=500.0)
        duration = st.number_input("Duration (days)", min_value=1, value=30)
        engagement = st.number_input("Engagement Score", min_value=0.0, max_value=10.0, value=7.5)
        leads = st.number_input("Leads Generated", min_value=0, value=150)
        conversions = st.number_input("Conversions", min_value=0, value=15)

    with col2:
        st.subheader("Categorical Details")
        # Ensure these options match the unique values in your raw dataset
        campaign_type = st.selectbox("Campaign Type", ["Awareness", "Conversion", "Promotion", "Retention"])
        target_audience = st.selectbox("Target Audience", ["Men", "Women", "All Ages", "Youth"])
        language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Hindi"])
        customer_segment = st.selectbox("Customer Segment", ["New", "Returning", "Premium", "Standard"])
        
        # Multi-select for channels to replicate the multi-label encoding
        channels = st.multiselect("Channels Used", ["Email", "Social Media", "Search", "Display", "Influencer", "TV"])

    # Section 2: Prediction Engine
    if st.button("Predict Performance", type="primary"):
        st.header("2. Prediction Results")
        
        # Initialize an empty dictionary with all required regression features set to 0
        input_data = {col: 0 for col in reg_features}
        
        # Map quantitative numerical inputs
        input_data['Impressions'] = impressions
        input_data['Clicks'] = clicks
        input_data['Acquisition_Cost'] = cost
        input_data['Duration'] = duration
        input_data['Engagement_Score'] = engagement
        input_data['Leads'] = leads
        input_data['Conversions'] = conversions
        
        # Map categorical one-hot encoded inputs
        # The pd.get_dummies format is usually 'ColumnName_Value'
        cat_mappings = {
            f'Campaign_Type_{campaign_type}': 1,
            f'Target_Audience_{target_audience}': 1,
            f'Language_{language}': 1,
            f'Customer_Segment_{customer_segment}': 1
        }
        for feature, value in cat_mappings.items():
            if feature in input_data:
                input_data[feature] = value
                
        # Map multi-label binarizer channels
        for channel in channels:
            if channel in input_data:
                input_data[channel] = 1
        
        # Convert to DataFrame
        df_pred_reg = pd.DataFrame([input_data])
        
        # Predict Revenue (Regression)
        predicted_revenue = reg_model.predict(df_pred_reg)[0]
        
        # Predict Profitability (Classification)[cite: 1]
        # Filter the DataFrame to strictly use only the features required by the classification model
        df_pred_clf = df_pred_reg[[col for col in clf_features]]
        profit_prediction = clf_model.predict(df_pred_clf)[0]
        
        # Display Results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label="Predicted Revenue", value=f"${predicted_revenue:,.2f}")
            
        with res_col2:
            if profit_prediction == 1:
                st.success("Target Outcome: Profitable Campaign (Profit)")
            else:
                st.error("Target Outcome: Loss-Making Campaign (Loss)")
                
        # Section 3: Visualizing Inputs and Predictions[cite: 1]
        st.subheader("Cost vs. Predicted Revenue Analysis")
        chart_data = pd.DataFrame({
            "Category": ["Acquisition Cost", "Predicted Revenue"],
            "Amount ($)": [cost, predicted_revenue]
        })
        st.bar_chart(chart_data, x="Category", y="Amount ($)", color="Category")

if __name__ == "__main__":
    main()