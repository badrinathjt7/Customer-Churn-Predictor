import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import lime
import lime.lime_tabular
import joblib
import shap
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Customer Churn Analysis", layout="wide")

# UI Setup
st.title("Customer Churn Analysis Dashboard")
st.write("Upload your customer dataset to analyze churn patterns and predict customer behavior.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df_original = pd.read_csv(uploaded_file)
    
    # Display original data info
    st.sidebar.title("Filters")
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=df_original['Geography'].unique().tolist(),
        default=df_original['Geography'].unique().tolist()
    )
    
    # Filter dataset based on selected countries
    df = df_original[df_original['Geography'].isin(selected_countries)].copy()
    
    if len(selected_countries) == 0:
        st.warning("Please select at least one country to analyze.")
        st.stop()
    
    # Display selected data size
    st.sidebar.metric("Selected Data Size", len(df))
    
    # Calculate and display churn rate for selected countries
    churn_rate = (df['Exited'].mean() * 100).round(2)
    st.sidebar.metric("Churn Rate", f"{churn_rate}%")
    
    # Navigation
    page = st.sidebar.radio("Go to", ["Data Overview", "Customer Demographics", 
                                      "Financial Analysis", "Churn Analysis", 
                                      "Predictive Modeling"])

    if page == "Data Overview":
        st.header("Data Overview")
        
        # Total Customers, Average Credit Score, Age, and more
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("Average Credit Score", int(df['CreditScore'].mean()))
        with col3:
            st.metric("Average Age", round(df['Age'].mean(), 1))
        with col4:
            st.metric("Average Balance", round(df['Balance'].mean(), 1))

        # Display basic dataset information
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Display summary statistics for all columns
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
        
        # Distribution comparisons between countries
        st.subheader("Key Metrics by Country")
        fig = make_subplots(rows=3, cols=3,
                            subplot_titles=('Credit Score Distribution', 'Age Distribution',
                                            'Balance Distribution', 'Salary Distribution',
                                            'Tenure Distribution', 'Gender Distribution',
                                            'Active Member Distribution', 'Exited Distribution'))
        
        for idx, country in enumerate(selected_countries):
            country_data = df[df['Geography'] == country]
            
            # Credit Score
            fig.add_trace(
                go.Histogram(x=country_data['CreditScore'], name=f'{country} - Credit',
                             showlegend=True, opacity=0.7),
                row=1, col=1
            )
            
            # Age
            fig.add_trace(
                go.Histogram(x=country_data['Age'], name=f'{country} - Age',
                             showlegend=True, opacity=0.7),
                row=1, col=2
            )
            
            # Balance
            fig.add_trace(
                go.Histogram(x=country_data['Balance'], name=f'{country} - Balance',
                             showlegend=True, opacity=0.7),
                row=2, col=1
            )
            
            # Salary
            fig.add_trace(
                go.Histogram(x=country_data['EstimatedSalary'], name=f'{country} - Salary',
                             showlegend=True, opacity=0.7),
                row=2, col=2
            )
            
            # Tenure
            fig.add_trace(
                go.Histogram(x=country_data['Tenure'], name=f'{country} - Tenure',
                             showlegend=True, opacity=0.7),
                row=3, col=1
            )
            
            # Gender
            fig.add_trace(
                go.Histogram(x=country_data['Gender'], name=f'{country} - Gender',
                             showlegend=True, opacity=0.7),
                row=3, col=2
            )
            
            # Active Member
            fig.add_trace(
                go.Histogram(x=country_data['IsActiveMember'], name=f'{country} - Active Member',
                             showlegend=True, opacity=0.7),
                row=3, col=3
            )
            
            # Exited
            fig.add_trace(
                go.Histogram(x=country_data['Exited'], name=f'{country} - Exited',
                             showlegend=True, opacity=0.7),
                row=3, col=3
            )

        fig.update_layout(height=900)
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Customer Demographics":
        st.header("Customer Demographics")
        
        # Gender Distribution
        gender_dist = df['Gender'].value_counts(normalize=True) * 100
        fig_gender = px.pie(values=gender_dist, names=gender_dist.index,
                            title='Gender Distribution')
        st.plotly_chart(fig_gender)
        
        # Age Distribution
        fig_age = px.histogram(df, x='Age', color='Gender', nbins=20,
                               title='Age Distribution by Gender')
        st.plotly_chart(fig_age)

        # Geographic Distribution
        geo_dist = df['Geography'].value_counts(normalize=True) * 100
        fig_geo = px.bar(x=geo_dist.index, y=geo_dist.values, 
                         title='Geographic Distribution of Customers',
                         labels={'x': 'Geography', 'y': 'Percentage'})
        st.plotly_chart(fig_geo)

    elif page == "Financial Analysis":
        st.header("Financial Analysis")
        
        # Balance Distribution
        fig_balance = px.histogram(df, x='Balance', nbins=30,
                                   title='Balance Distribution')
        st.plotly_chart(fig_balance)
        
        # Salary Distribution
        fig_salary = px.histogram(df, x='EstimatedSalary', nbins=30,
                                  title='Estimated Salary Distribution')
        st.plotly_chart(fig_salary)

        # Credit Score Analysis by Country
        fig_credit_country = px.box(df, x='Geography', y='CreditScore', color='Geography',
                                    title='Credit Score by Geography')
        st.plotly_chart(fig_credit_country)

    elif page == "Churn Analysis":
        st.header("Churn Analysis")
        
        # Churn metrics by country
        churn_by_country = df.groupby('Geography')['Exited'].mean() * 100
        
        # Display metrics
        cols = st.columns(len(selected_countries))
        for idx, country in enumerate(selected_countries):
            if country in churn_by_country.index:
                cols[idx].metric(f"Churn Rate - {country}", 
                                 f"{churn_by_country[country]:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn by Geography
            churn_by_geo = pd.crosstab(df['Geography'], df['Exited'], normalize='index') * 100
            fig_geo_churn = px.bar(churn_by_geo, 
                                   title='Churn Rate by Country',
                                   labels={'value': 'Churn Rate (%)'})
            st.plotly_chart(fig_geo_churn)
            
            # Churn by Age Groups and Country
            df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], 
                                    labels=['<30', '30-40', '40-50', '50-60', '>60'])
            churn_by_age_country = pd.crosstab([df['Geography'], df['AgeGroup']], 
                                               df['Exited'], normalize='index') * 100
            
            churn_by_age_long = pd.melt(churn_by_age_country.reset_index(), 
                                        id_vars=['Geography', 'AgeGroup'], 
                                        var_name='Exited', value_name='Churn Rate')
            
            fig_age_churn = px.bar(churn_by_age_long,
                                   x='AgeGroup', y='Churn Rate', color='Exited',
                                   title='Churn Rate by Age Group and Country',
                                   labels={'Churn Rate': 'Churn Rate (%)'})
            st.plotly_chart(fig_age_churn)

        with col2:
            # Churn by Credit Score Range and Country
            df['CreditScoreGroup'] = pd.qcut(df['CreditScore'], q=5, 
                                             labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            churn_by_credit_country = pd.crosstab([df['Geography'], df['CreditScoreGroup']], 
                                                  df['Exited'], normalize='index') * 100
            
            churn_by_credit_long = pd.melt(churn_by_credit_country.reset_index(), 
                                           id_vars=['Geography', 'CreditScoreGroup'], 
                                           var_name='Exited', value_name='Churn Rate')
            
            fig_credit_churn = px.bar(churn_by_credit_long,
                                      x='CreditScoreGroup', y='Churn Rate', color='Exited',
                                      title='Churn Rate by Credit Score Group and Country',
                                      labels={'Churn Rate': 'Churn Rate (%)'})
            st.plotly_chart(fig_credit_churn)
            
            # Churn by Tenure
            churn_by_tenure = pd.crosstab(df['Tenure'], df['Exited'], normalize='index') * 100
            fig_tenure_churn = px.line(churn_by_tenure, 
                                       title='Churn Rate by Tenure',
                                       labels={'value': 'Churn Rate (%)'})
            st.plotly_chart(fig_tenure_churn)
    
    elif page == "Predictive Modeling":
        st.header("Predictive Modeling")
        
        # Preprocessing for modeling
        X = df.drop(columns=['Exited', 'CustomerId', 'Surname', 'RowNumber'])
        y = df['Exited']
        
        # One-hot encoding for categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Balance classes using SMOTE + Tomek Links
        smt = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
        
        # Model training
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_resampled, y_resampled)
        
        # Model evaluation
        y_pred = clf.predict(X_test)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Visualize Classification Report
        st.subheader("Model Performance")
        
        # Create a DataFrame from the classification report
        report_df = pd.DataFrame(class_report).transpose()
        
        # Visualization of Classification Metrics
        fig = go.Figure()
        
        # Add traces for precision, recall, and f1-score
        metrics = ['precision', 'recall', 'f1-score']
        colors = ['blue', 'green', 'red']
        
        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Bar(
                x=report_df.index[:-3],  # Exclude the last 3 rows (accuracy, macro avg, weighted avg)
                y=report_df.loc[report_df.index[:-3], metric],
                name=metric.capitalize(),
                marker_color=color
            ))
        
        fig.update_layout(
            title='Classification Report Metrics by Class',
            xaxis_title='Class',
            yaxis_title='Score',
            barmode='group'
        )
        
        st.plotly_chart(fig)
        
        # Display numerical classification report
        st.subheader("Detailed Classification Report")
        st.dataframe(report_df)
        
        # Accuracy Score
        st.metric("Accuracy Score", f"{accuracy_score(y_test, y_pred):.2%}")
        
        # Confusion Matrix Visualization
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, 
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Not Exited', 'Exited'],
                           y=['Not Exited', 'Exited'],
                           title="Confusion Matrix")
        st.plotly_chart(fig_cm)
        
        # Save the trained model
        model_filename = 'churn_model.pkl'
        joblib.dump(clf, model_filename)
        st.write(f"Model saved as {model_filename}")

        # LIME for Model Interpretation
        st.subheader("Model Explanation using LIME")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            class_names=['Not Exited', 'Exited'],
            mode='classification'
        )

        # Select an instance for explanation
        instance_idx = st.slider("Select an instance to explain", 0, len(X_test) - 1, 0)
        explanation = explainer.explain_instance(X_test.iloc[instance_idx].values, clf.predict_proba, num_features=5)
        explanation_html = explanation.as_html()
        st.components.v1.html(explanation_html, height=500)

        # SHAP for Global Feature Importance
        st.subheader("SHAP Feature Importance (Churn Analysis)")
        shap_explainer = shap.TreeExplainer(clf)
        shap_values = shap_explainer.shap_values(X_test)

        # Plot SHAP summary plot
        st.subheader("SHAP Summary Plot")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)