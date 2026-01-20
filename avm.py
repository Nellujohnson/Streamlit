import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve,
                             mean_absolute_error, mean_squared_error, r2_score)
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="EDA + ML cu Streamlit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = {}

# Header
st.markdown('<h1 class="main-header">EDA + ML cu Streamlit</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    selected_page = st.radio(
        "Select Module:",
        ["Upload & Filter", "Data Overview", "Numeric Analysis", "Categoric Analysis",
         "Correlation & Outliers", "ML: Problem Setup", "ML: Preprocessing",
         "ML: Train Models", "ML: Evaluation & Comparison"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.info("Upload a CSV file to begin")

# ============================================================================
# CERINTA 1: Upload & Filter
# ============================================================================
if selected_page == "Upload & Filter":
    st.header("Upload CSV File")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_original = df.copy()

            st.success(f"File uploaded successfully: {uploaded_file.name}")
            st.info(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")

            st.subheader("First 10 rows:")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("---")
            st.header("Filter Data")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categoric_cols = df.select_dtypes(include=['object']).columns.tolist()

            df_filtered = df.copy()

            if numeric_cols:
                st.subheader("Numeric Columns (Sliders)")
                filter_cols = st.columns(2)

                for idx, col in enumerate(numeric_cols):
                    with filter_cols[idx % 2]:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())

                        if min_val != max_val:
                            selected_range = st.slider(
                                f"{col}",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"slider_{col}"
                            )
                            df_filtered = df_filtered[
                                (df_filtered[col] >= selected_range[0]) &
                                (df_filtered[col] <= selected_range[1])
                                ]

            if categoric_cols:
                st.subheader("Categoric Columns (Multiselect)")
                filter_cols = st.columns(2)

                for idx, col in enumerate(categoric_cols):
                    with filter_cols[idx % 2]:
                        unique_values = df[col].dropna().unique().tolist()
                        selected_values = st.multiselect(
                            f"{col}",
                            options=unique_values,
                            default=unique_values,
                            key=f"multiselect_{col}"
                        )
                        if selected_values:
                            df_filtered = df_filtered[df_filtered[col].isin(selected_values)]

            st.session_state.df_filtered = df_filtered

            st.markdown("---")
            st.subheader("Filter Results")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows before filter", len(df))
            with col2:
                st.metric("Rows after filter", len(df_filtered), delta=len(df_filtered) - len(df))

            st.subheader("Filtered Data (first 10 rows):")
            st.dataframe(df_filtered.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin")

# ============================================================================
# CERINTA 2: Data Overview
# ============================================================================
elif selected_page == "Data Overview":
    if st.session_state.df_filtered is not None:
        df = st.session_state.df_filtered

        st.header("Data Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Number of Rows", len(df))
        with col2:
            st.metric("Number of Columns", len(df.columns))
        with col3:
            numeric_count = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_count)
        with col4:
            categoric_count = len(df.select_dtypes(include=['object']).columns)
            st.metric("Categoric Columns", categoric_count)

        st.markdown("---")

        st.subheader("Data Types")
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str)
        })
        st.dataframe(dtypes_df, use_container_width=True)

        st.markdown("---")

        st.subheader("Missing Values Analysis")

        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Percent': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

        if len(missing_data) > 0:
            st.dataframe(missing_data, use_container_width=True)

            fig = px.bar(
                missing_data,
                x='Column',
                y='Percent',
                title='Missing Values Percentage by Column',
                labels={'Percent': 'Missing %'},
                text='Missing Count'
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values detected!")

        st.markdown("---")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            st.subheader("Descriptive Statistics (Numeric Columns)")

            stats_df = df[numeric_cols].describe().T
            stats_df['median'] = df[numeric_cols].median()
            stats_df = stats_df[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']]

            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

    else:
        st.warning("Please upload and filter data first in the 'Upload & Filter' section")

# ============================================================================
# CERINTA 3: Numeric Analysis
# ============================================================================
elif selected_page == "Numeric Analysis":
    if st.session_state.df_filtered is not None:
        df = st.session_state.df_filtered
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            st.header("Numeric Column Analysis")

            selected_col = st.selectbox("Select Numeric Column:", numeric_cols)

            if selected_col:
                st.markdown("---")

                st.subheader("Interactive Histogram")

                num_bins = st.slider("Number of bins (10-100):", 10, 100, 30)

                fig_hist = px.histogram(
                    df,
                    x=selected_col,
                    nbins=num_bins,
                    title=f'Histogram: {selected_col}',
                    labels={selected_col: selected_col}
                )
                fig_hist.update_traces(marker_line_width=1, marker_line_color="white")
                st.plotly_chart(fig_hist, use_container_width=True)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Mean", f"{df[selected_col].mean():.2f}")
                with col2:
                    st.metric("Median", f"{df[selected_col].median():.2f}")
                with col3:
                    st.metric("Std Deviation", f"{df[selected_col].std():.2f}")

                st.markdown("---")

                st.subheader("Box Plot")

                fig_box = px.box(
                    df,
                    y=selected_col,
                    title=f'Box Plot: {selected_col}',
                    points='outliers'
                )
                st.plotly_chart(fig_box, use_container_width=True)

                q1 = df[selected_col].quantile(0.25)
                q2 = df[selected_col].quantile(0.50)
                q3 = df[selected_col].quantile(0.75)

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Min", f"{df[selected_col].min():.2f}")
                with col2:
                    st.metric("Q1 (25%)", f"{q1:.2f}")
                with col3:
                    st.metric("Median (50%)", f"{q2:.2f}")
                with col4:
                    st.metric("Q3 (75%)", f"{q3:.2f}")
                with col5:
                    st.metric("Max", f"{df[selected_col].max():.2f}")
        else:
            st.warning("No numeric columns found in the dataset")
    else:
        st.warning("Please upload and filter data first")

# ============================================================================
# CERINTA 4: Categoric Analysis
# ============================================================================
elif selected_page == "Categoric Analysis":
    if st.session_state.df_filtered is not None:
        df = st.session_state.df_filtered
        categoric_cols = df.select_dtypes(include=['object']).columns.tolist()

        if categoric_cols:
            st.header("Categoric Column Analysis")

            selected_col = st.selectbox("Select Categoric Column:", categoric_cols)

            if selected_col:
                st.markdown("---")

                st.subheader("Count Plot (Bar Chart)")

                value_counts = df[selected_col].value_counts().reset_index()
                value_counts.columns = [selected_col, 'Count']
                value_counts['Percent'] = (value_counts['Count'] / len(df) * 100).round(2)

                fig = px.bar(
                    value_counts,
                    x=selected_col,
                    y='Count',
                    title=f'Frequency Distribution: {selected_col}',
                    text='Count',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                st.subheader("Frequency Table")

                freq_table = value_counts.copy()
                freq_table.columns = ['Category', 'Absolute Frequency', 'Relative Frequency (%)']

                total_row = pd.DataFrame({
                    'Category': ['TOTAL'],
                    'Absolute Frequency': [freq_table['Absolute Frequency'].sum()],
                    'Relative Frequency (%)': [100.00]
                })
                freq_table = pd.concat([freq_table, total_row], ignore_index=True)

                st.dataframe(
                    freq_table.style.format({
                        'Absolute Frequency': '{:.0f}',
                        'Relative Frequency (%)': '{:.2f}'
                    }),
                    use_container_width=True
                )
        else:
            st.warning("No categoric columns found in the dataset")
    else:
        st.warning("Please upload and filter data first")

# ============================================================================
# CERINTA 5: Correlation & Outliers
# ============================================================================
elif selected_page == "Correlation & Outliers":
    if st.session_state.df_filtered is not None:
        df = st.session_state.df_filtered
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            st.header("Correlation Analysis & Outlier Detection")

            st.subheader("Correlation Matrix (Heatmap)")

            corr_matrix = df[numeric_cols].corr()

            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.3f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                title='Correlation Matrix'
            )
            fig_corr.update_xaxes(side="bottom")
            st.plotly_chart(fig_corr, use_container_width=True)

            st.info("Blue = Positive correlation, Red = Negative correlation")

            st.markdown("---")

            st.subheader("Scatter Plot")

            col1, col2 = st.columns(2)

            with col1:
                x_var = st.selectbox("X Variable:", numeric_cols, key='x_var')
            with col2:
                y_var = st.selectbox("Y Variable:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key='y_var')

            if x_var and y_var:
                valid_data = df[[x_var, y_var]].dropna()
                if len(valid_data) >= 2:
                    pearson_corr = valid_data[x_var].corr(valid_data[y_var])

                    fig_scatter = px.scatter(
                        df,
                        x=x_var,
                        y=y_var,
                        title=f'Scatter Plot: {x_var} vs {y_var}',
                        opacity=0.6
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    st.markdown(f"""
                    <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3;'>
                        <h3 style='margin: 0; color: #1976d2;'>Pearson Correlation Coefficient</h3>
                        <h1 style='margin: 10px 0; color: #1976d2;'>{pearson_corr:.4f}</h1>
                        <p style='margin: 0;'>
                            <strong>Strength:</strong> {'Weak' if abs(pearson_corr) < 0.3 else 'Moderate' if abs(pearson_corr) < 0.7 else 'Strong'} | 
                            <strong>Direction:</strong> {'Positive' if pearson_corr > 0 else 'Negative'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

            st.subheader("Outlier Detection (IQR Method)")

            for col in numeric_cols:
                with st.expander(f"{col}"):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_percent = (outlier_count / len(df) * 100) if len(df) > 0 else 0

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Outliers Count", outlier_count)
                    with col2:
                        st.metric("Percent", f"{outlier_percent:.2f}%")
                    with col3:
                        st.metric("Lower Bound", f"{lower_bound:.2f}")
                    with col4:
                        st.metric("Upper Bound", f"{upper_bound:.2f}")

                    if outlier_count > 0:
                        st.warning(f"{outlier_count} outliers detected")

                        outlier_values = outliers[col].values
                        st.write("**Outlier values:**")
                        st.write(", ".join([f"{val:.2f}" for val in outlier_values[:15]]))
                        if outlier_count > 15:
                            st.write(f"... and {outlier_count - 15} more")

                        fig_outlier = px.box(
                            df,
                            y=col,
                            title=f'Box Plot with Outliers: {col}',
                            points='outliers'
                        )
                        st.plotly_chart(fig_outlier, use_container_width=True)
                    else:
                        st.success("No outliers detected using IQR method")

        else:
            st.warning("Need at least 2 numeric columns for correlation analysis")
    else:
        st.warning("Please upload and filter data first")

# ============================================================================
# ML PARTEA 1: Problem Setup
# ============================================================================
elif selected_page == "ML: Problem Setup":
    st.header("Machine Learning: Problem Setup")

    if st.session_state.df_filtered is not None:
        df = st.session_state.df_filtered

        st.subheader("1. Select Target Column")

        all_columns = df.columns.tolist()
        target_col = st.selectbox("Target Variable (what you want to predict):", all_columns)

        if target_col:
            st.session_state['target_col'] = target_col

            # Determine problem type
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                problem_type = "Classification"
                st.info(f"Detected problem type: **Classification** ({df[target_col].nunique()} classes)")
            else:
                problem_type = "Regression"
                st.info(f"Detected problem type: **Regression** (continuous values)")

            st.session_state['problem_type'] = problem_type

            st.markdown("---")

            st.subheader("2. Select Features")

            feature_selection_mode = st.radio(
                "Feature selection mode:",
                ["Select All (except target)", "Manual Selection", "Exclude Columns"]
            )

            available_features = [col for col in all_columns if col != target_col]

            if feature_selection_mode == "Select All (except target)":
                selected_features = available_features
            elif feature_selection_mode == "Manual Selection":
                selected_features = st.multiselect(
                    "Select features to use:",
                    available_features,
                    default=available_features
                )
            else:  # Exclude Columns
                excluded = st.multiselect(
                    "Select columns to exclude:",
                    available_features
                )
                selected_features = [col for col in available_features if col not in excluded]

            st.session_state['selected_features'] = selected_features

            if selected_features:
                st.success(f"Selected {len(selected_features)} features")

                with st.expander("View selected features"):
                    col1, col2 = st.columns(2)

                    numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
                    categoric_features = df[selected_features].select_dtypes(include=['object']).columns.tolist()

                    with col1:
                        st.write("**Numeric Features:**")
                        st.write(numeric_features if numeric_features else "None")

                    with col2:
                        st.write("**Categoric Features:**")
                        st.write(categoric_features if categoric_features else "None")

                st.markdown("---")

                st.subheader("3. Target Variable Distribution")

                if problem_type == "Classification":
                    value_counts = df[target_col].value_counts()
                    fig = px.bar(
                        x=value_counts.index.astype(str),
                        y=value_counts.values,
                        title=f'Target Distribution: {target_col}',
                        labels={'x': target_col, 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(
                        df,
                        x=target_col,
                        title=f'Target Distribution: {target_col}',
                        nbins=50
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean", f"{df[target_col].mean():.2f}")
                    with col2:
                        st.metric("Median", f"{df[target_col].median():.2f}")
                    with col3:
                        st.metric("Std Dev", f"{df[target_col].std():.2f}")
            else:
                st.warning("Please select at least one feature")
    else:
        st.warning("Please upload data first in the 'Upload & Filter' section")

# ============================================================================
# ML PARTEA 2: Preprocessing
# ============================================================================
elif selected_page == "ML: Preprocessing":
    st.header("Machine Learning: Preprocessing Configuration")

    if 'target_col' in st.session_state and 'selected_features' in st.session_state:
        df = st.session_state.df_filtered
        target_col = st.session_state['target_col']
        selected_features = st.session_state['selected_features']

        X = df[selected_features]
        y = df[target_col]

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categoric_features = X.select_dtypes(include=['object']).columns.tolist()

        st.subheader("1. Imputation Strategy")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Numeric Imputation:**")
            numeric_impute = st.selectbox(
                "Method for numeric columns:",
                ["mean", "median", "most_frequent"],
                help="Strategy to fill missing values in numeric columns"
            )

        with col2:
            st.write("**Categoric Imputation:**")
            categoric_impute = st.selectbox(
                "Method for categoric columns:",
                ["most_frequent", "constant"],
                help="Strategy to fill missing values in categoric columns"
            )

        st.session_state['numeric_impute'] = numeric_impute
        st.session_state['categoric_impute'] = categoric_impute

        st.markdown("---")

        st.subheader("2. Scaling Strategy")

        scaling_method = st.selectbox(
            "Select scaling method for numeric features:",
            ["StandardScaler", "MinMaxScaler", "None"],
            help="StandardScaler: mean=0, std=1 | MinMaxScaler: range [0,1]"
        )

        st.session_state['scaling_method'] = scaling_method

        st.markdown("---")

        st.subheader("3. Feature Selection (Optional)")

        use_feature_selection = st.checkbox("Use SelectKBest for feature selection")

        if use_feature_selection:
            k_features = st.slider(
                "Number of top features to select:",
                min_value=1,
                max_value=len(selected_features),
                value=min(10, len(selected_features))
            )
            st.session_state['use_feature_selection'] = True
            st.session_state['k_features'] = k_features
        else:
            st.session_state['use_feature_selection'] = False

        st.markdown("---")

        st.subheader("4. Outlier Removal (Optional)")

        remove_outliers = st.checkbox("Remove outliers using IQR method")

        if remove_outliers:
            iqr_multiplier = st.slider(
                "IQR multiplier (higher = less aggressive):",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1
            )
            st.session_state['remove_outliers'] = True
            st.session_state['iqr_multiplier'] = iqr_multiplier
        else:
            st.session_state['remove_outliers'] = False

        st.markdown("---")

        st.subheader("5. Train/Test Split Configuration")

        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider(
                "Test set size (%):",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            )
            st.session_state['test_size'] = test_size / 100

        with col2:
            random_state = st.number_input(
                "Random state (for reproducibility):",
                min_value=0,
                max_value=1000,
                value=42
            )
            st.session_state['random_state'] = random_state

        st.info(f"Split: {100 - test_size}% train / {test_size}% test")

        st.markdown("---")

        st.subheader("Pipeline Summary")

        st.write("**Preprocessing pipeline will include:**")
        st.write(f"1. Imputation - Numeric: {numeric_impute}, Categoric: {categoric_impute}")
        st.write(f"2. Encoding - Categoric features will be one-hot encoded")
        st.write(f"3. Scaling - {scaling_method}")
        if use_feature_selection:
            st.write(f"4. Feature Selection - Top {k_features} features")
        if remove_outliers:
            st.write(f"5. Outlier Removal - IQR method (multiplier: {iqr_multiplier})")

    else:
        st.warning("Please complete the 'ML: Problem Setup' section first")

# ============================================================================
# ML PARTEA 3: Train Models
# ============================================================================
elif selected_page == "ML: Train Models":
    st.header("Machine Learning: Model Training")

    if 'target_col' in st.session_state and 'selected_features' in st.session_state:
        df = st.session_state.df_filtered
        problem_type = st.session_state['problem_type']

        st.subheader(f"1. Select Models ({problem_type})")

        if problem_type == "Classification":
            available_models = {
                "Logistic Regression": LogisticRegression,
                "Random Forest": RandomForestClassifier,
                "SVM": SVC,
                "KNN": KNeighborsClassifier,
                "Gradient Boosting": GradientBoostingClassifier
            }
        else:
            available_models = {
                "Linear Regression": LinearRegression,
                "Ridge Regression": Ridge,
                "Lasso Regression": Lasso,
                "Random Forest": RandomForestRegressor,
                "SVR": SVR,
                "Gradient Boosting": GradientBoostingRegressor
            }

        selected_models = st.multiselect(
            "Select 2-3 models to train and compare:",
            list(available_models.keys()),
            default=list(available_models.keys())[:3]
        )

        if len(selected_models) >= 2:
            st.markdown("---")

            st.subheader("2. Configure Hyperparameters")

            model_params = {}

            for model_name in selected_models:
                with st.expander(f"Configure {model_name}"):
                    if model_name == "Logistic Regression":
                        c = st.slider("C (Regularization strength)", 0.01, 10.0, 1.0, key=f"{model_name}_C")
                        max_iter = st.slider("Max iterations", 100, 1000, 200, key=f"{model_name}_iter")
                        solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"], key=f"{model_name}_solver")
                        model_params[model_name] = {"C": c, "max_iter": max_iter, "solver": solver,
                                                    "random_state": st.session_state.get('random_state', 42)}

                    elif model_name == "Random Forest":
                        n_estimators = st.slider("Number of trees", 10, 200, 100, key=f"{model_name}_trees")
                        max_depth = st.slider("Max depth", 2, 50, 10, key=f"{model_name}_depth")
                        min_samples = st.slider("Min samples split", 2, 20, 2, key=f"{model_name}_samples")
                        model_params[model_name] = {"n_estimators": n_estimators, "max_depth": max_depth,
                                                    "min_samples_split": min_samples,
                                                    "random_state": st.session_state.get('random_state', 42)}

                    elif model_name == "SVM":
                        c = st.slider("C (Regularization)", 0.1, 10.0, 1.0, key=f"{model_name}_C")
                        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key=f"{model_name}_kernel")
                        gamma = st.selectbox("Gamma", ["scale", "auto"], key=f"{model_name}_gamma")
                        model_params[model_name] = {"C": c, "kernel": kernel, "gamma": gamma,
                                                    "random_state": st.session_state.get('random_state', 42)}

                    elif model_name == "KNN":
                        n_neighbors = st.slider("Number of neighbors", 1, 50, 5, key=f"{model_name}_neighbors")
                        weights = st.selectbox("Weights", ["uniform", "distance"], key=f"{model_name}_weights")
                        metric = st.selectbox("Distance metric", ["euclidean", "manhattan", "minkowski"],
                                              key=f"{model_name}_metric")
                        model_params[model_name] = {"n_neighbors": n_neighbors, "weights": weights, "metric": metric}

                    elif model_name == "Gradient Boosting":
                        n_estimators = st.slider("Number of estimators", 10, 200, 100, key=f"{model_name}_estimators")
                        learning_rate = st.slider("Learning rate", 0.01, 1.0, 0.1, key=f"{model_name}_lr")
                        max_depth = st.slider("Max depth", 2, 10, 3, key=f"{model_name}_depth")
                        model_params[model_name] = {"n_estimators": n_estimators, "learning_rate": learning_rate,
                                                    "max_depth": max_depth,
                                                    "random_state": st.session_state.get('random_state', 42)}

                    elif model_name == "Linear Regression":
                        fit_intercept = st.checkbox("Fit intercept", value=True, key=f"{model_name}_intercept")
                        model_params[model_name] = {"fit_intercept": fit_intercept}

                    elif model_name == "Ridge Regression":
                        alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0, key=f"{model_name}_alpha")
                        solver = st.selectbox("Solver", ["auto", "svd", "cholesky", "lsqr"], key=f"{model_name}_solver")
                        model_params[model_name] = {"alpha": alpha, "solver": solver,
                                                    "random_state": st.session_state.get('random_state', 42)}

                    elif model_name == "Lasso Regression":
                        alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0, key=f"{model_name}_alpha")
                        max_iter = st.slider("Max iterations", 100, 5000, 1000, key=f"{model_name}_iter")
                        model_params[model_name] = {"alpha": alpha, "max_iter": max_iter,
                                                    "random_state": st.session_state.get('random_state', 42)}

                    elif model_name == "SVR":
                        c = st.slider("C (Regularization)", 0.1, 10.0, 1.0, key=f"{model_name}_C")
                        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key=f"{model_name}_kernel")
                        epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1, key=f"{model_name}_epsilon")
                        model_params[model_name] = {"C": c, "kernel": kernel, "epsilon": epsilon}

            st.markdown("---")

            st.subheader("3. Train Models")

            if st.button("Start Training", type="primary"):
                try:
                    with st.spinner("Training models..."):
                        # Prepare data
                        target_col = st.session_state['target_col']
                        selected_features = st.session_state['selected_features']

                        X = df[selected_features].copy()
                        y = df[target_col].copy()

                        # Remove outliers if selected
                        if st.session_state.get('remove_outliers', False):
                            numeric_cols = X.select_dtypes(include=[np.number]).columns
                            multiplier = st.session_state.get('iqr_multiplier', 1.5)

                            mask = pd.Series([True] * len(X))
                            for col in numeric_cols:
                                Q1 = X[col].quantile(0.25)
                                Q3 = X[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower = Q1 - multiplier * IQR
                                upper = Q3 + multiplier * IQR
                                mask &= (X[col] >= lower) & (X[col] <= upper)

                            X = X[mask]
                            y = y[mask]
                            st.info(f"Removed {(~mask).sum()} outliers. Remaining samples: {len(X)}")

                        # Split data
                        test_size = st.session_state.get('test_size', 0.2)
                        random_state = st.session_state.get('random_state', 42)

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state
                        )

                        st.info(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

                        # Build preprocessing pipeline
                        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                        categoric_features = X.select_dtypes(include=['object']).columns.tolist()

                        numeric_transformer_steps = []
                        numeric_impute = st.session_state.get('numeric_impute', 'mean')
                        numeric_transformer_steps.append(('imputer', SimpleImputer(strategy=numeric_impute)))

                        scaling_method = st.session_state.get('scaling_method', 'StandardScaler')
                        if scaling_method == 'StandardScaler':
                            numeric_transformer_steps.append(('scaler', StandardScaler()))
                        elif scaling_method == 'MinMaxScaler':
                            numeric_transformer_steps.append(('scaler', MinMaxScaler()))

                        numeric_transformer = Pipeline(steps=numeric_transformer_steps)

                        categoric_impute = st.session_state.get('categoric_impute', 'most_frequent')
                        categoric_transformer = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy=categoric_impute, fill_value='missing')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                        ])

                        transformers = []
                        if numeric_features:
                            transformers.append(('num', numeric_transformer, numeric_features))
                        if categoric_features:
                            transformers.append(('cat', categoric_transformer, categoric_features))

                        preprocessor = ColumnTransformer(transformers=transformers)

                        # Encode target if classification
                        if problem_type == "Classification":
                            le = LabelEncoder()
                            y_train_encoded = le.fit_transform(y_train)
                            y_test_encoded = le.transform(y_test)
                            st.session_state['label_encoder'] = le
                        else:
                            y_train_encoded = y_train
                            y_test_encoded = y_test

                        # Train models
                        trained_models = {}
                        results = {}

                        progress_bar = st.progress(0)

                        for idx, model_name in enumerate(selected_models):
                            st.write(f"Training {model_name}...")

                            # Create model
                            model_class = available_models[model_name]
                            params = model_params[model_name]
                            model = model_class(**params)

                            # Create full pipeline
                            if st.session_state.get('use_feature_selection', False):
                                k_features = st.session_state.get('k_features', 10)
                                score_func = f_classif if problem_type == "Classification" else f_regression

                                full_pipeline = Pipeline([
                                    ('preprocessor', preprocessor),
                                    ('feature_selection', SelectKBest(score_func=score_func, k=k_features)),
                                    ('model', model)
                                ])
                            else:
                                full_pipeline = Pipeline([
                                    ('preprocessor', preprocessor),
                                    ('model', model)
                                ])

                            # Train
                            full_pipeline.fit(X_train, y_train_encoded)

                            # Predict
                            y_pred_train = full_pipeline.predict(X_train)
                            y_pred_test = full_pipeline.predict(X_test)

                            # Store
                            trained_models[model_name] = full_pipeline
                            results[model_name] = {
                                'y_train_true': y_train_encoded,
                                'y_train_pred': y_pred_train,
                                'y_test_true': y_test_encoded,
                                'y_test_pred': y_pred_test
                            }

                            # Probability predictions for classification
                            if problem_type == "Classification" and hasattr(full_pipeline.named_steps['model'],
                                                                            'predict_proba'):
                                try:
                                    y_pred_proba = full_pipeline.predict_proba(X_test)
                                    results[model_name]['y_pred_proba'] = y_pred_proba
                                except:
                                    pass

                            progress_bar.progress((idx + 1) / len(selected_models))

                        # Store results
                        st.session_state['trained_models'] = trained_models
                        st.session_state['ml_results'] = results
                        st.session_state['X_train'] = X_train
                        st.session_state['X_test'] = X_test
                        st.session_state['y_train'] = y_train
                        st.session_state['y_test'] = y_test

                        st.success(f"Successfully trained {len(selected_models)} models!")
                        st.info("Go to 'ML: Evaluation & Comparison' to see results")

                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())

        else:
            st.warning("Please select at least 2 models")

    else:
        st.warning("Please complete the 'ML: Problem Setup' and 'ML: Preprocessing' sections first")

# ============================================================================
# ML PARTEA 4: Evaluation & Comparison
# ============================================================================
elif selected_page == "ML: Evaluation & Comparison":
    st.header("Machine Learning: Evaluation & Comparison")

    if st.session_state.get('ml_results'):
        problem_type = st.session_state['problem_type']
        results = st.session_state['ml_results']

        st.subheader("Model Performance Comparison")

        # Calculate metrics
        comparison_data = []

        if problem_type == "Classification":
            for model_name, result in results.items():
                y_true = result['y_test_true']
                y_pred = result['y_test_pred']

                accuracy = accuracy_score(y_true, y_pred)

                # Handle multiclass
                n_classes = len(np.unique(y_true))
                avg_method = 'binary' if n_classes == 2 else 'weighted'

                precision = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
                recall = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
                f1 = f1_score(y_true, y_pred, average=avg_method, zero_division=0)

                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                })

            comparison_df = pd.DataFrame(comparison_data)

            # Display table
            st.dataframe(
                comparison_df.style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1-Score': '{:.4f}'
                }).highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen'),
                use_container_width=True
            )

            # Best model selection
            st.markdown("---")
            st.subheader("Best Model Selection")

            metric_choice = st.selectbox(
                "Select metric for best model:",
                ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            )

            best_idx = comparison_df[metric_choice].idxmax()
            best_model = comparison_df.loc[best_idx, 'Model']
            best_score = comparison_df.loc[best_idx, metric_choice]

            st.success(f"Best Model: **{best_model}** with {metric_choice}: **{best_score:.4f}**")

            # Visualizations
            st.markdown("---")
            st.subheader("Performance Visualizations")

            # Bar chart comparison
            fig = go.Figure()

            for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    text=comparison_df[metric].round(3),
                    textposition='auto'
                ))

            fig.update_layout(
                title='Model Comparison - All Metrics',
                barmode='group',
                xaxis_title='Model',
                yaxis_title='Score',
                yaxis_range=[0, 1]
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed results per model
            st.markdown("---")
            st.subheader("Detailed Results per Model")

            for model_name, result in results.items():
                with st.expander(f"{model_name} - Detailed Metrics"):
                    y_true = result['y_test_true']
                    y_pred = result['y_test_pred']

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Confusion Matrix:**")
                        cm = confusion_matrix(y_true, y_pred)

                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            aspect='auto',
                            color_continuous_scale='Blues',
                            labels=dict(x="Predicted", y="Actual"),
                            title=f'Confusion Matrix - {model_name}'
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with col2:
                        st.write("**Classification Metrics:**")

                        n_classes = len(np.unique(y_true))
                        avg_method = 'binary' if n_classes == 2 else 'weighted'

                        st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
                        st.metric("Precision",
                                  f"{precision_score(y_true, y_pred, average=avg_method, zero_division=0):.4f}")
                        st.metric("Recall", f"{recall_score(y_true, y_pred, average=avg_method, zero_division=0):.4f}")
                        st.metric("F1-Score", f"{f1_score(y_true, y_pred, average=avg_method, zero_division=0):.4f}")

                    # ROC curve for binary classification
                    if n_classes == 2 and 'y_pred_proba' in result:
                        st.write("**ROC Curve:**")

                        y_proba = result['y_pred_proba'][:, 1]
                        fpr, tpr, _ = roc_curve(y_true, y_proba)
                        auc_score = roc_auc_score(y_true, y_proba)

                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'ROC (AUC = {auc_score:.3f})',
                            line=dict(color='blue', width=2)
                        ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random',
                            line=dict(color='red', width=2, dash='dash')
                        ))

                        fig_roc.update_layout(
                            title=f'ROC Curve - {model_name}',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            showlegend=True
                        )

                        st.plotly_chart(fig_roc, use_container_width=True)

        else:  # Regression
            for model_name, result in results.items():
                y_true = result['y_test_true']
                y_pred = result['y_test_pred']

                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)

                comparison_data.append({
                    'Model': model_name,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2
                })

            comparison_df = pd.DataFrame(comparison_data)

            # Display table
            st.dataframe(
                comparison_df.style.format({
                    'MAE': '{:.4f}',
                    'MSE': '{:.4f}',
                    'RMSE': '{:.4f}',
                    'R2': '{:.4f}'
                }).highlight_min(subset=['MAE', 'MSE', 'RMSE'], color='lightgreen')
                .highlight_max(subset=['R2'], color='lightgreen'),
                use_container_width=True
            )

            # Best model selection
            st.markdown("---")
            st.subheader("Best Model Selection")

            metric_choice = st.selectbox(
                "Select metric for best model:",
                ['R2', 'MAE', 'RMSE', 'MSE']
            )

            if metric_choice == 'R2':
                best_idx = comparison_df[metric_choice].idxmax()
            else:
                best_idx = comparison_df[metric_choice].idxmin()

            best_model = comparison_df.loc[best_idx, 'Model']
            best_score = comparison_df.loc[best_idx, metric_choice]

            st.success(f"Best Model: **{best_model}** with {metric_choice}: **{best_score:.4f}**")

            # Visualizations
            st.markdown("---")
            st.subheader("Performance Visualizations")

            # Metrics comparison
            fig = go.Figure()

            # Normalize metrics for better visualization
            for metric in ['MAE', 'RMSE', 'R2']:
                if metric in comparison_df.columns:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=comparison_df['Model'],
                        y=comparison_df[metric],
                        text=comparison_df[metric].round(3),
                        textposition='auto'
                    ))

            fig.update_layout(
                title='Model Comparison - Metrics',
                barmode='group',
                xaxis_title='Model',
                yaxis_title='Score'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed results per model
            st.markdown("---")
            st.subheader("Detailed Results per Model")

            for model_name, result in results.items():
                with st.expander(f"{model_name} - Detailed Analysis"):
                    y_true = result['y_test_true']
                    y_pred = result['y_test_pred']

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Regression Metrics:**")
                        st.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.4f}")
                        st.metric("MSE", f"{mean_squared_error(y_true, y_pred):.4f}")
                        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
                        st.metric("R² Score", f"{r2_score(y_true, y_pred):.4f}")

                    with col2:
                        st.write("**Prediction vs Actual:**")

                        fig_scatter = go.Figure()

                        fig_scatter.add_trace(go.Scatter(
                            x=y_true,
                            y=y_pred,
                            mode='markers',
                            name='Predictions',
                            marker=dict(size=8, opacity=0.6)
                        ))

                        # Perfect prediction line
                        min_val = min(y_true.min(), y_pred.min())
                        max_val = max(y_true.max(), y_pred.max())
                        fig_scatter.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')
                        ))

                        fig_scatter.update_layout(
                            title=f'Actual vs Predicted - {model_name}',
                            xaxis_title='Actual Values',
                            yaxis_title='Predicted Values'
                        )

                        st.plotly_chart(fig_scatter, use_container_width=True)

                    # Residuals plot
                    st.write("**Residuals Plot:**")

                    residuals = y_true - y_pred

                    fig_residuals = go.Figure()

                    fig_residuals.add_trace(go.Scatter(
                        x=y_pred,
                        y=residuals,
                        mode='markers',
                        marker=dict(size=8, opacity=0.6)
                    ))

                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")

                    fig_residuals.update_layout(
                        title=f'Residuals Plot - {model_name}',
                        xaxis_title='Predicted Values',
                        yaxis_title='Residuals'
                    )

                    st.plotly_chart(fig_residuals, use_container_width=True)

    else:
        st.warning("Please train models first in the 'ML: Train Models' section")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>EDA + ML cu Streamlit - Complete Data Analysis & Machine Learning Tool</p>
    <p>Built with Streamlit, Scikit-learn & Plotly</p>
</div>
""", unsafe_allow_html=True)