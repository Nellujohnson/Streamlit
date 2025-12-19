import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Page config
st.set_page_config(
    page_title="EDA cu Streamlit",
    page_icon="📊",
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
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None

# Header
st.markdown('<h1 class="main-header">EDA cu Streamlit - Exploratory Data Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    selected_page = st.radio(
        "Select Module:",
        ["Upload & Filter", "Data Overview", "Numeric Analysis", "Categoric Analysis", "Correlation & Outliers"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.info("Upload a CSV file to begin analysis")

# ============================================================================
# CERINTA 1: Upload & Filter
# ============================================================================
if selected_page == "Upload & Filter":
    st.header("Upload CSV File")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.session_state.df_original = df.copy()

            # Success message
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            st.info(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")

            # Display first 10 rows
            st.subheader("First 10 rows:")
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("---")

            # FILTERING SECTION
            st.header("Filter Data")

            # Identify numeric and categoric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categoric_cols = df.select_dtypes(include=['object']).columns.tolist()

            df_filtered = df.copy()

            # Numeric filters (sliders)
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

            # Categoric filters (multiselect)
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

            # Store filtered data
            st.session_state.df_filtered = df_filtered

            # Display results
            st.markdown("---")
            st.subheader("Filter Results")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows before filter", len(df))
            with col2:
                st.metric("Rows after filter", len(df_filtered), delta=len(df_filtered) - len(df))

            # Display filtered data
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

        # Basic metrics
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

        # Data types
        st.subheader("Data Types")
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str)
        })
        st.dataframe(dtypes_df, use_container_width=True)

        st.markdown("---")

        # Missing values analysis
        st.subheader("Missing Values Analysis")

        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Percent': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

        if len(missing_data) > 0:
            st.dataframe(missing_data, use_container_width=True)

            # Visualization
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

        # Descriptive statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            st.subheader("Descriptive Statistics (Numeric Columns)")

            stats_df = df[numeric_cols].describe().T
            stats_df['median'] = df[numeric_cols].median()

            # Reorder columns
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

            # Column selection
            selected_col = st.selectbox("Select Numeric Column:", numeric_cols)

            if selected_col:
                st.markdown("---")

                # Histogram
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

                # Statistics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Mean", f"{df[selected_col].mean():.2f}")
                with col2:
                    st.metric("Median", f"{df[selected_col].median():.2f}")
                with col3:
                    st.metric("Std Deviation", f"{df[selected_col].std():.2f}")

                st.markdown("---")

                # Box plot
                st.subheader("Box Plot")

                fig_box = px.box(
                    df,
                    y=selected_col,
                    title=f'Box Plot: {selected_col}',
                    points='outliers'
                )
                st.plotly_chart(fig_box, use_container_width=True)

                # Box plot statistics
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
        st.warning("Please upload and filter data first in the 'Upload & Filter' section")

# ============================================================================
# CERINTA 4: Categoric Analysis
# ============================================================================
elif selected_page == "Categoric Analysis":
    if st.session_state.df_filtered is not None:
        df = st.session_state.df_filtered
        categoric_cols = df.select_dtypes(include=['object']).columns.tolist()

        if categoric_cols:
            st.header("Categoric Column Analysis")

            # Column selection
            selected_col = st.selectbox("Select Categoric Column:", categoric_cols)

            if selected_col:
                st.markdown("---")

                # Count plot (bar chart)
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

                # Frequency table
                st.subheader("Frequency Table")

                freq_table = value_counts.copy()
                freq_table.columns = ['Category', 'Absolute Frequency', 'Relative Frequency (%)']

                # Add total row
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
        st.warning("Please upload and filter data first in the 'Upload & Filter' section")

# ============================================================================
# CERINTA 5: Correlation & Outliers
# ============================================================================
elif selected_page == "Correlation & Outliers":
    if st.session_state.df_filtered is not None:
        df = st.session_state.df_filtered
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            st.header("Correlation Analysis & Outlier Detection")

            # Correlation matrix
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

            # Scatter plot
            st.subheader("Scatter Plot")

            col1, col2 = st.columns(2)

            with col1:
                x_var = st.selectbox("X Variable:", numeric_cols, key='x_var')
            with col2:
                y_var = st.selectbox("Y Variable:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key='y_var')

            if x_var and y_var:
                # Calculate correlation
                valid_data = df[[x_var, y_var]].dropna()
                if len(valid_data) >= 2:
                    pearson_corr = valid_data[x_var].corr(valid_data[y_var])

                    # Scatter plot WITHOUT trendline (to avoid statsmodels dependency)
                    fig_scatter = px.scatter(
                        df,
                        x=x_var,
                        y=y_var,
                        title=f'Scatter Plot: {x_var} vs {y_var}',
                        opacity=0.6
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    # Display correlation coefficient
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

            # Outlier detection using IQR method
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

                        # Show outlier values
                        outlier_values = outliers[col].values
                        st.write("**Outlier values:**")
                        st.write(", ".join([f"{val:.2f}" for val in outlier_values[:15]]))
                        if outlier_count > 15:
                            st.write(f"... and {outlier_count - 15} more")

                        # Visualize
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
        st.warning("Please upload and filter data first in the 'Upload & Filter' section")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>EDA cu Streamlit - Exploratory Data Analysis Tool</p>
    <p>Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)