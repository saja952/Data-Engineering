import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv("Dataset - Updated.csv")

st.title("Medical Dataset Analysis")

st.subheader("Dataset Overview")
st.write(f"**Number of Rows:** {df.shape[0]}")
st.write(f"**Number of Columns:** {df.shape[1]}")
st.write("**Missing Values per Column:**")
st.write(df.isnull().sum())

st.subheader("Dataset Preview")
st.dataframe(df.head())

column_descriptions = {
    "Age": "Age of the patient in years.",
    "Systolic BP": "Systolic Blood Pressure (upper value of BP).",
    "Diastolic": "Diastolic Blood Pressure (lower value of BP).",
    "BS": "Blood Sugar level of the patient.",
    "Body Temp": "Body Temperature in Celsius.",
    "BMI": "Body Mass Index, indicator of body fat based on weight and height.",
    "Previous Complications": "History of previous medical complications.",
    "Preexisting Diabetes": "Whether the patient has diabetes before pregnancy (Yes/No).",
    "Gestational Diabetes": "Presence of gestational diabetes (Yes/No).",
    "Mental Health": "Mental health condition of the patient.",
    "Heart Rate": "Patient’s heart rate (beats per minute).",
    "Risk Level": "Predicted medical risk level (Low, Mid, High)."
}

selected_col = st.sidebar.selectbox("Choose a column to analyze:", df.columns)

st.sidebar.markdown(f"**Description:** {column_descriptions.get(selected_col, 'No description available')}")

st.subheader(f"Analysis of {selected_col}")

if np.issubdtype(df[selected_col].dtype, np.number):
    st.write("**Statistical Summary:**")
    st.write(df[selected_col].describe())


else:
    st.write("**Value Counts:**")
    st.write(df[selected_col].value_counts())

st.subheader("Handle Missing Values (Per Column)")

methods = {}
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if col == "Risk Level":
            methods[col] = st.selectbox(
                f"Choose method for column '{col}':",
                ["DropRows", "Mode"],
                key=col
            )
        elif np.issubdtype(df[col].dtype, np.number):
            methods[col] = st.selectbox(
                f"Choose method for column '{col}':",
                ["Mean", "Median"],
                key=col
            )
        else:
            methods[col] = st.selectbox(
                f"Choose method for column '{col}':",
                ["Mode" ],
                key=col
            )

if st.button("Run Missing Value Handling"):
    df_cleaned = df.copy()
    for col, method in methods.items():
        if method == "Mean":
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
        elif method == "Median":
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        elif method == "Mode":
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        elif method == "DropRows":
            df_cleaned = df_cleaned.dropna(subset=[col])

    st.write("**Missing values after cleaning:**")
    st.write(df_cleaned.isnull().sum())
    st.subheader("Dataset After Handling Missing Values")
    st.dataframe(df_cleaned.head())
else:
    df_cleaned = df.copy()

st.subheader("Exploratory Data Analysis (EDA) & Relationships")

numeric_df = df_cleaned.select_dtypes(include=[np.number])

st.write("**Correlation Matrix (Numeric Features):**")
corr = numeric_df.corr()
st.dataframe(corr)


st.write("**Top Feature Correlations:**")
corr_pairs = corr.unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1]  # exclude self-correlation
st.write(corr_pairs.head(5))


if "Risk Level" in df_cleaned.columns:
    st.write("**Average Numeric Features by Risk Level:**")
    grouped = df_cleaned.groupby("Risk Level")[numeric_df.columns].mean()
    st.dataframe(grouped)





st.subheader("Encoding Categorical Features")
encoding_choice = st.sidebar.radio("Choose encoding method:", ["Label Encoding", "One-Hot Encoding"])
if st.sidebar.button("Run Encoding"):
    df_encoded = df_cleaned.copy()
    categorical_cols = df_encoded.select_dtypes(exclude=[np.number]).columns

    if encoding_choice == "Label Encoding":
        for col in categorical_cols:
            df_encoded[col] = df_encoded[col].astype("category").cat.codes
        st.write("**Applied Label Encoding.**")
    else:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols)
        st.write("**Applied One-Hot Encoding.**")

    st.subheader("Dataset After Encoding")
    st.dataframe(df_encoded.head())