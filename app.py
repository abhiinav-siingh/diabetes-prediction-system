
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Diabetes AI | Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111c30;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #60a5fa;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .info-card {
        background: #172943;
        border: 1px solid #29466a;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .small-text {
        color: #94a3b8;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background-color: #172943;
        border: 1px solid #29466a;
        padding: 15px;
        border-radius: 12px;
    }

    .stButton button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
    }

    h1, h2, h3 {
        color: #e2e8f0;
    }

    hr {
        border-color: #29466a;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv("diabetes.csv")

    required = [
        "Glucose",
        "BMI",
        "Insulin",
        "Age",
        "Outcome"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[required].copy()

    X = df[["Glucose", "BMI", "Insulin", "Age"]].copy()
    y = df["Outcome"].copy()

    # Treat zero measurements as missing
    for col in ["Glucose", "BMI", "Insulin"]:
        X[col] = X[col].replace(0, np.nan)

    return df, X, y


try:
    data, X, y = load_data()

except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()


# ==========================================
# 4. TRAIN MACHINE LEARNING MODELS
# ==========================================

@st.cache_resource
def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Logistic Regression
    logistic_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Random Forest
    forest_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    logistic_model.fit(X_train, y_train)
    forest_model.fit(X_train, y_train)

    predictions = {
        "Logistic Regression": logistic_model.predict(X_test),
        "Random Forest": forest_model.predict(X_test)
    }

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": forest_model
    }

    return models, X_test, y_test, predictions


try:
    models, X_test, y_test, predictions = train_models(X, y)

except Exception as e:
    st.error(f"Model training error: {e}")
    st.stop()


# ==========================================
# 5. CALCULATE MODEL METRICS
# ==========================================

metrics = {}

for name, pred in predictions.items():

    metrics[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, pred, zero_division=0
        )
    }

metrics_df = pd.DataFrame(metrics).T


# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.markdown("# 🩺 Diabetes AI")
st.sidebar.caption("Machine Learning Prediction System")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🔍 Diabetes Prediction",
        "📊 Data Visualization",
        "🧠 Model Comparison",
        "📈 Model Evaluation",
        "📁 Dataset Information",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "**AI Applications Lab**  \n"
    "Course Code: 26ECP-102"
)

st.sidebar.caption(
    "Educational machine learning project"
)


# ==========================================
# 7. DASHBOARD
# ==========================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">Diabetes Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered classification and data analytics dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Educational project only. Model predictions are not "
        "medical diagnoses."
    )

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(data))
    col2.metric("Input Features", 4)
    col3.metric("ML Models", 2)
    col4.metric("Test Records", len(y_test))

    st.markdown("---")

    st.header("Machine Learning Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Logistic Regression")

        st.metric(
            "Accuracy",
            f"{metrics['Logistic Regression']['Accuracy']*100:.2f}%"
        )

        st.metric(
            "Recall",
            f"{metrics['Logistic Regression']['Recall']*100:.2f}%"
        )

    with col2:
        st.subheader("Random Forest")

        st.metric(
            "Accuracy",
            f"{metrics['Random Forest']['Accuracy']*100:.2f}%"
        )

        st.metric(
            "Recall",
            f"{metrics['Random Forest']['Recall']*100:.2f}%"
        )

    st.markdown("---")

    st.header("Model Performance Chart")

    st.bar_chart(metrics_df)

    st.subheader("Dataset Outcome Distribution")

    outcome_counts = data["Outcome"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.pie(
        outcome_counts,
        labels=["No Diabetes", "Diabetes"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title("Distribution of Dataset Outcomes")

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 8. INTERACTIVE PREDICTION
# ==========================================

elif page == "🔍 Diabetes Prediction":

    st.title("🔍 Diabetes Prediction")

    st.write(
        "Enter numerical values to generate a prediction "
        "using the trained machine learning models."
    )

    st.info(
        "The output is an educational model classification, "
        "not a clinical assessment."
    )

    with st.form("prediction_form"):

        st.subheader("Input Information")

        col1, col2 = st.columns(2)

        with col1:

            glucose = st.number_input(
                "Glucose Level",
                min_value=1,
                max_value=300,
                value=120
            )

            bmi = st.number_input(
                "BMI",
                min_value=1.0,
                max_value=80.0,
                value=25.0
            )

        with col2:

            insulin = st.number_input(
                "Insulin Level",
                min_value=1,
                max_value=900,
                value=80
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30
            )

        model_choice = st.selectbox(
            "Select Machine Learning Model",
            [
                "Logistic Regression",
                "Random Forest"
            ]
        )

        submit = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    if submit:

        new_data = pd.DataFrame(
            [[glucose, bmi, insulin, age]],
            columns=["Glucose", "BMI", "Insulin", "Age"]
        )

        selected_model = models[model_choice]

        result = int(selected_model.predict(new_data)[0])

        probability = float(
            selected_model.predict_proba(new_data)[0][1]
        )

        st.markdown("---")
        st.subheader("Prediction Result")

        if result == 1:
            st.warning(
                "Model Output: Class 1 (Diabetes)"
            )
        else:
            st.success(
                "Model Output: Class 0 (No Diabetes)"
            )

        st.metric(
            "Estimated Model Probability of Class 1",
            f"{probability*100:.2f}%"
        )

        st.progress(probability)

        st.caption(
            "This probability is generated by the selected "
            "machine learning model. It is not a personal "
            "medical risk estimate or diagnosis."
        )

        # Save prediction in session state
        report = pd.DataFrame([{
            "Date and Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Glucose": glucose,
            "BMI": bmi,
            "Insulin": insulin,
            "Age": age,
            "Model": model_choice,
            "Predicted Class": result,
            "Class 1 Probability (%)": round(
                probability * 100, 2
            )
        }])

        st.session_state["latest_report"] = report

        st.markdown("---")

        st.subheader("Download Prediction Report")

        st.download_button(
            label="⬇️ Download Prediction Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="diabetes_prediction_report.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==========================================
# 9. DATA VISUALIZATION
# ==========================================

elif page == "📊 Data Visualization":

    st.title("📊 Interactive Data Visualization")

    st.write(
        "Explore the distribution of the four input features "
        "in the PIMA dataset."
    )

    feature = st.selectbox(
        "Select Feature",
        ["Glucose", "BMI", "Insulin", "Age"]
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        data=data,
        x=feature,
        hue="Outcome",
        bins=25,
        kde=True,
        element="step",
        ax=ax
    )

    ax.set_title(f"{feature} Distribution by Outcome")
    ax.set_xlabel(feature)
    ax.set_ylabel("Number of Records")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.subheader("Compare Feature Distributions")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    features = ["Glucose", "BMI", "Insulin", "Age"]

    for ax, feature_name in zip(axes.flatten(), features):

        sns.boxplot(
            data=data,
            x="Outcome",
            y=feature_name,
            ax=ax
        )

        ax.set_title(feature_name)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Diabetes", "Diabetes"])

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 10. MODEL COMPARISON
# ==========================================

elif page == "🧠 Model Comparison":

    st.title("🧠 Machine Learning Model Comparison")

    st.write(
        "Compare Logistic Regression and Random Forest "
        "using four classification metrics."
    )

    st.dataframe(
        (metrics_df * 100).round(2).rename_axis("Model"),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Performance Comparison")

    st.bar_chart(metrics_df)

    st.markdown("---")

    st.subheader("Best Model by Individual Metric")

    for metric_name in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        best_model = metrics_df[metric_name].idxmax()

        st.write(
            f"**{metric_name}:** {best_model} "
            f"({metrics_df.loc[best_model, metric_name]*100:.2f}%)"
        )

    st.caption(
        "The best model depends on the evaluation metric. "
        "Accuracy alone does not describe all aspects of performance."
    )


# ==========================================
# 11. MODEL EVALUATION
# ==========================================

elif page == "📈 Model Evaluation":

    st.title("📈 Model Evaluation")

    selected = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest"]
    )

    pred = predictions[selected]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics[selected]['Accuracy']*100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{metrics[selected]['Precision']*100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{metrics[selected]['Recall']*100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{metrics[selected]['F1 Score']*100:.2f}%"
    )

    st.markdown("---")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, pred)

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(selected)

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )


# ==========================================
# 12. DATASET INFORMATION
# ==========================================

elif page == "📁 Dataset Information":

    st.title("📁 Dataset Information")

    st.write("Dataset: PIMA Indians Diabetes Database")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(data))
    col2.metric("Total Columns", len(data.columns))
    col3.metric(
        "Diabetes Class 1 Records",
        int((y == 1).sum())
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        data.describe().round(2),
        use_container_width=True
    )

    st.subheader("Missing Values After Zero Handling")

    missing_values = X.isnull().sum()

    st.dataframe(
        missing_values.rename("Missing Values"),
        use_container_width=True
    )

    st.caption(
        "Missing measurements are imputed using training-set "
        "medians during model fitting."
    )


# ==========================================
# 13. ABOUT PROJECT
# ==========================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About This Project")

    st.subheader("Diabetes Prediction Using Machine Learning")

    st.write(
        "This project demonstrates how supervised machine "
        "learning classification algorithms can be used to "
        "predict the outcome labels in the PIMA Indians "
        "Diabetes Dataset."
    )

    st.markdown("### Project Details")

    st.markdown("""
    - **Subject:** AI Applications Lab
    - **Course Code:** 26ECP-102
    - **Problem Type:** Classification
    - **Dataset:** PIMA Indians Diabetes Database
    - **Target Variable:** Outcome
    - **Input Features:** Glucose, BMI, Insulin and Age
    """)

    st.markdown("---")

    st.markdown("### Machine Learning Algorithms")

    st.markdown("""
    **1. Logistic Regression**

    A classification algorithm that estimates the probability
    of a sample belonging to a particular class.

    **2. Random Forest**

    An ensemble classification algorithm that combines
    predictions from multiple decision trees.
    """)

    st.markdown("---")

    st.markdown("### Technologies Used")

    st.markdown("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Seaborn
    - Streamlit
    """)

    st.markdown("---")

    st.markdown("### Project Objective")

    st.write(
        "To understand data preprocessing, supervised "
        "classification, model training, performance "
        "evaluation and interactive visualization."
    )

    st.warning(
        "This project is for academic demonstration only. "
        "It must not be used for medical diagnosis or "
        "healthcare decisions."
    )


# ==========================================
# 14. FOOTER
# ==========================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "🩺 Diabetes Prediction System"
)

st.sidebar.caption(
    "AI Applications Lab | 26ECP-102"
)

st.sidebar.caption(
    "Educational Project | Version 2.0"
)
