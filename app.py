import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Page config 
st.set_page_config(page_title="House Price Analysis", layout="wide")

#  Load data & model 
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.jb")

df = load_data()
model = load_model()

MODEL_FEATURES = [
    'OverallQual', 'GrLivArea', 'GarageArea', '1stFlrSF',
    'FullBath', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces',
    'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', 'OpenPorchSF', 'LotArea',
    'CentralAir'
]

# ── Sidebar navigation 
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview & EDA", "Visualization", "Case Study", "ML Prediction"]
)

# 
# 1. OVERVIEW & EDA
# 
if page == "Overview & EDA":
    st.title("House Price Dataset — Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric Features", df.select_dtypes(include=np.number).shape[1])
    col4.metric("Categorical Features", df.select_dtypes(include="object").shape[1])

    st.markdown("---")

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T.round(2), use_container_width=True)

    st.markdown("---")

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.write("No missing values found.")
    else:
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)

    st.markdown("---")

    st.subheader("Data Types")
    dtype_df = pd.DataFrame({
        "Column": df.dtypes.index,
        "Type": df.dtypes.values.astype(str)
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown("---")

    st.subheader("Target Variable — SalePrice")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["SalePrice"].dropna(), bins=50, edgecolor="black", color="steelblue")
        ax.set_xlabel("Sale Price ($)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Sale Price")
        st.pyplot(fig)
        plt.close()
    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.log1p(df["SalePrice"].dropna()), bins=50, edgecolor="black", color="steelblue")
        ax.set_xlabel("Log(Sale Price)")
        ax.set_ylabel("Count")
        ax.set_title("Log-Transformed Sale Price")
        st.pyplot(fig)
        plt.close()

    price_stats = df["SalePrice"].describe()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"${price_stats['mean']:,.0f}")
    c2.metric("Median", f"${price_stats['50%']:,.0f}")
    c3.metric("Min", f"${price_stats['min']:,.0f}")
    c4.metric("Max", f"${price_stats['max']:,.0f}")


# 
# 2. VISUALIZATION
# 
elif page == "Visualization":
    st.title("Data Visualization")

    # ── Correlation heatmap ───
    st.subheader("Correlation Heatmap — Top Numeric Features")
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False)
    top_features = corr.head(15).index.tolist() + ["SalePrice"]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        num_df[top_features].corr(),
        annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, ax=ax
    )
    ax.set_title("Correlation Matrix (Top 15 Features vs SalePrice)")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Scatter plots ─────────
    st.subheader("Feature vs Sale Price")
    scatter_col = st.selectbox(
        "Select a numeric feature",
        options=[c for c in num_df.columns if c != "SalePrice"]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df[scatter_col], df["SalePrice"], alpha=0.4, s=15, color="steelblue")
    ax.set_xlabel(scatter_col)
    ax.set_ylabel("Sale Price ($)")
    ax.set_title(f"{scatter_col} vs Sale Price")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Box plots ─────────────
    st.subheader("Sale Price by Category")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    box_col = st.selectbox("Select a categorical feature", options=cat_cols)
    top_cats = df[box_col].value_counts().head(10).index
    plot_df = df[df[box_col].isin(top_cats)]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.boxplot(column="SalePrice", by=box_col, ax=ax, grid=False)
    ax.set_xlabel(box_col)
    ax.set_ylabel("Sale Price ($)")
    ax.set_title(f"Sale Price by {box_col} (Top 10 categories)")
    plt.suptitle("")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Year built trend ──────
    st.subheader("Average Sale Price by Year Built")
    year_price = df.groupby("YearBuilt")["SalePrice"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(year_price["YearBuilt"], year_price["SalePrice"], linewidth=1.5, color="steelblue")
    ax.set_xlabel("Year Built")
    ax.set_ylabel("Average Sale Price ($)")
    ax.set_title("Average Sale Price by Year Built")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Neighborhood bar chart 
    st.subheader("Average Sale Price by Neighborhood")
    nbr = df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(nbr.index, nbr.values, color="steelblue", edgecolor="black")
    ax.set_xlabel("Neighborhood")
    ax.set_ylabel("Average Sale Price ($)")
    ax.set_title("Average Sale Price by Neighborhood")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Overall quality ───────
    st.subheader("Sale Price by Overall Quality")
    qual_price = df.groupby("OverallQual")["SalePrice"].median().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(qual_price["OverallQual"].astype(str), qual_price["SalePrice"],
           color="steelblue", edgecolor="black")
    ax.set_xlabel("Overall Quality (1-10)")
    ax.set_ylabel("Median Sale Price ($)")
    ax.set_title("Median Sale Price by Overall Quality")
    st.pyplot(fig)
    plt.close()


# 
# 3. CASE STUDY
# 
elif page == "Case Study":
    st.title("Case Study — House Price Prediction")

    st.subheader("Problem Statement")
    st.write(
        "The goal of this project is to predict the final sale price of residential homes "
        "in Ames, Iowa, based on 79 explanatory variables describing almost every aspect of "
        "the property. Accurate price prediction helps buyers, sellers, and real estate agents "
        "make informed decisions."
    )

    st.markdown("---")

    st.subheader("Dataset Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Source:** Ames Housing Dataset (Kaggle)")
        st.markdown(f"**Total Records:** {len(df):,}")
        st.markdown(f"**Total Features:** {df.shape[1] - 1}")
        st.markdown("**Target Variable:** SalePrice")
        st.markdown("**Task Type:** Supervised Regression")
    with col2:
        st.markdown("**Numeric Features:** " + str(df.select_dtypes(include=np.number).shape[1] - 1))
        st.markdown("**Categorical Features:** " + str(df.select_dtypes(include="object").shape[1]))
        st.markdown(f"**Price Range:** ${df['SalePrice'].min():,} — ${df['SalePrice'].max():,}")
        st.markdown(f"**Mean Price:** ${df['SalePrice'].mean():,.0f}")
        st.markdown(f"**Median Price:** ${df['SalePrice'].median():,.0f}")

    st.markdown("---")

    st.subheader("Key Findings from EDA")

    findings = {
        "Overall Quality": "Strongest single predictor. Median price nearly doubles from quality 5 to quality 9.",
        "Living Area (GrLivArea)": "Strong positive linear relationship with sale price. Larger homes command higher prices.",
        "Neighborhood": "Location significantly impacts price. NridgHt and NoRidge neighborhoods have the highest average prices.",
        "Year Built": "Newer homes tend to sell for more, with a notable price increase post-2000.",
        "Garage Area": "Homes with larger garages consistently sell at higher prices.",
        "Basement Finish": "Finished basements (GLQ type) add significant value compared to unfinished ones.",
        "Central Air": "Homes with central air conditioning sell for approximately 25% more on average.",
        "Fireplaces": "Each additional fireplace correlates with a meaningful price increase.",
    }

    for feature, insight in findings.items():
        st.markdown(f"- **{feature}:** {insight}")

    st.markdown("---")

    st.subheader("Top Correlated Features with Sale Price")
    num_df = df.select_dtypes(include=np.number)
    corr_vals = num_df.corr()["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(corr_vals.index[::-1], corr_vals.values[::-1], color="steelblue", edgecolor="black")
    ax.set_xlabel("Absolute Correlation with SalePrice")
    ax.set_title("Top 10 Features Correlated with Sale Price")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    st.subheader("Modeling Approach")
    st.markdown("""
**Algorithm:** XGBoost Regressor

XGBoost (Extreme Gradient Boosting) was selected for the following reasons:
- Handles missing values natively
- Robust to outliers compared to linear models
- Captures non-linear relationships between features and price
- Provides feature importance rankings
- Consistently achieves top performance on tabular regression tasks

**Feature Engineering Applied:**
- Encoded `CentralAir` (Y/N) as binary (1/0)
- Used 15 key features selected based on correlation analysis and domain knowledge
- Missing values in numeric features filled with column medians during preprocessing
    """)

    st.markdown("---")

    st.subheader("Business Impact")
    st.markdown("""
- **Buyers** can assess whether a listed price is fair relative to comparable properties.
- **Sellers** can identify which home improvements yield the highest return on investment.
- **Real estate agents** can provide data-driven pricing recommendations.
- **Investors** can identify undervalued properties in specific neighborhoods.
    """)

    st.markdown("---")

    st.subheader("Price Segment Analysis")
    bins = [0, 100000, 150000, 200000, 250000, 300000, 500000, 1000000]
    labels = ["<100K", "100-150K", "150-200K", "200-250K", "250-300K", "300-500K", ">500K"]
    df["PriceSegment"] = pd.cut(df["SalePrice"], bins=bins, labels=labels)
    segment_counts = df["PriceSegment"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(segment_counts.index.astype(str), segment_counts.values,
           color="steelblue", edgecolor="black")
    ax.set_xlabel("Price Segment")
    ax.set_ylabel("Number of Houses")
    ax.set_title("Distribution of Houses by Price Segment")
    st.pyplot(fig)
    plt.close()

    segment_df = pd.DataFrame({
        "Price Segment": segment_counts.index.astype(str),
        "Count": segment_counts.values,
        "Percentage": (segment_counts.values / len(df) * 100).round(1)
    })
    st.dataframe(segment_df, use_container_width=True)


# 
# 4. ML PREDICTION
# 
elif page == "ML Prediction":
    st.title("Machine Learning — House Price Prediction")

    tab1, tab2 = st.tabs(["Model Performance", "Predict a House"])

    # ── Tab 1: Model Performance 
    with tab1:
        st.subheader("Model Evaluation on Dataset")

        eval_df = df[MODEL_FEATURES + ["SalePrice"]].copy()
        eval_df["MasVnrArea"] = eval_df["MasVnrArea"].fillna(0)
        eval_df["LotFrontage"] = eval_df["LotFrontage"].fillna(eval_df["LotFrontage"].median())
        eval_df["BsmtFinSF1"] = eval_df["BsmtFinSF1"].fillna(0)
        eval_df["GarageArea"] = eval_df["GarageArea"].fillna(0)
        eval_df["CentralAir"] = eval_df["CentralAir"].map({"Y": 1, "N": 0}).fillna(1)
        eval_df = eval_df.dropna()

        X = eval_df[MODEL_FEATURES]
        y = eval_df["SalePrice"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score", f"{r2:.4f}")
        c2.metric("MAE", f"${mae:,.0f}")
        c3.metric("RMSE", f"${rmse:,.0f}")
        c4.metric("MAPE", f"{mape:.2f}%")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Actual vs Predicted**")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, y_pred, alpha=0.4, s=15, color="steelblue")
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
            ax.set_xlabel("Actual Price ($)")
            ax.set_ylabel("Predicted Price ($)")
            ax.set_title("Actual vs Predicted Sale Price")
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("**Residuals Distribution**")
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(residuals, bins=50, edgecolor="black", color="steelblue")
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
            ax.set_xlabel("Residual ($)")
            ax.set_ylabel("Count")
            ax.set_title("Residuals Distribution")
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        st.subheader("Feature Importance")
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": MODEL_FEATURES,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(
                importance_df["Feature"][::-1],
                importance_df["Importance"][::-1],
                color="steelblue", edgecolor="black"
            )
            ax.set_xlabel("Importance Score")
            ax.set_title("XGBoost Feature Importance")
            st.pyplot(fig)
            plt.close()

            st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)

    # ── Tab 2: Predict a House 
    with tab2:
        st.subheader("Enter House Details to Predict Sale Price")

        col1, col2, col3 = st.columns(3)

        with col1:
            overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
            gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", value=1500, step=50)
            garage_area = st.number_input("Garage Area (sq ft)", value=400, step=10)
            year_built = st.number_input("Year Built", value=1990, step=1, min_value=1800, max_value=2025)
            year_remod = st.number_input("Year Remodeled", value=1990, step=1, min_value=1800, max_value=2025)

        with col2:
            first_flr_sf = st.number_input("1st Floor Area (sq ft)", value=1000, step=50)
            full_bath = st.number_input("Full Bathrooms", value=2, step=1, min_value=0, max_value=10)
            fireplaces = st.number_input("Fireplaces", value=0, step=1, min_value=0, max_value=5)
            bsmt_fin_sf1 = st.number_input("Finished Basement Area (sq ft)", value=0, step=50)
            mas_vnr_area = st.number_input("Masonry Veneer Area (sq ft)", value=0, step=10)

        with col3:
            lot_frontage = st.number_input("Lot Frontage (ft)", value=70, step=1)
            lot_area = st.number_input("Lot Area (sq ft)", value=8000, step=100)
            wood_deck_sf = st.number_input("Wood Deck Area (sq ft)", value=0, step=10)
            open_porch_sf = st.number_input("Open Porch Area (sq ft)", value=0, step=10)
            central_air = st.selectbox("Central Air Conditioning", options=["Yes", "No"])

        if st.button("Predict Sale Price"):
            input_data = {
                "OverallQual": overall_qual,
                "GrLivArea": gr_liv_area,
                "GarageArea": garage_area,
                "1stFlrSF": first_flr_sf,
                "FullBath": full_bath,
                "YearBuilt": year_built,
                "YearRemodAdd": year_remod,
                "MasVnrArea": mas_vnr_area,
                "Fireplaces": fireplaces,
                "BsmtFinSF1": bsmt_fin_sf1,
                "LotFrontage": lot_frontage,
                "WoodDeckSF": wood_deck_sf,
                "OpenPorchSF": open_porch_sf,
                "LotArea": lot_area,
                "CentralAir": 1 if central_air == "Yes" else 0,
            }

            input_df = pd.DataFrame([input_data], columns=MODEL_FEATURES)
            prediction = model.predict(input_df)[0]

            st.markdown("---")
            st.success(f"Predicted Sale Price: **${prediction:,.2f}**")

            price_range_low = prediction * 0.90
            price_range_high = prediction * 1.10
            st.info(f"Estimated price range (±10%): ${price_range_low:,.0f} — ${price_range_high:,.0f}")

            # Show where this price falls in the dataset
            percentile = (df["SalePrice"] < prediction).mean() * 100
            st.write(f"This predicted price is higher than **{percentile:.1f}%** of homes in the dataset.")