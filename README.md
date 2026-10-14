# Customer Segmentation Web App

A beginner-friendly Python project that groups customers with K-Means clustering and displays the results in a Streamlit web app.

## What you will learn
- Load customer data from a CSV file with pandas.
- Select numeric features such as age, annual income, and spending score.
- Standardize features so variables on different scales are comparable.
- Use K-Means clustering to assign each customer to a segment.
- Explore segment sizes, a scatter plot, and average feature profiles.
- Export the resulting customer labels to CSV.

## Project structure
```text
customer_segmentation_project/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── customers.csv
```

## Run it locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended):
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Start the web app: `streamlit run app.py`
6. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## CSV format
The included sample data has:
- `CustomerID`: customer identifier (not used by default for clustering)
- `Gender`: demographic category
- `Age`: age in years
- `Annual Income (k$)`: annual income in thousands of dollars
- `Spending Score (1-100)`: illustrative spending score

Upload your own CSV in the sidebar. At least two numeric columns and more complete rows than the selected number of clusters are required. Select only meaningful numeric features; do not include ID numbers as clustering features.

## How it works
1. **Feature selection:** choose numeric variables that describe customer behavior or demographics.
2. **Cleaning:** rows with missing values in the selected features are excluded for that run.
3. **Scaling:** `StandardScaler` converts features to comparable scales.
4. **Clustering:** K-Means assigns each row to one of K groups by minimizing within-cluster distances.
5. **Evaluation:** the silhouette score is a rough measure of how separated the groups are; it is not a business success score.
6. **Interpretation:** average feature values help you describe the groups. Use domain knowledge before making decisions.

## Suggested one-day learning plan
- **Hour 1:** understand the dataset and the goal of segmentation.
- **Hours 2–3:** learn pandas, missing-value checks, and feature selection.
- **Hours 4–5:** understand scaling and K-Means; run the app.
- **Hours 6–7:** test different feature combinations and K values.
- **Hour 8:** interpret segment profiles, take screenshots, and write a short project summary.

## Important
The sample dataset is synthetic and is provided for learning/demo purposes. This project is not a production customer profiling system. Do not upload confidential customer information to a public repository.
