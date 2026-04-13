
# Imports
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st

st.set_page_config(
    page_title="Buy Now, Pay Later Risk Intelligence Tool",
    layout="wide"
)

# Load model artifacts
ARTIFACT_DIR = Path("app_artifacts")

model = joblib.load(ARTIFACT_DIR / "xgb_model.joblib")
feature_cols = joblib.load(ARTIFACT_DIR / "feature_columns.joblib")

# Oreprocess artifacts
train_medians = {}
train_modes = {}
numeric_nan_cols = []
categorical_nan_cols = []

if (ARTIFACT_DIR / "train_medians.joblib").exists():
    train_medians = joblib.load(ARTIFACT_DIR / "train_medians.joblib")

if (ARTIFACT_DIR / "train_modes.joblib").exists():
    train_modes = joblib.load(ARTIFACT_DIR / "train_modes.joblib")

if (ARTIFACT_DIR / "numeric_nan_cols.joblib").exists():
    numeric_nan_cols = joblib.load(ARTIFACT_DIR / "numeric_nan_cols.joblib")

if (ARTIFACT_DIR / "categorical_nan_cols.joblib").exists():
    categorical_nan_cols = joblib.load(ARTIFACT_DIR / "categorical_nan_cols.joblib")

# Core utility functions
def clean_feature_names(columns: pd.Index) -> pd.Index:
    return (
        pd.Index(columns)
        .astype(str)
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )


def percent_range_to_midpoint(label: str) -> float:
    nums = [float(x) for x in re.findall(r"\d+", label)]
    if len(nums) >= 2:
        return round((nums[0] + nums[1]) / 2, 1)
    return float(nums[0]) if nums else 0.0


def rate_label_to_value(label: str) -> float:
    nums = re.findall(r"\d+", label)
    return float(nums[0]) if nums else 0.0


def fico_band_to_low_high(label: str) -> tuple[int, int]:
    mapping = {
        "Poor (580–619)": (580, 619),
        "Fair (620–659)": (620, 659),
        "Good (660–699)": (660, 699),
        "Very Good (700–739)": (700, 739),
        "Strong (740–779)": (740, 779),
        "Excellent (780–850)": (780, 850),
    }
    return mapping.get(label, (660, 699))


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def build_default_input() -> dict:
    return {
        "loan_amnt": 10000.0,
        "funded_amnt": 10000.0,
        "term": " 36 months",
        "int_rate": 12.0,
        "installment": 334.0,
        "grade": "B",
        "sub_grade": "B3",
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "annual_inc": 60000.0,
        "verification_status": "Verified",
        "issue_d": "Dec-2018",
        "purpose": "other",
        "addr_state": "CA",
        "dti": 17.0,
        "delinq_2yrs": 0.0,
        "fico_range_low": 660.0,
        "fico_range_high": 699.0,
        "open_acc": 8.0,
        "pub_rec": 0.0,
        "revol_bal": 8500.0,
        "revol_util": 42.0,
        "total_acc": 21.0,
        "mort_acc": 1.0,
        "pub_rec_bankruptcies": 0.0,
        "issue_date": "2018-12-01",
        "issue_year": 2018.0,
        "issue_month": 12.0,
        "cpi_inflation_proxy": 2.1,
        "consumer_credit_revolving": 1040.0,
        "consumer_loan_delinquency_rate": 2.3,
        "consumer_sentiment_index": 98.3,
        "credit_card_delinquency_rate": 2.4,
        "credit_card_interest_rate": 14.9,
        "personal_savings_rate": 6.8,
        "unemployment_rate": 3.9,
        "cfpb_complaint_count": 1500.0,
        "cfpb_complaint_count_state": 120.0,
    }


def validate_thresholds(review_threshold: float, escalate_threshold: float) -> bool:
    return escalate_threshold > review_threshold


def risk_band(prob: float) -> str:
    if prob < 0.20:
        return "Low"
    if prob < 0.40:
        return "Moderate"
    if prob < 0.60:
        return "Elevated"
    return "High"


def review_priority(prob: float, review_threshold: float, escalate_threshold: float) -> str:
    if prob < review_threshold:
        return "Standard processing"
    if prob < escalate_threshold:
        return "Manual review"
    return "Escalation"


def recommended_action(prob: float, review_threshold: float, escalate_threshold: float) -> str:
    if prob < review_threshold:
        return (
            "The application falls below the current manual review trigger and would generally "
            "remain in the standard processing path."
        )
    if prob < escalate_threshold:
        return (
            "The application falls between the manual review and escalation triggers, so an "
            "additional review step would be appropriate."
        )
    return (
        "The application exceeds the escalation trigger under the current settings and should be "
        "treated as a higher-priority review case."
    )


def prepare_input(input_dict: dict) -> pd.DataFrame:
    input_df = pd.DataFrame([input_dict])

    for col in numeric_nan_cols:
        if col in input_df.columns and col in train_medians:
            input_df[col] = input_df[col].fillna(train_medians[col])

    for col in categorical_nan_cols:
        if col in input_df.columns and col in train_modes:
            input_df[col] = input_df[col].fillna(train_modes[col])

    input_encoded = pd.get_dummies(input_df, drop_first=False)
    input_encoded.columns = clean_feature_names(input_encoded.columns)
    input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

    return input_encoded


def predict_default_risk(input_dict: dict) -> tuple[float, pd.DataFrame]:
    processed = prepare_input(input_dict)
    prob = float(model.predict_proba(processed)[0][1])
    return prob, processed

# Scenario modeling
def scenario_preset_values(preset_name: str, base_input: dict) -> dict:
    preset_map = {
        "Baseline": {
            "cpi_inflation_proxy": base_input["cpi_inflation_proxy"],
            "consumer_credit_revolving": base_input["consumer_credit_revolving"],
            "consumer_loan_delinquency_rate": base_input["consumer_loan_delinquency_rate"],
            "consumer_sentiment_index": base_input["consumer_sentiment_index"],
            "credit_card_delinquency_rate": base_input["credit_card_delinquency_rate"],
            "credit_card_interest_rate": base_input["credit_card_interest_rate"],
            "personal_savings_rate": base_input["personal_savings_rate"],
            "unemployment_rate": base_input["unemployment_rate"],
            "cfpb_complaint_count": base_input["cfpb_complaint_count"],
            "cfpb_complaint_count_state": base_input["cfpb_complaint_count_state"],
            "int_rate_stress": 0.0,
            "revol_util_stress": 0.0,
            "dti_stress": 0.0,
            "issue_d": base_input["issue_d"],
            "issue_date": base_input["issue_date"],
            "issue_year": base_input["issue_year"],
            "issue_month": base_input["issue_month"],
        },
        "Moderate stress": {
            "cpi_inflation_proxy": base_input["cpi_inflation_proxy"] + 1.0,
            "consumer_credit_revolving": base_input["consumer_credit_revolving"] + 35.0,
            "consumer_loan_delinquency_rate": base_input["consumer_loan_delinquency_rate"] + 0.5,
            "consumer_sentiment_index": max(0.0, base_input["consumer_sentiment_index"] - 8.0),
            "credit_card_delinquency_rate": base_input["credit_card_delinquency_rate"] + 0.5,
            "credit_card_interest_rate": base_input["credit_card_interest_rate"] + 1.0,
            "personal_savings_rate": max(0.0, base_input["personal_savings_rate"] - 0.8),
            "unemployment_rate": base_input["unemployment_rate"] + 1.5,
            "cfpb_complaint_count": base_input["cfpb_complaint_count"] * 1.25,
            "cfpb_complaint_count_state": base_input["cfpb_complaint_count_state"] * 1.25,
            "int_rate_stress": 1.0,
            "revol_util_stress": 5.0,
            "dti_stress": 2.0,
            "issue_d": base_input["issue_d"],
            "issue_date": base_input["issue_date"],
            "issue_year": base_input["issue_year"],
            "issue_month": base_input["issue_month"],
        },
        "Severe stress": {
            "cpi_inflation_proxy": base_input["cpi_inflation_proxy"] + 2.0,
            "consumer_credit_revolving": base_input["consumer_credit_revolving"] + 75.0,
            "consumer_loan_delinquency_rate": base_input["consumer_loan_delinquency_rate"] + 1.2,
            "consumer_sentiment_index": max(0.0, base_input["consumer_sentiment_index"] - 15.0),
            "credit_card_delinquency_rate": base_input["credit_card_delinquency_rate"] + 1.2,
            "credit_card_interest_rate": base_input["credit_card_interest_rate"] + 2.0,
            "personal_savings_rate": max(0.0, base_input["personal_savings_rate"] - 1.5),
            "unemployment_rate": base_input["unemployment_rate"] + 3.0,
            "cfpb_complaint_count": base_input["cfpb_complaint_count"] * 1.60,
            "cfpb_complaint_count_state": base_input["cfpb_complaint_count_state"] * 1.60,
            "int_rate_stress": 2.0,
            "revol_util_stress": 10.0,
            "dti_stress": 4.0,
            "issue_d": base_input["issue_d"],
            "issue_date": base_input["issue_date"],
            "issue_year": base_input["issue_year"],
            "issue_month": base_input["issue_month"],
        },
    }
    return preset_map[preset_name].copy()


def apply_custom_stress_adjustments(
    input_data: dict,
    unemployment_shift: float,
    inflation_shift: float,
    complaint_shift_pct: int,
    sentiment_shift: float,
    delinquency_shift: float,
    cc_delinquency_shift: float,
) -> dict:
    adjusted = input_data.copy()

    adjusted["unemployment_rate"] = max(0.0, adjusted["unemployment_rate"] + unemployment_shift)
    adjusted["cpi_inflation_proxy"] = adjusted["cpi_inflation_proxy"] + inflation_shift
    adjusted["consumer_sentiment_index"] = max(0.0, adjusted["consumer_sentiment_index"] + sentiment_shift)
    adjusted["consumer_loan_delinquency_rate"] = max(
        0.0, adjusted["consumer_loan_delinquency_rate"] + delinquency_shift
    )
    adjusted["credit_card_delinquency_rate"] = max(
        0.0, adjusted["credit_card_delinquency_rate"] + cc_delinquency_shift
    )
    adjusted["cfpb_complaint_count"] = max(
        0.0, adjusted["cfpb_complaint_count"] * (1 + complaint_shift_pct / 100)
    )
    adjusted["cfpb_complaint_count_state"] = max(
        0.0, adjusted["cfpb_complaint_count_state"] * (1 + complaint_shift_pct / 100)
    )

    adjusted["int_rate"] = max(0.0, adjusted["int_rate"] + adjusted.get("int_rate_stress", 0.0))
    adjusted["revol_util"] = min(
        150.0,
        max(0.0, adjusted["revol_util"] + adjusted.get("revol_util_stress", 0.0))
    )
    adjusted["dti"] = min(
        80.0,
        max(0.0, adjusted["dti"] + adjusted.get("dti_stress", 0.0))
    )

    return adjusted


def compare_review_outcomes(
    base_prob: float,
    other_prob: float,
    review_threshold: float,
    escalate_threshold: float,
):
    base_priority = review_priority(base_prob, review_threshold, escalate_threshold)
    other_priority = review_priority(other_prob, review_threshold, escalate_threshold)

    if base_priority != other_priority:
        st.error(
            f"The comparison changes the recommended path from **{base_priority}** to **{other_priority}**."
        )
    else:
        st.info(f"The score changes, but the recommended path remains **{base_priority}**.")


def interpret_delta(delta: float, label: str = "scenario"):
    if delta > 0.03:
        st.error(f"The {label} meaningfully increases estimated risk.")
    elif delta > 0:
        st.warning(f"The {label} increases estimated risk modestly.")
    elif delta < -0.03:
        st.success(f"The {label} meaningfully lowers estimated risk.")
    elif delta < 0:
        st.success(f"The {label} lowers estimated risk slightly.")
    else:
        st.info(f"The {label} produces little or no meaningful change in estimated risk.")

# Prediction display
@st.cache_resource
def get_shap_explainer():
    return shap.TreeExplainer(model)


def get_local_shap_dataframe(processed_row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    explainer = get_shap_explainer()

    try:
        shap_values = explainer.shap_values(processed_row)
    except Exception:
        explanation = explainer(processed_row)
        shap_values = explanation.values

    if isinstance(shap_values, list):
        shap_array = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    else:
        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            if shap_values.shape[2] > 1:
                shap_array = shap_values[0, :, 1]
            else:
                shap_array = shap_values[0, :, 0]
        elif shap_values.ndim == 2:
            shap_array = shap_values[0]
        elif shap_values.ndim == 1:
            shap_array = shap_values
        else:
            return pd.DataFrame()

    shap_df = pd.DataFrame(
        {
            "feature": processed_row.columns,
            "value": processed_row.iloc[0].values,
            "shap_value": shap_array,
        }
    )

    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = (
        shap_df[shap_df["abs_shap"] > 0]
        .sort_values("abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    shap_df["direction"] = np.where(
        shap_df["shap_value"] >= 0,
        "Increased estimated risk",
        "Reduced estimated risk",
    )

    return shap_df


def nice_feature_name(feature: str) -> str:
    feature = feature.replace("_", " ").strip()

    replacements = {
        "loan amnt": "loan amount",
        "annual inc": "annual income",
        "dti": "debt-to-income ratio",
        "int rate": "interest rate",
        "revol util": "revolving utilization",
        "fico range low": "lower FICO range",
        "fico range high": "upper FICO range",
        "emp length": "employment length",
        "home ownership": "home ownership",
        "verification status": "verification status",
        "cfpb complaint count": "national complaint volume",
        "cfpb complaint count state": "state complaint volume",
        "consumer sentiment index": "consumer sentiment",
        "consumer loan delinquency rate": "loan delinquency rate",
        "credit card delinquency rate": "card delinquency rate",
        "cpi inflation proxy": "inflation environment",
        "unemployment rate": "unemployment environment",
        "pub rec bankruptcies": "bankruptcy history",
        "mort acc": "mortgage accounts",
        "open acc": "open accounts",
        "total acc": "total accounts",
        "revol bal": "revolving balance",
        "delinq 2yrs": "recent delinquencies",
        "sub grade": "sub-grade",
        "term_36_months": "shorter repayment structure",
        "term_60_months": "longer repayment structure",
    }

    cleaned = replacements.get(feature, feature)
    cleaned = re.sub(r"\b\d+\b", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def build_plain_language_explanation(shap_df: pd.DataFrame) -> list[str]:
    if shap_df.empty:
        return ["Detailed explanation was not available for this application."]

    top_rows = shap_df.head(3).copy()
    top_rows["display_name"] = top_rows["feature"].apply(nice_feature_name)

    explanations = []
    for _, row in top_rows.iterrows():
        if row["shap_value"] >= 0:
            explanations.append(
                f"{row['display_name'].capitalize()} was associated with higher estimated risk for this borrower."
            )
        else:
            explanations.append(
                f"{row['display_name'].capitalize()} reduced the estimated risk for this application."
            )

    return explanations


def summarize_key_drivers(input_data: dict, scenario_included: bool) -> list[str]:
    notes = []

    if input_data["dti"] >= 25:
        notes.append("Debt-to-income ratio is elevated, which may indicate repayment strain.")
    if input_data["revol_util"] >= 60:
        notes.append("Revolving utilization is high, suggesting heavier existing balance usage.")
    if input_data["annual_inc"] <= 40000:
        notes.append("Annual income is relatively low, which may reduce repayment flexibility.")
    if input_data["grade"] in ["D", "E", "F", "G"]:
        notes.append("The credit grade indicates a weaker baseline borrower profile.")
    if input_data["fico_range_low"] < 660:
        notes.append("The lower FICO band suggests weaker baseline credit quality.")

    if scenario_included:
        if input_data["unemployment_rate"] >= 6:
            notes.append("The selected environment reflects a weaker labor market.")
        if input_data["consumer_sentiment_index"] <= 85:
            notes.append("Consumer sentiment is relatively weak in this scenario.")
        if input_data["consumer_loan_delinquency_rate"] >= 3:
            notes.append("Broader delinquency conditions are elevated in this scenario.")
        if input_data["cfpb_complaint_count_state"] >= 200:
            notes.append("Complaint activity is elevated in the selected environment.")

    if not notes:
        notes.append("No major borrower or scenario warning flags were triggered in this case.")

    return notes


def make_case_summary_df(input_data: dict, scenario_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Field": [
                "Loan Amount",
                "Annual Income",
                "Debt-to-Income Ratio",
                "Interest Rate",
                "Revolving Utilization",
                "FICO Range",
                "Credit Grade",
                "Repayment Structure",
                "Economic Scenario",
            ],
            "Value": [
                format_currency(input_data["loan_amnt"]),
                format_currency(input_data["annual_inc"]),
                f"{input_data['dti']:.1f}%",
                f"{input_data['int_rate']:.1f}%",
                f"{input_data['revol_util']:.1f}%",
                f"{int(input_data['fico_range_low'])}-{int(input_data['fico_range_high'])}",
                input_data["grade"],
                input_data["term"].strip(),
                scenario_name,
            ],
        }
    )


def show_prediction_block(
    prob: float,
    processed: pd.DataFrame,
    input_data: dict,
    review_threshold: float,
    escalate_threshold: float,
    scenario_name: str,
):
    st.subheader("Risk assessment")

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Default Risk", f"{prob:.1%}")
    c2.metric("Risk Band", risk_band(prob))
    c3.metric("Recommended Review Path", review_priority(prob, review_threshold, escalate_threshold))

    st.markdown("### Decision guidance")
    st.info(
        f"""
**Recommended path:** {review_priority(prob, review_threshold, escalate_threshold)}

{recommended_action(prob, review_threshold, escalate_threshold)}

"This output should be used to support screening and review decisions in the BNPL context rather than as a fully automated approval rule."
"""
    )

    borderline_margin = 0.05
    if abs(prob - review_threshold) <= borderline_margin:
        st.warning(
            "This result is close to the manual review trigger. A small change in borrower or economic conditions could change the recommendation."
        )
    elif abs(prob - escalate_threshold) <= borderline_margin:
        st.warning(
            "This result is close to the escalation trigger. Reviewing nearby scenarios would be reasonable before making a final decision."
        )

    shap_df = get_local_shap_dataframe(processed, top_n=8)

    st.markdown("### Model explanation")
    st.write(
        "The explanation below highlights the factors that had the strongest influence on this prediction for the current borrower profile."
    )

    for line in build_plain_language_explanation(shap_df):
        st.write(f"- {line}")

    with st.expander("Why this score was assigned", expanded=False):
        if not shap_df.empty:
            chart_df = shap_df.copy()
            chart_df["feature"] = chart_df["feature"].apply(nice_feature_name)
            chart_df = chart_df.set_index("feature")[["shap_value"]]
            st.bar_chart(chart_df)

            detail_df = shap_df[["feature", "value", "shap_value", "direction"]].copy()
            detail_df["feature"] = detail_df["feature"].apply(nice_feature_name)
            detail_df.columns = ["Feature", "Application Value", "SHAP Contribution", "Direction"]
            st.dataframe(detail_df, hide_index=True, use_container_width=True)
        else:
            st.write("Detailed SHAP output was not available for this case.")

    st.markdown("### Additional interpretation notes")
    for note in summarize_key_drivers(input_data, scenario_included=True):
        st.write(f"- {note}")

    with st.expander("Case summary", expanded=False):
        st.dataframe(
            make_case_summary_df(input_data, scenario_name),
            hide_index=True,
            use_container_width=True,
        )

# Input options
annual_income_options = list(range(10000, 210000, 10000))

dti_options = [
    "0-4% (Very low)",
    "5-9% (Low)",
    "10-14% (Manageable)",
    "15-19% (Moderate)",
    "20-24% (Rising)",
    "25-29% (Elevated)",
    "30-34% (High)",
    "35-39% (Very high)",
    "40-44% (Severe)",
    "45-49% (Severe)",
    "50-54% (Severe)",
    "55-59% (Severe)",
    "60-64% (Severe)",
]

interest_rate_options = [
    "1% (Very low)", "2% (Very low)", "3% (Very low)", "4% (Very low)",
    "5% (Low)", "6% (Low)", "7% (Low)", "8% (Low)",
    "9% (Moderate)", "10% (Moderate)", "11% (Moderate)", "12% (Moderate)",
    "13% (Moderately high)", "14% (Moderately high)", "15% (Moderately high)", "16% (Moderately high)",
    "17% (High)", "18% (High)", "19% (High)", "20% (High)",
    "21% (Very high)", "22% (Very high)", "23% (Very high)", "24% (Very high)",
    "25% (Severe)", "26% (Severe)", "27% (Severe)", "28% (Severe)",
    "29% (Severe)", "30% (Severe)", "31% (Severe)", "32% (Severe)",
    "33% (Severe)", "34% (Severe)", "35% (Severe)", "36% (Severe)",
    "37% (Severe)", "38% (Severe)", "39% (Severe)", "40% (Severe)",
]

revol_util_options = [
    "0-4% (Very low)", "5-9% (Low)", "10-14% (Low)", "15-19% (Moderate)",
    "20-24% (Moderate)", "25-29% (Building)", "30-34% (Elevated)", "35-39% (Elevated)",
    "40-44% (High)", "45-49% (High)", "50-54% (Very high)", "55-59% (Very high)",
    "60-64% (Very high)", "65-69% (Very high)", "70-74% (Severe)", "75-79% (Severe)",
    "80-84% (Severe)", "85-89% (Severe)", "90-94% (Maxed)", "95-99% (Maxed)",
    "100-104% (Maxed)",
]

fico_band_options = [
    "Poor (580–619)",
    "Fair (620–659)",
    "Good (660–699)",
    "Very Good (700–739)",
    "Strong (740–779)",
    "Excellent (780–850)",
]

grade_display_options = {
    "A - Lowest modeled credit risk": "A",
    "B - Strong credit profile": "B",
    "C - Moderate credit quality": "C",
    "D - Elevated credit risk": "D",
    "E - High credit risk": "E",
    "F - Very high credit risk": "F",
    "G - Highest modeled credit risk": "G",
}

purpose_display_options = {
    "Everyday purchase": "other",
    "Retail installment purchase": "other",
    "Large planned purchase": "major_purchase",
    "Home-related purchase": "home_improvement",
    "Existing balance payoff": "debt_consolidation",
    "Credit card payoff": "credit_card",
    "Other consumer purchase": "other",
}

state_options = [
    "CA", "NY", "TX", "FL", "IL", "NJ", "WA", "AZ", "NV", "CO",
    "NC", "GA", "VA", "PA", "OH", "MI", "MA", "OR", "UT", "MN"
]

# Sidebar
with st.sidebar:
    st.header("Decision settings")
    st.caption(
        "This tool is designed for borrower screening, manual review prioritization, and scenario-based risk analysis."
    )

    review_threshold = st.slider("Manual review trigger", 0.10, 0.80, 0.35, 0.01)
    escalate_threshold = st.slider("Escalation trigger", 0.20, 0.95, 0.50, 0.01)

    if not validate_thresholds(review_threshold, escalate_threshold):
        st.error("The escalation trigger must be higher than the manual review trigger.")

    st.caption("Lower thresholds flag more applications for review.")

    st.markdown("**How to read the recommendation**")
    st.caption(
        "Below the manual review trigger: standard processing\n\n"
        "Between triggers: manual review\n\n"
        "Above the escalation trigger: escalation"
    )

    st.markdown("**Responsible use**")
    st.caption(
        "The score is best used for relative risk ranking and review support rather than as a standalone approval rule."
    )

# App header
st.title("Buy Now, Pay Later Risk Intelligence Tool")

st.markdown(
    """
This dashboard applies the final project model to estimate borrower default risk in a way that supports practical decision-making.

The interface is organized around one primary workflow:

**Step 1:** Enter a borrower profile  
**Step 2:** Select an economic scenario  
**Step 3:** Review the risk estimate, recommendation, and explanation

The tool is intended to support screening, threshold calibration, and scenario testing rather than automate approval decisions.
"""
)

st.info(
    "This model was trained on Lending Club data rather than a true BNPL transaction dataset. "
    "For that reason, it should be interpreted as a structured approximation of BNPL-style lending risk rather than a production approval engine."
)

with st.expander("Evaluator quick start", expanded=False):
    st.markdown(
        """
1. Leave the default borrower profile in place and click **Run risk assessment**.  
2. Change the economic scenario from **Baseline** to **Moderate stress** or **Severe stress**.  
3. Review the estimated risk, recommended review path, and explanation details.  
4. Use the comparison section to test how borrower assumptions or stress conditions affect the result.
"""
    )

# Main workflow
st.header("Step 1. Borrower profile")

col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0, step=500.0)
    annual_inc = st.selectbox("Annual Income", options=annual_income_options, index=5)

    dti_label = st.selectbox(
        "Debt-to-Income Ratio (%)",
        options=dti_options,
        index=dti_options.index("15-19% (Moderate)")
    )
    dti = percent_range_to_midpoint(dti_label)

    int_rate_label = st.selectbox(
        "Interest Rate (%)",
        options=interest_rate_options,
        index=11
    )
    int_rate = rate_label_to_value(int_rate_label)

    revol_util_label = st.selectbox(
        "Revolving Utilization (%)",
        options=revol_util_options,
        index=revol_util_options.index("40-44% (High)")
    )
    revol_util = percent_range_to_midpoint(revol_util_label)

    fico_band = st.selectbox("FICO Range", options=fico_band_options, index=2)
    fico_range_low, fico_range_high = fico_band_to_low_high(fico_band)

with col2:
    term_label = st.selectbox(
        "Repayment Structure",
        [
            "Pay in 4 (6 weeks)",
            "Pay monthly (3 months)",
            "Pay monthly (6 months)",
            "Pay monthly (12 months)"
        ],
        index=0
    )
    model_term = " 36 months" if term_label in ["Pay in 4 (6 weeks)", "Pay monthly (3 months)"] else " 60 months"

    grade_label = st.selectbox("Credit Grade", options=list(grade_display_options.keys()), index=1)
    grade = grade_display_options[grade_label]

    emp_length = st.selectbox(
        "Employment Length",
        ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
        index=5
    )

    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "ANY"], index=0)
    verification_status = st.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"], index=0)

    purpose_label = st.selectbox("Purchase Type", options=list(purpose_display_options.keys()), index=1)
    purpose = purpose_display_options[purpose_label]

with st.expander("Advanced borrower and loan inputs", expanded=False):
    adv1, adv2 = st.columns(2)

    with adv1:
        funded_amnt = st.number_input("Funded Amount", min_value=0.0, value=10000.0, step=500.0)
        installment = st.number_input("Installment", min_value=0.0, value=334.0, step=10.0)
        sub_grade = st.text_input("Sub-grade", value="B3")
        addr_state = st.selectbox("State", options=state_options, index=0)

    with adv2:
        open_acc = st.number_input("Open Accounts", min_value=0.0, value=8.0, step=1.0)
        total_acc = st.number_input("Total Accounts", min_value=0.0, value=21.0, step=1.0)
        revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=8500.0, step=100.0)
        delinq_2yrs = st.number_input("Delinquencies in 2 Years", min_value=0.0, value=0.0, step=1.0)
        pub_rec = st.number_input("Public Records", min_value=0.0, value=0.0, step=1.0)
        mort_acc = st.number_input("Mortgage Accounts", min_value=0.0, value=1.0, step=1.0)
        pub_rec_bankruptcies = st.number_input("Bankruptcies on Public Record", min_value=0.0, value=0.0, step=1.0)

base_input_data = build_default_input()
base_input_data.update(
    {
        "loan_amnt": loan_amnt,
        "funded_amnt": funded_amnt,
        "term": model_term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "addr_state": addr_state,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "fico_range_low": fico_range_low,
        "fico_range_high": fico_range_high,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
    }
)

st.session_state["base_input_data"] = base_input_data

st.header("Step 2. Economic scenario")

preset_slider_defaults = {
    "Baseline": {
        "unemployment_shift": 0.0,
        "inflation_shift": 0.0,
        "complaint_shift_pct": 0,
        "sentiment_shift": 0.0,
        "delinquency_shift": 0.0,
        "cc_delinquency_shift": 0.0,
    },
    "Moderate stress": {
        "unemployment_shift": 1.5,
        "inflation_shift": 1.0,
        "complaint_shift_pct": 25,
        "sentiment_shift": -8.0,
        "delinquency_shift": 0.5,
        "cc_delinquency_shift": 0.5,
    },
    "Severe stress": {
        "unemployment_shift": 3.0,
        "inflation_shift": 2.0,
        "complaint_shift_pct": 60,
        "sentiment_shift": -15.0,
        "delinquency_shift": 1.2,
        "cc_delinquency_shift": 1.2,
    },
}

scenario_choice = st.radio(
    "Select environment",
    ["Baseline", "Moderate stress", "Severe stress"],
    horizontal=True,
    key="scenario_choice",
)

if "last_scenario_choice" not in st.session_state:
    st.session_state["last_scenario_choice"] = scenario_choice

for key, value in preset_slider_defaults[scenario_choice].items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state["last_scenario_choice"] != scenario_choice:
    for key, value in preset_slider_defaults[scenario_choice].items():
        st.session_state[key] = value
    st.session_state["last_scenario_choice"] = scenario_choice

scenario_values = scenario_preset_values(scenario_choice, base_input_data)

# Default issue date values (set internally)
HARD_CODED_ISSUE = {
    "issue_d": "Dec-2018",
    "issue_date": "2018-12-01",
    "issue_year": 2018,
    "issue_month": 12,
}

with st.expander("Customize economic assumptions", expanded=False):
    stress1, stress2 = st.columns(2)

    with stress1:
        unemployment_shift = st.slider(
            "Adjustment to unemployment rate",
            -3.0,
            5.0,
            step=0.1,
            key="unemployment_shift",
        )
        inflation_shift = st.slider(
            "Adjustment to inflation proxy",
            -3.0,
            5.0,
            step=0.1,
            key="inflation_shift",
        )
        complaint_shift_pct = st.slider(
            "Adjustment to complaint volume (%)",
            -50,
            200,
            step=10,
            key="complaint_shift_pct",
        )

    with stress2:
        sentiment_shift = st.slider(
            "Adjustment to consumer sentiment",
            -30.0,
            30.0,
            step=1.0,
            key="sentiment_shift",
        )
        delinquency_shift = st.slider(
            "Adjustment to loan delinquency rate",
            -2.0,
            4.0,
            step=0.1,
            key="delinquency_shift",
        )
        cc_delinquency_shift = st.slider(
            "Adjustment to card delinquency rate",
            -2.0,
            4.0,
            step=0.1,
            key="cc_delinquency_shift",
        )

st.caption("Changes will be reflected after clicking 'Run risk assessment' below.")

scenario_input_data = base_input_data.copy()
scenario_input_data.update(scenario_values)
scenario_input_data.update(HARD_CODED_ISSUE)

scenario_input_data = apply_custom_stress_adjustments(
    scenario_input_data,
    unemployment_shift=unemployment_shift,
    inflation_shift=inflation_shift,
    complaint_shift_pct=complaint_shift_pct,
    sentiment_shift=sentiment_shift,
    delinquency_shift=delinquency_shift,
    cc_delinquency_shift=cc_delinquency_shift,
)

baseline_input_data = base_input_data.copy()
baseline_input_data.update(scenario_preset_values("Baseline", base_input_data))
baseline_input_data.update(HARD_CODED_ISSUE)

baseline_input_data = apply_custom_stress_adjustments(
    baseline_input_data,
    unemployment_shift=0.0,
    inflation_shift=0.0,
    complaint_shift_pct=0,
    sentiment_shift=0.0,
    delinquency_shift=0.0,
    cc_delinquency_shift=0.0,
)

st.header("Step 3. Score and review")

if st.button("Run risk assessment", use_container_width=True):
    if validate_thresholds(review_threshold, escalate_threshold):
        with st.spinner("Scoring application..."):
            baseline_prob, baseline_processed = predict_default_risk(baseline_input_data)
            scored_prob, scored_processed = predict_default_risk(scenario_input_data)

        delta_prob = scored_prob - baseline_prob
        diff_mask = baseline_processed.iloc[0] != scored_processed.iloc[0]
        changed_features = baseline_processed.columns[diff_mask].tolist()
        changed_feature_count = len(changed_features)

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline Risk", f"{baseline_prob:.1%}")
        c2.metric("Selected Scenario Risk", f"{scored_prob:.1%}")
        c3.metric("Change vs Baseline", f"{delta_prob:+.1%}")

        c4, c5 = st.columns(2)
        c4.metric("Changed Model Inputs", changed_feature_count)
        c5.metric("Prediction Changed", "Yes" if abs(delta_prob) > 1e-10 else "No")

        tolerance = 1e-10

        if changed_feature_count == 0:
            st.error(
                "The processed model inputs are identical between baseline and the selected scenario. "
                "This means the scenario adjustments are not changing the encoded input seen by the model."
            )
        elif abs(delta_prob) <= tolerance:
            st.info(
                f"""
The selected **{scenario_choice}** scenario changed the model inputs, but it did not produce a measurable change in the predicted score for this borrower profile.

This is a valid outcome in a tree-based model. The scenario engine is applying stress conditions, but the trained model is currently relying more heavily on borrower-level characteristics for this case.
"""
            )
        elif delta_prob > 0:
            st.warning(
                "The selected scenario increases estimated risk relative to the baseline environment."
            )
        else:
            st.success(
                "The selected scenario lowers estimated risk relative to the baseline environment."
            )

        with st.expander("Scenario values used for scoring", expanded=False):
            scenario_summary_df = pd.DataFrame(
                {
                    "Variable": [
                        "Unemployment Rate",
                        "Inflation Proxy",
                        "Consumer Sentiment",
                        "Loan Delinquency Rate",
                        "Card Delinquency Rate",
                        "Complaint Count",
                        "State Complaint Count",
                        "Interest Rate",
                        "Revolving Utilization",
                        "Debt-to-Income Ratio",
                    ],
                    "Baseline": [
                        baseline_input_data["unemployment_rate"],
                        baseline_input_data["cpi_inflation_proxy"],
                        baseline_input_data["consumer_sentiment_index"],
                        baseline_input_data["consumer_loan_delinquency_rate"],
                        baseline_input_data["credit_card_delinquency_rate"],
                        baseline_input_data["cfpb_complaint_count"],
                        baseline_input_data["cfpb_complaint_count_state"],
                        baseline_input_data["int_rate"],
                        baseline_input_data["revol_util"],
                        baseline_input_data["dti"],
                    ],
                    scenario_choice: [
                        scenario_input_data["unemployment_rate"],
                        scenario_input_data["cpi_inflation_proxy"],
                        scenario_input_data["consumer_sentiment_index"],
                        scenario_input_data["consumer_loan_delinquency_rate"],
                        scenario_input_data["credit_card_delinquency_rate"],
                        scenario_input_data["cfpb_complaint_count"],
                        scenario_input_data["cfpb_complaint_count_state"],
                        scenario_input_data["int_rate"],
                        scenario_input_data["revol_util"],
                        scenario_input_data["dti"],
                    ],
                }
            )
            scenario_summary_df["Difference"] = (
                scenario_summary_df[scenario_choice] - scenario_summary_df["Baseline"]
            )
            scenario_summary_df["Baseline"] = scenario_summary_df["Baseline"].astype(float).round(1)
            scenario_summary_df[scenario_choice] = scenario_summary_df[scenario_choice].astype(float).round(1)
            scenario_summary_df["Difference"] = scenario_summary_df["Difference"].astype(float).round(1)

            st.dataframe(scenario_summary_df, use_container_width=True, hide_index=True)

        with st.expander("Changed encoded features seen by the model", expanded=False):
            if changed_feature_count == 0:
                st.write("No encoded feature values changed between baseline and the selected scenario.")
            else:
                changed_df = pd.DataFrame(
                    {
                        "Feature": changed_features,
                        "Baseline Value": baseline_processed.iloc[0][changed_features].values,
                        f"{scenario_choice} Value": scored_processed.iloc[0][changed_features].values,
                    }
                )
                st.dataframe(changed_df, use_container_width=True, hide_index=True)

        show_prediction_block(
            prob=scored_prob,
            processed=scored_processed,
            input_data=scenario_input_data,
            review_threshold=review_threshold,
            escalate_threshold=escalate_threshold,
            scenario_name=scenario_choice,
        )

# Comparison tools
st.markdown("---")
st.header("Scenario comparison")

if "base_input_data" not in st.session_state:
    st.warning("Please enter a borrower profile in Step 1 first to use the comparison tools.")
else:
    base_input_data = st.session_state["base_input_data"]

    with st.expander("A. Compare economic conditions", expanded=True):
        st.write(
            "In this view, the borrower profile stays fixed so you can see how the recommendation changes under different economic conditions."
        )

        compare_scenario = st.selectbox(
            "Comparison scenario",
            ["Baseline", "Moderate stress", "Severe stress"],
            index=1
        )

        baseline_case = base_input_data.copy()
        baseline_case.update(scenario_preset_values("Baseline", base_input_data))
        baseline_case = apply_custom_stress_adjustments(
            baseline_case,
            unemployment_shift=0.0,
            inflation_shift=0.0,
            complaint_shift_pct=0,
            sentiment_shift=0.0,
            delinquency_shift=0.0,
            cc_delinquency_shift=0.0,
        )

        comparison_case = base_input_data.copy()
        comparison_case.update(scenario_preset_values(compare_scenario, base_input_data))
        comparison_case = apply_custom_stress_adjustments(
            comparison_case,
            unemployment_shift=0.0,
            inflation_shift=0.0,
            complaint_shift_pct=0,
            sentiment_shift=0.0,
            delinquency_shift=0.0,
            cc_delinquency_shift=0.0,
        )

        if st.button("Compare economic scenarios", use_container_width=True):
            if validate_thresholds(review_threshold, escalate_threshold):
                base_prob, _ = predict_default_risk(baseline_case)
                other_prob, _ = predict_default_risk(comparison_case)
                delta = other_prob - base_prob

                c1, c2, c3 = st.columns(3)
                c1.metric("Baseline Risk", f"{round(base_prob * 100, 1)}%")
                c2.metric(f"{compare_scenario} Risk", f"{round(other_prob * 100, 1)}%")
                c3.metric("Change in Risk", f"{round(delta * 100, 1):+}%")

                compare_review_outcomes(base_prob, other_prob, review_threshold, escalate_threshold)
                interpret_delta(delta, label="economic scenario")

                summary_df = pd.DataFrame(
                    {
                        "Variable": [
                            "Unemployment Rate",
                            "Inflation Proxy",
                            "Consumer Sentiment",
                            "Loan Delinquency Rate",
                            "Card Delinquency Rate",
                            "Complaint Count",
                            "State Complaint Count",
                            "Interest Rate",
                            "Revolving Utilization",
                            "Debt-to-Income Ratio",
                        ],
                        "Baseline": [
                            baseline_case["unemployment_rate"],
                            baseline_case["cpi_inflation_proxy"],
                            baseline_case["consumer_sentiment_index"],
                            baseline_case["consumer_loan_delinquency_rate"],
                            baseline_case["credit_card_delinquency_rate"],
                            baseline_case["cfpb_complaint_count"],
                            baseline_case["cfpb_complaint_count_state"],
                            baseline_case["int_rate"],
                            baseline_case["revol_util"],
                            baseline_case["dti"],
                        ],
                        compare_scenario: [
                            comparison_case["unemployment_rate"],
                            comparison_case["cpi_inflation_proxy"],
                            comparison_case["consumer_sentiment_index"],
                            comparison_case["consumer_loan_delinquency_rate"],
                            comparison_case["credit_card_delinquency_rate"],
                            comparison_case["cfpb_complaint_count"],
                            comparison_case["cfpb_complaint_count_state"],
                            comparison_case["int_rate"],
                            comparison_case["revol_util"],
                            comparison_case["dti"],
                        ],
                    }
                )
                summary_df["Difference"] = summary_df[compare_scenario] - summary_df["Baseline"]

                for col in ["Baseline", compare_scenario, "Difference"]:
                    summary_df[col] = summary_df[col].astype(float).round(1)

                st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with st.expander("B. Compare borrower assumptions", expanded=False):
        st.write(
            "This section holds the economic environment constant and shows how the recommendation changes when borrower assumptions change."
        )

        cmp1, cmp2 = st.columns(2)

        with cmp1:
            alt_loan_amnt = st.number_input(
                "Alternative Loan Amount",
                min_value=0.0,
                value=loan_amnt,
                step=500.0
            )
            alt_annual_inc = st.selectbox(
                "Alternative Annual Income",
                options=annual_income_options,
                index=annual_income_options.index(annual_inc)
            )
            alt_dti_label = st.selectbox(
                "Alternative Debt-to-Income Ratio",
                options=dti_options,
                index=dti_options.index(dti_label)
            )
            alt_dti = percent_range_to_midpoint(alt_dti_label)

        with cmp2:
            alt_int_rate_label = st.selectbox(
                "Alternative Interest Rate",
                options=interest_rate_options,
                index=interest_rate_options.index(int_rate_label)
            )
            alt_int_rate = rate_label_to_value(alt_int_rate_label)

            alt_revol_util_label = st.selectbox(
                "Alternative Revolving Utilization",
                options=revol_util_options,
                index=revol_util_options.index(revol_util_label)
            )
            alt_revol_util = percent_range_to_midpoint(alt_revol_util_label)

            alt_fico_band = st.selectbox(
                "Alternative FICO Range",
                options=fico_band_options,
                index=fico_band_options.index(fico_band)
            )
            alt_fico_low, alt_fico_high = fico_band_to_low_high(alt_fico_band)

        current_case = scenario_input_data.copy()

        alternative_case = scenario_input_data.copy()
        alternative_case.update(
            {
                "loan_amnt": alt_loan_amnt,
                "annual_inc": alt_annual_inc,
                "dti": alt_dti,
                "int_rate": alt_int_rate,
                "revol_util": alt_revol_util,
                "fico_range_low": alt_fico_low,
                "fico_range_high": alt_fico_high,
            }
        )

        if st.button("Compare borrower scenarios", use_container_width=True):
            if validate_thresholds(review_threshold, escalate_threshold):
                base_prob, _ = predict_default_risk(current_case)
                alt_prob, _ = predict_default_risk(alternative_case)
                diff = alt_prob - base_prob

                c1, c2, c3 = st.columns(3)
                c1.metric("Current Risk", f"{round(base_prob * 100, 1)}%")
                c2.metric("Alternative Risk", f"{round(alt_prob * 100, 1)}%")
                c3.metric("Change in Risk", f"{round(diff * 100, 1):+}%")

                compare_review_outcomes(base_prob, alt_prob, review_threshold, escalate_threshold)
                interpret_delta(diff, label="borrower comparison")

                comparison_df = pd.DataFrame(
                    {
                        "Input": [
                            "Loan Amount",
                            "Annual Income",
                            "Debt-to-Income Ratio",
                            "Interest Rate",
                            "Revolving Utilization",
                            "FICO Range",
                        ],
                        "Current": [
                            format_currency(current_case["loan_amnt"]),
                            format_currency(current_case["annual_inc"]),
                            f"{current_case['dti']:.1f}%",
                            f"{current_case['int_rate']:.1f}%",
                            f"{current_case['revol_util']:.1f}%",
                            f"{int(current_case['fico_range_low'])}-{int(current_case['fico_range_high'])}",
                        ],
                        "Alternative": [
                            format_currency(alternative_case["loan_amnt"]),
                            format_currency(alternative_case["annual_inc"]),
                            f"{alternative_case['dti']:.1f}%",
                            f"{alternative_case['int_rate']:.1f}%",
                            f"{alternative_case['revol_util']:.1f}%",
                            f"{int(alternative_case['fico_range_low'])}-{int(alternative_case['fico_range_high'])}",
                        ],
                    }
                )
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Final guidance
st.markdown("---")

with st.expander("Limitations and responsible use", expanded=False):
    st.write(
        """
This model was trained on Lending Club data rather than a true BNPL transaction dataset, so it should be interpreted as a structured approximation of BNPL-style lending behavior. The dashboard is most useful for relative risk ranking, manual review prioritization, and scenario testing. It is not intended to replace human judgment or serve as a production approval engine without additional work on fairness assessment, monitoring, governance, and secure data handling.
"""
    )

with st.expander("Project context", expanded=False):
    st.write(
        """
The dashboard uses the final XGBoost model from the BNPL risk project. It combines borrower and loan characteristics with broader economic and complaint-based signals to support risk screening, threshold calibration, and scenario-based analysis.
"""
    )

with st.expander("AI use disclosure", expanded=False):
    st.write(
        """
ChatGPT was used in a limited capacity to assist with debugging and implementation support during development. The project team determined the modeling approach, interface structure, interpretation, and final design decisions.
"""
    )
