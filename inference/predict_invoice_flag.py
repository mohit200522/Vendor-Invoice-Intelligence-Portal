import joblib
import pandas as pd
from pathlib import Path

# ======================================================
# Project Root
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# ======================================================
# Model & Scaler Paths
# ======================================================
MODEL_PATH = BASE_DIR / "invoice_flagging" / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "invoice_flagging" / "models" / "scaler.pkl"


def load_model():
    """
    Load trained classifier model and scaler.
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def predict_invoice_flag(input_data):
    """
    Predict approval flag for new invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    DataFrame with predictions
    """

    model, scaler = load_model()

    input_df = pd.DataFrame(input_data)

    # Scale input
    X = scaler.transform(input_df)

    # Prediction
    predictions = model.predict(X)

    input_df["Predicted_Approval"] = predictions

    # Convert numeric prediction into readable labels
    input_df["Predicted_Approval"] = input_df["Predicted_Approval"].map({
        0: "Rejected",
        1: "Approved"
    })

    return input_df


if __name__ == "__main__":

    sample_data = {
        "VendorNumber": [105, 480, 516],
        "Quantity": [6, 10100, 1935],
        "Dollars": [214.26, 137483.78, 15527.25],
        "Freight": [3.47, 2935.20, 429.20]
    }

    result = predict_invoice_flag(sample_data)

    print("\nInvoice Approval Prediction")
    print("=" * 50)
    print(result)