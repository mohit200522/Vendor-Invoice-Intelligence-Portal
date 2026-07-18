import os
import joblib

from data_preprocessing import (
    load_invoice_data,
    split_data,
    scale_features,
    apply_labels
)

from modeling_evaluation import (
    train_random_forest,
    evaluate_classifier
)

# =====================================================
# Features & Target
# =====================================================

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    # Create models folder
    os.makedirs("invoice_flagging/models", exist_ok=True)

    print("=" * 60)
    print("Loading Invoice Dataset...")
    print("=" * 60)

    # Load data
    df = load_invoice_data()

    # Create target labels
    df = apply_labels(df)

    print(df[FEATURES + [TARGET]].head())

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        "invoice_flagging/models/scaler.pkl"
    )

    # Train model
    grid_search = train_random_forest(
        X_train_scaled,
        y_train
    )

    # Evaluate model
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )

    # Save trained model
    MODEL_PATH = "invoice_flagging/models/predict_flag_invoice.pkl"

    joblib.dump(
        grid_search.best_estimator_,
        MODEL_PATH
    )

    print("\n" + "=" * 60)
    print("Model Saved Successfully")
    print(f"Model : {MODEL_PATH}")
    print("Scaler: invoice_flagging/models/scaler.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()