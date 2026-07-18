import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from modeling_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():
    # Project root directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Database path
    db_path = BASE_DIR / "data" / "inventory.db"

    # Models folder
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)

    # Check database
    print(f"Database Path : {db_path}")
    print(f"Database Exists : {db_path.exists()}")

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at:\n{db_path}"
        )

    # Load data
    df = load_vendor_invoice_data(str(db_path))

    # Prepare features
    X, y = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    results = [
        evaluate_model(lr_model, X_test, y_test, "Linear Regression"),
        evaluate_model(dt_model, X_test, y_test, "Decision Tree Regression"),
        evaluate_model(rf_model, X_test, y_test, "Random Forest Regression")
    ]

    # Select best model (lowest MAE)
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]

    models = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }

    best_model = models[best_model_name]

    # Save model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print("\n================================")
    print(f"Best Model : {best_model_name}")
    print(f"Model Saved At : {model_path}")
    print("================================")


if __name__ == "__main__":
    main()