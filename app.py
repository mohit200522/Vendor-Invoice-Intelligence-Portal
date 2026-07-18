import streamlit as st
import pandas as pd

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)

# ==========================================================
# HEADER
# ==========================================================

st.title("📦 Vendor Invoice Intelligence Portal")

st.markdown("""
## AI-Powered Freight Cost Prediction & Invoice Risk Detection

This intelligent portal helps finance teams to:

- 🚚 Predict Freight Cost
- 🚨 Detect Risky Vendor Invoices
- 📊 Improve Finance Decision Making
""")

st.divider()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Model Selection")

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    (
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    )
)

st.sidebar.markdown("""
### Business Impact

✅ Improved Cost Forecasting

✅ Detect Risky Invoices

✅ Faster Finance Operations

✅ Reduce Manual Workload
""")
# ==========================================================
# FREIGHT COST PREDICTION
# ==========================================================

if selected_model == "Freight Cost Prediction":

    st.subheader("🚚 Freight Cost Prediction")

    st.write(
        "Enter the invoice details below to estimate the freight cost."
    )

    with st.form("freight_form"):

        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1200,
                step=1
            )

        with col2:
            dollars = st.number_input(
                "Invoice Dollars ($)",
                min_value=1.0,
                value=18500.0,
                step=100.0
            )

        predict_btn = st.form_submit_button("Predict Freight Cost")

    if predict_btn:

        input_data = {
            # Your trained regression model uses only Dollars
            "Dollars": [dollars]
        }

        prediction = predict_freight_cost(input_data)

        freight = float(prediction["Predicted_Freight"].iloc[0])

        st.success("Prediction Completed Successfully")

        st.metric(
            label="Estimated Freight Cost",
            value=f"${freight:.2f}"
        )

        st.info(
            f"""
            Quantity : {quantity}

            Invoice Amount : ${dollars:,.2f}

            Estimated Freight : ${freight:.2f}
            """
        )
# ==========================================================
# INVOICE FLAG PREDICTION
# ==========================================================

else:

    st.subheader("🚨 Invoice Manual Approval Prediction")

    st.write(
        "Predict whether an invoice requires manual approval."
    )

    with st.form("flag_form"):

        col1, col2 = st.columns(2)

        with col1:

            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=1,
                value=50
            )

            invoice_dollars = st.number_input(
                "Invoice Dollars ($)",
                min_value=1.0,
                value=352.95
            )

            freight = st.number_input(
                "Freight Cost ($)",
                min_value=0.0,
                value=1.73
            )

        with col2:

            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )

            total_item_dollars = st.number_input(
                "Total Item Dollars ($)",
                min_value=1.0,
                value=2476.00
            )

        predict_btn = st.form_submit_button(
            "Predict Approval Status"
        )

    if predict_btn:

        input_data = {

            "invoice_quantity": [invoice_quantity],

            "invoice_dollars": [invoice_dollars],

            "Freight": [freight],      # IMPORTANT: Capital F

            "total_item_quantity": [total_item_quantity],

            "total_item_dollars": [total_item_dollars]

        }

        try:

            prediction = predict_invoice_flag(input_data)

            result = prediction["Predicted_Approval"].iloc[0]

            if result == "Approved":

                st.success("✅ Invoice Approved")

            else:

                st.error("⚠ Manual Approval Required")

            st.dataframe(prediction)

        except Exception as e:

            st.error("Prediction Failed")

            st.exception(e)