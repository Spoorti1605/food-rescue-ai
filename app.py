import streamlit as st
from datetime import datetime, timedelta
import os

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

from main import app as workflow

st.set_page_config(
    page_title="Smart Food Rescue System",
    page_icon="🍱",
    layout="wide"p
)

st.title("🍱 Smart Food Rescue & Redistribution")
st.subheader("Agentic AI-Based Food Donation Management System")

st.write(
    "Connect food donors with suitable NGOs through "
    "an intelligent five-agent workflow."
)

st.divider()

# Sidebar
st.sidebar.title("About the System")
st.sidebar.write(
    "This prototype uses five intelligent agents "
    "coordinated through LangGraph."
)

st.sidebar.markdown("""
**Our five agents:**

1. Food Collection Agent
2. Food Safety Agent
3. NGO Matching Agent
4. Logistics Agent
5. Notification & Tracking Agent
""")

st.sidebar.divider()

st.sidebar.caption(
    "Demonstration prototype. NGO records, "
    "notifications and delivery estimates are simulated."
)

# Donation form
st.header("Submit a Food Donation")

now = datetime.now()

with st.form("donation_form"):

    st.subheader("Donor Information")

    provider = st.text_input(
        "Restaurant / Donor Name",
        placeholder="Enter restaurant name"
    )

    pickup_location = st.text_input(
        "Pickup Location",
        value="Hubballi"
    )

    st.subheader("Food Information")

    col1, col2 = st.columns(2)

    with col1:
        food_type = st.selectbox(
            "Food Type",
            ["rice", "bread", "vegetables", "fruits", "seafood", "other"]
        )

        quantity = st.number_input(
            "Quantity (kg)",
            min_value=-1000.0,
            max_value=10000.0,
            value=10.0,
            step=1.0
        )

    with col2:
        condition = st.selectbox(
            "Reported Food Condition",
            ["fresh", "spoiled", "unknown", "uncertain"]
        )

        st.caption(
            "The AI performs preliminary screening only. "
            "Human inspection is still required."
        )

    st.subheader("Preparation and Expiry Details")

    col3, col4 = st.columns(2)

    with col3:
        preparation_date = st.date_input(
            "Preparation Date",
            value=(now - timedelta(hours=1)).date()
        )

        preparation_time = st.time_input(
            "Preparation Time",
            value=(now - timedelta(hours=1)).time()
        )

    with col4:
        expiry_date = st.date_input(
            "Expiry Date",
            value=(now + timedelta(hours=6)).date()
        )

        expiry_time = st.time_input(
            "Expiry Time",
            value=(now + timedelta(hours=6)).time()
        )

    st.subheader("Pickup Coordinates")

    col5, col6 = st.columns(2)

    with col5:
        latitude = st.number_input(
            "Latitude",
            value=15.3647,
            format="%.6f"
        )

    with col6:
        longitude = st.number_input(
            "Longitude",
            value=75.1240,
            format="%.6f"
        )

    submitted = st.form_submit_button(
        "Submit Donation and Start AI Workflow",
        use_container_width=True
    )


# Process donation
if submitted:

    if not provider.strip() or not pickup_location.strip():

        st.error("Please enter the donor name and pickup location.")

    else:

        donation_data = {
            "provider": provider.strip(),
            "food_type": food_type,
            "quantity_kg": quantity,
            "condition": condition,
            "preparation_time": datetime.combine(
                preparation_date, preparation_time
            ).isoformat(),
            "expiry_time": datetime.combine(
                expiry_date, expiry_time
            ).isoformat(),
            "pickup_location": pickup_location,
            "latitude": latitude,
            "longitude": longitude,
            "simulate_api_failure": False
        }

        with st.spinner("Five agents are processing your donation..."):

            try:

                result = workflow.invoke(donation_data)

                st.divider()
                st.header("Donation Processing Results")

                # Agent 1
                st.subheader("1. Food Collection Agent")

                if result.get("collection_status") == "VALID":
                    st.success("Donation information validated.")
                else:
                    st.error("Donation information rejected.")

                st.write(result.get("collection_message", ""))

                # Agent 2
                st.subheader("2. Food Safety Agent")

                safety_status = result.get(
                    "safety_status", "NOT RUN"
                )

                if safety_status == "PRELIMINARILY_ELIGIBLE":

                    st.success(
                        "Preliminary screening: Reported as fresh"
                    )

                    st.warning(
                        "Physical human inspection is still required "
                        "before distribution."
                    )

                elif safety_status == "REJECTED":

                    st.error("Food donation rejected.")

                elif safety_status == "MANUAL_INSPECTION":

                    st.warning("Manual human inspection required.")

                st.write(result.get("safety_message", ""))

                # Agent 3
                st.subheader("3. NGO Matching Agent")

                matched_ngo = result.get("matched_ngo")

                if matched_ngo:

                    st.success(f"Matched NGO: {matched_ngo}")

                    st.write(
                        "Location:",
                        result.get("ngo_location")
                    )

                    st.write(
                        "Estimated straight-line distance:",
                        result.get("distance_km"),
                        "km"
                    )

                elif safety_status == "PRELIMINARILY_ELIGIBLE":

                    st.warning("No suitable NGO found.")

                else:

                    st.info(
                        "NGO matching was not performed because "
                        "the donation did not pass the previous stage."
                    )

                st.write(
                    result.get("ngo_match_message", "")
                )

                # Agent 4
                st.subheader("4. Logistics Agent")

                if matched_ngo:

                    st.write(
                        "Estimated delivery time:",
                        result.get("estimated_delivery_hours", 0),
                        "hours"
                    )

                    st.write(
                        "Urgency:",
                        result.get("urgency", "NOT CALCULATED")
                    )

                    st.caption(
                        result.get("logistics_message", "")
                    )

                else:

                    st.info("Logistics planning was not performed.")

                # Agent 5
                st.subheader("5. Notification & Tracking Agent")

                tracking = result.get(
                    "tracking_status", "NOT AVAILABLE"
                )

                st.write("Donation Tracking Status:", tracking)

                st.info(
                    result.get("notification_message", "")
                )

                # Final summary
                st.divider()

                st.header("Final Donation Summary")

                col7, col8 = st.columns(2)

                with col7:

                    st.metric(
                        "Collection Status",
                        result.get("collection_status", "NOT RUN")
                    )

                    st.metric(
                        "Safety Status",
                        safety_status
                    )

                with col8:

                    st.metric(
                        "Matched NGO",
                        matched_ngo or "None"
                    )

                    st.metric(
                        "Tracking Status",
                        tracking
                    )

                st.caption(
                    "This is a simulated demonstration. "
                    "No actual food pickup, delivery or notification "
                    "has occurred."
                )

            except Exception as error:

                st.error(
                    "The workflow encountered an unexpected error."
                )

                st.write(str(error))