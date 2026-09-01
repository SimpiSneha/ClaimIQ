"""
ClaimIQ frontend - Day 1 skeleton.
A simple form that submits a claim to the FastAPI backend.
Run with: streamlit run app.py
(Make sure the backend is already running at http://127.0.0.1:8000)
"""
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ClaimIQ", page_icon="🚗")
st.title("🚗 ClaimIQ — Submit a Claim")

with st.form("claim_form"):
    claimant_name = st.text_input("Full name")
    policy_number = st.text_input("Policy number")
    description = st.text_area("What happened?", placeholder="e.g. My car was hit while parked outside my house.")
    photo = st.file_uploader("Upload a photo of the damage (optional)", type=["jpg", "jpeg", "png"])

    submitted = st.form_submit_button("Submit Claim")

    if submitted:
        if not claimant_name or not policy_number or not description:
            st.error("Please fill in your name, policy number, and description.")
        else:
            files = {"photo": (photo.name, photo.getvalue())} if photo else None
            data = {
                "claimant_name": claimant_name,
                "policy_number": policy_number,
                "description": description,
            }
            try:
                response = requests.post(f"{API_URL}/submit-claim", data=data, files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Claim submitted! Your claim ID is **{result['claim_id']}**")
                else:
                    st.error(f"Something went wrong: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the backend. Make sure it's running (`uvicorn main:app --reload`).")

st.divider()
st.subheader("Check claim status")
claim_id_lookup = st.number_input("Enter your claim ID", min_value=1, step=1)
if st.button("Check status"):
    try:
        response = requests.get(f"{API_URL}/claim/{claim_id_lookup}")
        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error("Claim not found.")
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the backend.")

st.divider()
st.subheader("🤖 Process Claim (AI Decision)")
st.caption("Runs the full pipeline: policy coverage check, fraud risk scoring, and damage assessment.")

with st.form("process_claim_form"):
    process_claim_id = st.number_input("Claim ID to process", min_value=1, step=1, key="process_claim_id")

    st.write("**Fraud risk details** (defaults are used if left as-is)")
    past_claims = st.selectbox("Past number of claims", ["none", "1", "2 to 4", "more than 4"], key="past_claims")
    witness = st.selectbox("Was there a witness?", ["Yes", "No"], key="witness")
    police_report = st.selectbox("Was a police report filed?", ["Yes", "No"], key="police_report")
    days_since_claim = st.selectbox(
        "How soon after the policy started was this claimed?",
        ["more than 30", "15 to 30", "8 to 15", "1 to 7", "none"],
        key="days_since_claim",
    )
    address_change = st.selectbox(
        "Recent address change?",
        ["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"],
        key="address_change",
    )
    fault = st.selectbox("Who was at fault?", ["Policy Holder", "Third Party"], key="fault")
    vehicle_price = st.selectbox(
        "Vehicle price range",
        ["less than 20000", "20000 to 29000", "30000 to 39000", "40000 to 59000", "more than 69000"],
        key="vehicle_price",
    )
    claimant_age = st.number_input("Claimant age", min_value=16, max_value=100, value=30, key="claimant_age")

    process_submitted = st.form_submit_button("Process Claim with AI")

if process_submitted:
    try:
        data = {
            "claim_id": process_claim_id,
            "past_number_of_claims": past_claims,
            "witness_present": witness,
            "police_report_filed": police_report,
            "days_policy_claim": days_since_claim,
            "address_change_claim": address_change,
            "fault": fault,
            "vehicle_price": vehicle_price,
            "age": claimant_age,
        }
        st.write("Debug - values being sent:", data)  # temporary, remove once confirmed working
        with st.spinner("Running policy check, fraud scoring, and damage assessment..."):
            response = requests.post(f"{API_URL}/process-claim", data=data)

        if response.status_code == 200:
            result = response.json()
            decision = result["decision"]

            if "AUTO-APPROVE" in decision:
                st.success(f"**Decision:** {decision}")
            elif "DENY" in decision:
                st.error(f"**Decision:** {decision}")
            else:
                st.warning(f"**Decision:** {decision}")

            st.write("**Reasoning:**")
            for reason in result["reasoning"]:
                st.write(f"- {reason}")

            with st.expander("Full signal breakdown"):
                st.json(result["signals"])
        else:
            st.error(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the backend. Make sure it's running (`uvicorn main:app --reload`).")