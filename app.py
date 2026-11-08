
# CSC525: Principles of Machine Learning
# Portfolio Project: Proposal Intake Assistant
# Streamlit Chatbot Application
# Belinda Duncan
# November 8, 2026

# Streamlit application = Connects the chatbot interface, trained ML classifier, NLP extractor, and preliminary proposal generator.

# _____________________________________________________________________________________
# IMPORT REQUIRED LIBRARIES

import streamlit as st
import joblib
from pathlib import Path

from NLP_Processing.extract_requirements import extract_requirements
from Proposal_Generation.proposal_summary import generate_proposal_summary

# _____________________________________________________________________________________
# CONFIGURE STREAMLIT
# Streamlit provides a web interface without requiring HTML, CSS, or JavaScript.

st.set_page_config(
    page_title="Proposal Intake Assistant",
    page_icon="🤖",
    layout="centered"
)

# _____________________________________________________________________________________
# LOAD THE TRAINED MACHINE LEARNING MODEL

# The saved Logistic Regression model and TF-IDF vectorizer are loaded instead of retraining the model each time the app runs.
# The same vectorizer used during training is needed to convert new customer text into matching numerical features.
# cache_resource keeps the loaded model and vectorizer available across Streamlit reruns.

project_folder = Path(__file__).resolve().parent
model_folder = project_folder / "Model_Training" / "saved_models"


@st.cache_resource
def load_classifier():
    """Load the saved classifier and TF-IDF vectorizer."""

    model = joblib.load(
        model_folder / "logistic_regression_model.joblib"
    )

    vectorizer = joblib.load(
        model_folder / "tfidf_vectorizer.joblib"
    )

    return model, vectorizer


classifier, vectorizer = load_classifier()


# _____________________________________________________________________________________
# DISPLAY APPLICATION TITLE

st.title("Proposal Intake Assistant")

st.write(
    "Describe your industrial automation project, "
    "and I'll help identify requirements and "
    "information needed for a preliminary proposal."
)


# _____________________________________________________________________________________
# INITIALIZE SESSION STATE

# Streamlit reruns the script after user interactions. Session state keeps previously collected requirements and chat messages from being lost.

if "project_requirements" not in st.session_state:
    st.session_state.project_requirements = {
        "equipment": [],
        "communication_protocols": [],
        "data_destination": None
    }

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! Tell me about your project "
                "and what you're trying to accomplish."
            )
        }
    ]

# _____________________________________________________________________________________
# DISPLAY CHAT HISTORY

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# _____________________________________________________________________________________
# ACCEPT AND PROCESS CUSTOMER INPUT

customer_input = st.chat_input("Describe your project...")

if customer_input:

    # Save the customer's message so it remains in the chat history.
    st.session_state.messages.append({
        "role": "user",
        "content": customer_input
    })

    with st.chat_message("user"):
        st.write(customer_input)

    # Extract equipment, quantities, protocols, and destinations using spaCy and the technical vocabulary in technical_vocabulary.json.
    requirements = extract_requirements(customer_input)

    # TF-IDF transforms the new message into numerical features. Logistic Regression predicts one of the three project categories.
    text_features = vectorizer.transform([customer_input])
    predicted_category = classifier.predict(text_features)[0]

    category_labels = {
        "historian_visualization": "Historian and Visualization",
        "cloud_data_sharing": "Cloud Data Sharing",
        "source_connectivity": "Source Connectivity"
    }

    project_category = category_labels.get(
        predicted_category,
        predicted_category
    )

    # Update the saved requirements as the customer provides more details across messages.
    project = st.session_state.project_requirements

    for item in requirements["equipment"]:
        if item not in project["equipment"]:
            project["equipment"].append(item)

    for protocol in requirements["communication_protocols"]:
        if protocol not in project["communication_protocols"]:
            project["communication_protocols"].append(protocol)

    if requirements["data_destinations"]:
        project["data_destination"] = requirements["data_destinations"][0]

    protocols = project["communication_protocols"]
    equipment = project["equipment"]

    # Build the chatbot response using the predicted category and collected requirements.
    assistant_response = (
        f"**Predicted Project Category:** {project_category}\n\n"
        "Here's what I've identified:\n\n"
    )

    if equipment:
        assistant_response += "**Equipment:**\n"

        for item in equipment:
            equipment_type = item["type"]
            quantity = item["quantity"]

            if quantity is not None:
                assistant_response += f"- {quantity} {equipment_type}\n"
            else:
                assistant_response += f"- {equipment_type} (quantity unknown)\n"

    else:
        assistant_response += "I couldn't identify any equipment yet.\n"

    if protocols:
        assistant_response += "\n**Communication Protocols:**\n"

        for protocol in protocols:
            assistant_response += f"- {protocol}\n"

    else:
        assistant_response += "\nNo communication protocols identified yet.\n"

    # Ask for the data destination if it has not been identified.
    if project["data_destination"] is None:
        assistant_response += (
            "\nWhere would you like this equipment data "
            "to be collected or stored?"
        )

    else:
        assistant_response += (
            f"\n**Data Destination:** "
            f"{project['data_destination']}\n\n"
            "Thank you. I've recorded the data destination."
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response
    })

    with st.chat_message("assistant"):
        st.write(assistant_response)


# _____________________________________________________________________________________
# GENERATE PRELIMINARY PROJECT PROPOSAL

# proposal_summary.py inserts the collected requirements into the boilerplate text stored in proposal_templates.json.

st.divider()

project = st.session_state.project_requirements

summary = generate_proposal_summary(project)

st.markdown(summary)