
# CSC525: Principles of Machine Learning
# Portfolio Project: Proposal Intake Assistant
# Preliminary Proposal Summary Generation
# Belinda Duncan
# November 8, 2026

# Proposal Summary = Uses the extracted customer requirements and JSON boilerplate templates to generate a preliminary project proposal.
# _____________________________________________________________________________________
# IMPORT REQUIRED LIBRARIES

import json
from pathlib import Path


# _____________________________________________________________________________________
# LOAD PROPOSAL TEMPLATE

# JSON stores the boilerplate proposal language separately from the Python code, allowing the text to be updated without changing the application.

template_path = (
    Path(__file__).resolve().parent / "proposal_templates.json"
)


# _____________________________________________________________________________________
# GENERATE PRELIMINARY PROPOSAL

def generate_proposal_summary(project):
    """Generate a preliminary proposal using the collected customer requirements."""

    # Load the boilerplate language from the JSON file.
    with open(template_path, "r", encoding="utf-8") as file:
        templates = json.load(file)

    summary = "## Preliminary Project Proposal\n\n"

    # Format equipment descriptions and include quantities when available.
    equipment_list = []

    for item in project["equipment"]:
        equipment_type = item["type"]
        quantity = item["quantity"]

        if quantity is not None:
            equipment_list.append(f"{quantity} {equipment_type}")
        else:
            equipment_list.append(equipment_type)

    equipment_text = (
        ", ".join(equipment_list)
        if equipment_list else "equipment to be confirmed"
    )

    # Format communication protocols.
    protocols = project["communication_protocols"]

    protocols_text = (
        ", ".join(protocols)
        if protocols else "communication protocols to be confirmed"
    )

    # Use a placeholder if the customer has not identified a data destination.
    destination_text = (
        project["data_destination"]
        or "a data destination to be confirmed"
    )

    # Match the collected requirements to the placeholders in the JSON templates.
    values = {
        "equipment": equipment_text,
        "protocols": protocols_text,
        "data_destination": destination_text
    }

    # _____________________________________________________________________________________
    # BUILD PROPOSAL SECTIONS

    # .format(**values) replaces the JSON placeholders with the collected customer information.

    summary += "### 1. Executive Summary\n\n"
    summary += templates["executive_summary"].format(**values)
    summary += "\n\n"

    summary += "### 2. Project Scope and Objectives\n\n"
    summary += templates["project_scope"].format(**values)
    summary += "\n\n"

    summary += "### 3. Discovery Requirements\n\n"
    summary += templates["discovery_requirements"].format(**values)
    summary += "\n\n"

    # Include the extracted requirements as a separate section for reference.
    summary += "### 4. Identified Technical Requirements\n\n"

    summary += f"**Equipment:** {equipment_text}\n\n"
    summary += f"**Communication Protocols:** {protocols_text}\n\n"
    summary += f"**Data Destination:** {destination_text}\n\n"

    return summary
