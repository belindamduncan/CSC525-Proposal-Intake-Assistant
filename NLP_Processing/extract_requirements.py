
# CSC525: Principles of Machine Learning
# Portfolio Project: Proposal Intake Assistant
# NLP Processing: Technical Requirement Extraction
# Belinda Duncan
# November 8, 2026

# NLP = Natural Language Processing allows the application to identify equipment, communication protocols, and data destinations from customer text.
# _____________________________________________________________________________________
# IMPORT REQUIRED LIBRARIES

import json
from pathlib import Path

import spacy
from spacy.matcher import PhraseMatcher
# _____________________________________________________________________________________
# LOAD SPACY AND TECHNICAL VOCABULARY

# spaCy processes natural language text. The en_core_web_sm model provides English language processing capabilities.

nlp = spacy.load("en_core_web_sm")

vocabulary_path = (
    Path(__file__).resolve().parent / "technical_vocabulary.json"
)

# Load the industrial automation terms stored in the JSON vocabulary file.
with open(vocabulary_path, "r", encoding="utf-8") as file:
    vocabulary = json.load(file)
# _____________________________________________________________________________________
# CREATE PHRASE MATCHERS

# PhraseMatcher identifies predefined terms in customer text. LOWER ignores capitalization, while LEMMA matches words using their base forms.

# Communication protocols
protocol_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

protocol_patterns = [
    nlp.make_doc(protocol)
    for protocol in vocabulary["protocols"]
]

protocol_matcher.add("COMMUNICATION_PROTOCOL", protocol_patterns)


# Equipment
# LOWER matches equipment terms regardless of capitalization. Singular and plural terms are listed separately in the vocabulary.
equipment_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

equipment_patterns = [
    nlp(equipment)
    for equipment in vocabulary["equipment"]
]

equipment_matcher.add("EQUIPMENT", equipment_patterns)


# Data destinations
destination_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

destination_patterns = [
    nlp.make_doc(destination)
    for destination in vocabulary["data_destinations"]
]

destination_matcher.add("DATA_DESTINATION", destination_patterns)
# _____________________________________________________________________________________
# EXTRACT CUSTOMER REQUIREMENTS

def extract_requirements(customer_request):

    doc = nlp(customer_request)

    # Identify communication protocols and avoid duplicates.
    found_protocols = []

    for match_id, start, end in protocol_matcher(doc):
        protocol = doc[start:end].text

        if protocol not in found_protocols:
            found_protocols.append(protocol)

    # Identify equipment and check for a quantity immediately before the equipment name.
    found_equipment = []

    for match_id, start, end in equipment_matcher(doc):
        equipment = doc[start:end].text

        quantity = None

        if start > 0:
            previous_token = doc[start - 1]

            if previous_token.like_num:
                try:
                    quantity = int(previous_token.text)
                except ValueError:
                    pass

        found_equipment.append({
            "type": equipment,
            "quantity": quantity
        })

    # Identify data destinations and avoid duplicates.
    found_destinations = []

    for match_id, start, end in destination_matcher(doc):
        destination = doc[start:end].text

        if destination not in found_destinations:
            found_destinations.append(destination)

    return {
        "communication_protocols": found_protocols,
        "equipment": found_equipment,
        "data_destinations": found_destinations
    }
# _____________________________________________________________________________________
# TEST REQUIREMENT EXTRACTION

# Run this test only when executing this file directly, not when importing it into app.py.

if __name__ == "__main__":

    customer_request = (
        "We want to store the data in AVEVA PI."
    )

    results = extract_requirements(customer_request)

    print("Customer Request:")
    print(customer_request)

    print("\nExtracted Requirements:")
    print(results)
