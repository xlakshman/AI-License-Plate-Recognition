"""
utils/validator.py
License Plate Validation, State Code Mapping, and Character Heuristic Correction.
Designed for Indian HSRP (High Security Registration Plate), Bharat Series (BH), and Generic Formats.
"""

import re
from typing import Dict, Any, Optional

# Indian State & Union Territory Codes (36 RTO codes)
INDIAN_STATES: Dict[str, str] = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CH": "Chandigarh",
    "CG": "Chhattisgarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

# Character disambiguation maps based on optical character confusion
CHAR_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1', '|': '1',
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5',
    'G': '6', 'b': '6',
    'T': '7',
    'B': '8',
    'g': '9', 'q': '9'
}

DIGIT_TO_CHAR = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'P'
}

# Common State Code OCR Confusion corrections
STATE_CODE_CONFUSIONS = {
    "0L": "DL", "OL": "DL", "QL": "DL", "BL": "DL", "DI": "DL",
    "0P": "UP", "OP": "UP", "VP": "UP",
    "NH": "MH", "HH": "MH", "MK": "MH",
    "KR": "KA", "Kh": "KA",
    "IN": "TN", "TM": "TN",
    "HB": "HR", "HA": "HR",
    "GI": "GJ", "CJ": "GJ",
    "RJ": "RJ", "BI": "BR",
    "AP": "AP", "IS": "TS"
}


def clean_plate_text(raw_text: str) -> str:
    """Removes spaces, hyphens, and non-alphanumeric characters, returning uppercase string."""
    if not raw_text:
        return ""
    # Filter only alphanumeric characters
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
    return cleaned


def correct_ocr_mistakes_indian(text: str) -> str:
    """
    Applies contextual heuristics to fix common OCR confusions in Indian plate patterns.
    Standard Indian format: [2 letters State] [1-2 digits RTO] [1-3 letters Series] [4 digits Number]
    Total length typically: 9 or 10 characters.
    """
    text = clean_plate_text(text)
    if len(text) < 6:
        return text

    # Check for Bharat Series: e.g. 22BH1234AA (2 digits, BH, 4 digits, 2 letters)
    bh_match = re.match(r'^([0-9OIZESGB]{2})(BH)([0-9OIZESGB]{4})([A-Z0-9]{1,2})$', text)
    if bh_match:
        year_part = "".join(CHAR_TO_DIGIT.get(c, c) for c in bh_match.group(1))
        num_part = "".join(CHAR_TO_DIGIT.get(c, c) for c in bh_match.group(3))
        series_part = "".join(DIGIT_TO_CHAR.get(c, c) for c in bh_match.group(4))
        return f"{year_part}BH{num_part}{series_part}"

    # Standard format: State(2) + District(1 or 2) + Series(1 to 3) + Number(4)
    chars = list(text)
    n = len(chars)

    # First two should be letters (State code)
    if n >= 2:
        chars[0] = DIGIT_TO_CHAR.get(chars[0], chars[0])
        chars[1] = DIGIT_TO_CHAR.get(chars[1], chars[1])
        prefix = chars[0] + chars[1]
        if prefix in STATE_CODE_CONFUSIONS:
            fixed_pref = STATE_CODE_CONFUSIONS[prefix]
            chars[0] = fixed_pref[0]
            chars[1] = fixed_pref[1]

    # Last 4 characters should be digits (Unique number)
    if n >= 8:
        for idx in range(n - 4, n):
            chars[idx] = CHAR_TO_DIGIT.get(chars[idx], chars[idx])

    # Position 2 and 3 (RTO code) are usually digits
    if n >= 4:
        chars[2] = CHAR_TO_DIGIT.get(chars[2], chars[2])
        if n >= 9 and (chars[3].isdigit() or chars[3] in CHAR_TO_DIGIT):
            chars[3] = CHAR_TO_DIGIT.get(chars[3], chars[3])

    corrected = "".join(chars)
    return corrected


def validate_plate(text: str) -> Dict[str, Any]:
    """
    Validates a license plate against Indian Standard, Bharat Series, and Generic formats.
    Returns metadata including validity, state name, formatted plate, and components.
    """
    cleaned = clean_plate_text(text)
    corrected = correct_ocr_mistakes_indian(cleaned)

    # 1. Check Standard Indian Registration Format (e.g., MH12DE1433, DL01AB1234, KA03MG2341)
    # Pattern: 2 Letters (State) + 1-2 Digits (RTO) + 1-3 Letters (Series) + 4 Digits (Number)
    indian_regex = r'^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{4})$'
    match_indian = re.match(indian_regex, corrected)

    if match_indian:
        state_code = match_indian.group(1)
        rto_code = match_indian.group(2)
        series = match_indian.group(3)
        number = match_indian.group(4)
        state_name = INDIAN_STATES.get(state_code, "Indian Registered State")
        formatted = f"{state_code} {rto_code} {series} {number}"

        return {
            "is_valid": True,
            "plate_number": corrected,
            "formatted_plate": formatted,
            "format_type": "Indian Standard (HSRP)",
            "state_code": state_code,
            "state_name": state_name,
            "district_code": rto_code,
            "series": series,
            "unique_number": number,
            "raw_text": text
        }

    # 2. Check Bharat Series Format (e.g., 22BH8888AA)
    bh_regex = r'^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$'
    match_bh = re.match(bh_regex, corrected)
    if match_bh:
        year = match_bh.group(1)
        number = match_bh.group(3)
        series = match_bh.group(4)
        return {
            "is_valid": True,
            "plate_number": corrected,
            "formatted_plate": f"{year} BH {number} {series}",
            "format_type": "Bharat Series (BH)",
            "state_code": "BH",
            "state_name": "Bharat Series (Pan-India)",
            "district_code": year,
            "series": series,
            "unique_number": number,
            "raw_text": text
        }

    # 3. Check General / Partial Plate Format (e.g. 5 to 12 alphanumeric characters)
    if len(corrected) >= 5:
        # Check if first two characters match known state
        potential_state = corrected[:2]
        state_name = INDIAN_STATES.get(potential_state, "Custom / International")
        state_code = potential_state if potential_state in INDIAN_STATES else "OT"

        return {
            "is_valid": True if potential_state in INDIAN_STATES else False,
            "plate_number": corrected,
            "formatted_plate": corrected,
            "format_type": "Standard / Custom Plate",
            "state_code": state_code,
            "state_name": state_name,
            "district_code": "",
            "series": "",
            "unique_number": "",
            "raw_text": text
        }

    # Unrecognized / invalid
    return {
        "is_valid": False,
        "plate_number": corrected if corrected else "UNKNOWN",
        "formatted_plate": corrected if corrected else "UNKNOWN",
        "format_type": "Unrecognized Format",
        "state_code": "NA",
        "state_name": "Unknown",
        "district_code": "",
        "series": "",
        "unique_number": "",
        "raw_text": text
    }
