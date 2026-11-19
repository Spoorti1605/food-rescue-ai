import os
import re
from typing import TypedDict, Optional
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

try:
    if os.getenv("GOOGLE_API_KEY"):
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0
        )
    else:
        llm = None
except Exception as error:
    print("Gemini initialization failed:", error)
    llm = None


# ============================================================
# 2. SHARED STATE USED BY ALL FIVE AGENTS
# ============================================================

class FoodState(TypedDict, total=False):
    provider: str
    food_type: str
    quantity_kg: float
    condition: str
    preparation_time: str
    expiry_time: str
    pickup_location: str
    latitude: float
    longitude: float

    simulate_api_failure: bool

    collection_status: str
    collection_message: str

    safety_status: str
    safety_message: str

    matched_ngo: Optional[str]
    ngo_location: Optional[str]
    distance_km: float
    ngo_match_message: str

    estimated_delivery_hours: float
    urgency: str
    logistics_message: str

    notification_message: str
    tracking_status: str
    event_log: list


# ============================================================
# 3. DEMO NGO DATABASE
# ============================================================

NGO_DATABASE = [
    {
        "name": "Helping Hands NGO (Demo)",
        "location": "Hubballi",
        "latitude": 15.3700,
        "longitude": 75.1200,
        "supported_foods": ["rice", "bread"],
        "capacity_kg": 30.0
    },
    {
        "name": "Food for All NGO (Demo)",
        "location": "Dharwad",
        "latitude": 15.4589,
        "longitude": 75.0078,
        "supported_foods": ["vegetables", "bread"],
        "capacity_kg": 15.0
    },
    {
        "name": "Community Care NGO (Demo)",
        "location": "Hubballi",
        "latitude": 15.3400,
        "longitude": 75.1400,
        "supported_foods": ["rice", "vegetables", "bread"],
        "capacity_kg": 50.0
    }
]


# ============================================================
# 4. DISTANCE CALCULATION
# ============================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    earth_radius = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return earth_radius * c


# ============================================================
# AGENT 1: FOOD COLLECTION AGENT
# ============================================================

def food_collection_agent(state: FoodState):

    print("\n[AGENT 1] FOOD COLLECTION AGENT")

    required_fields = [
        "provider",
        "food_type",
        "condition",
        "preparation_time",
        "expiry_time",
        "pickup_location",
        "quantity_kg",
        "latitude",
        "longitude"
    ]

    for field in required_fields:
        if field not in state or state[field] is None:
            return {
                "collection_status": "INVALID",
                "collection_message": (
                    f"Missing required information: {field}"
                )
            }

    try:
        quantity = float(state["quantity_kg"])
    except (TypeError, ValueError):
        return {
            "collection_status": "INVALID",
            "collection_message": "Quantity must be a valid number."
        }

    if quantity <= 0:
        return {
            "collection_status": "INVALID",
            "collection_message": (
                "Quantity must be greater than zero."
            )
        }

    try:
        latitude = float(state["latitude"])
        longitude = float(state["longitude"])

        if not (-90 <= latitude <= 90):
            raise ValueError()

        if not (-180 <= longitude <= 180):
            raise ValueError()

    except (TypeError, ValueError):
        return {
            "collection_status": "INVALID",
            "collection_message": "Invalid pickup coordinates."
        }

    try:
        preparation = datetime.fromisoformat(
            state["preparation_time"]
        )

        expiry = datetime.fromisoformat(
            state["expiry_time"]
        )

        current_time = datetime.now(
            expiry.tzinfo
        )

        if preparation.tzinfo != expiry.tzinfo:
            raise ValueError(
                "Preparation and expiry times must use "
                "the same timezone format."
            )

        if preparation > current_time + timedelta(minutes=5):
            raise ValueError(
                "Preparation time cannot be in the future."
            )

        if expiry <= preparation:
            raise ValueError(
                "Expiry time must be later than preparation time."
            )

        if expiry <= current_time:
            raise ValueError("Food has already expired.")

    except (ValueError, TypeError) as error:
        return {
            "collection_status": "INVALID",
            "collection_message": str(error)
        }

    return {
        "collection_status": "VALID",
        "collection_message": (
            "Donation information successfully collected "
            "and validated."
        ),
        "event_log": ["Donation information collected."]
    }


# ============================================================
# GEMINI RESPONSE EXTRACTION
# ============================================================

def extract_response_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for block in content:

            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                if "text" in block:
                    text_parts.append(str(block["text"]))

        return " ".join(text_parts)

    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])

    return str(content)


def interpret_gemini_response(raw_text):

    text = raw_text.strip().upper()

    # Only accept an explicit classification token.
    match = re.search(
        r"\b(FRESH|SPOILED|UNCERTAIN)\b",
        text
    )

    if match:
        return match.group(1)

    return "UNCERTAIN"


# ============================================================
# AGENT 2: FOOD SAFETY AGENT
# ============================================================

def food_safety_agent(state: FoodState):

    print("\n[AGENT 2] FOOD SAFETY AGENT")

    condition = str(state["condition"]).strip().lower()

    if state.get("simulate_api_failure", False):
        return {
            "safety_status": "MANUAL_INSPECTION",
            "safety_message": (
                "Simulated Gemini API failure. "
                "Human inspection required."
            )
        }

    if llm is None:
        return {
            "safety_status": "MANUAL_INSPECTION",
            "safety_message": (
                "Gemini is unavailable. Human inspection required."
            )
        }

    if condition in ["unknown", "uncertain", "not sure", ""]:
        return {
            "safety_status": "MANUAL_INSPECTION",
            "safety_message": (
                "Food condition is unknown. "
                "Human inspection required."
            )
        }

    # Clear signs of spoilage are rejected conservatively.
    spoiled_terms = [
        "spoiled",
        "rotten",
        "decayed",
        "mouldy",
        "moldy"
    ]

    if any(term in condition for term in spoiled_terms):
        return {
            "safety_status": "REJECTED",
            "safety_message": (
                "Reported condition indicates spoiled food. "
                "Donation rejected."
            )
        }

    prompt = f"""
You are a preliminary food-condition screening assistant.

Food type: {state['food_type']}
Reported condition: {state['condition']}

Based only on the reported information, classify the condition.

Reply with exactly ONE word:

FRESH
SPOILED
UNCERTAIN

Do not provide explanations or additional text.

Important:
This is only preliminary screening, not food safety certification.
"""

    try:
        response = llm.invoke(prompt)

        raw_text = extract_response_text(response.content)

        decision = interpret_gemini_response(raw_text)

        print("Gemini classification:", decision)

        if decision == "FRESH":
            return {
                "safety_status": "PRELIMINARILY_ELIGIBLE",
                "safety_message": (
                    "Reported as fresh by Gemini. "
                    "Physical human inspection is still "
                    "required before distribution."
                )
            }

        if decision == "SPOILED":
            return {
                "safety_status": "REJECTED",
                "safety_message": (
                    "Gemini identified the reported condition "
                    "as spoiled. Donation rejected."
                )
            }

        return {
            "safety_status": "MANUAL_INSPECTION",
            "safety_message": (
                "AI response was uncertain or could not "
                "be interpreted. Human inspection required."
            )
        }

    except Exception as error:

        print("Gemini API error:", str(error))

        return {
            "safety_status": "MANUAL_INSPECTION",
            "safety_message": (
                "AI screening failed because of an API error. "
                "Human inspection required."
            )
        }


# ============================================================
# AGENT 3: NGO MATCHING AGENT
# ============================================================

def ngo_matching_agent(state: FoodState):

    print("\n[AGENT 3] NGO MATCHING AGENT")

    food = state["food_type"].strip().lower()
    quantity = float(state["quantity_kg"])

    suitable_ngos = []

    for ngo in NGO_DATABASE:

        supports_food = food in ngo["supported_foods"]

        has_capacity = quantity <= ngo["capacity_kg"]

        if supports_food and has_capacity:

            distance = calculate_distance(
                state["latitude"],
                state["longitude"],
                ngo["latitude"],
                ngo["longitude"]
            )

            suitable_ngos.append({
                "name": ngo["name"],
                "location": ngo["location"],
                "distance": distance
            })

    if not suitable_ngos:
        return {
            "matched_ngo": None,
            "ngo_location": None,
            "distance_km": 0.0,
            "ngo_match_message": (
                "No suitable NGO found for the food type "
                "or available quantity."
            )
        }

    nearest_ngo = min(
        suitable_ngos,
        key=lambda ngo: ngo["distance"]
    )

    return {
        "matched_ngo": nearest_ngo["name"],
        "ngo_location": nearest_ngo["location"],
        "distance_km": round(nearest_ngo["distance"], 2),
        "ngo_match_message": (
            "Suitable NGO found successfully."
        ),
        "event_log": ["Suitable NGO matched."]
    }


# ============================================================
# AGENT 4: LOGISTICS AGENT
# ============================================================

def logistics_agent(state: FoodState):

    print("\n[AGENT 4] LOGISTICS AGENT")

    distance = state["distance_km"]

    # Demonstration assumption:
    # Average speed = 25 km/hour.
    # Handling and preparation allowance = 30 minutes.

    average_speed = 25.0
    handling_time = 0.5

    travel_time = distance / average_speed

    total_delivery_time = travel_time + handling_time

    expiry = datetime.fromisoformat(state["expiry_time"])

    remaining_time = (
        expiry - datetime.now(expiry.tzinfo)
    ).total_seconds() / 3600

    usable_time = remaining_time - total_delivery_time

    if usable_time <= 2:
        urgency = "CRITICAL"
    elif usable_time <= 4:
        urgency = "HIGH"
    elif usable_time <= 8:
        urgency = "MEDIUM"
    else:
        urgency = "LOW"

    return {
        "estimated_delivery_hours": round(
            total_delivery_time, 2
        ),
        "urgency": urgency,
        "logistics_message": (
            "Delivery time estimated using distance, "
            "assumed travel speed and handling time. "
            "This is a demonstration estimate, not live routing."
        ),
        "event_log": ["Logistics estimate generated."]
    }


# ============================================================
# AGENT 5: NOTIFICATION AND TRACKING AGENT
# ============================================================

def notification_tracking_agent(state: FoodState):

    print("\n[AGENT 5] NOTIFICATION AND TRACKING AGENT")

    if state.get("collection_status") == "INVALID":

        message = (
            "Donation rejected during collection validation."
        )
        tracking = "REJECTED"

    elif state.get("safety_status") == "REJECTED":

        message = (
            "Donation rejected by preliminary food-condition screening."
        )
        tracking = "REJECTED"

    elif state.get("safety_status") == "MANUAL_INSPECTION":

        message = (
            "Donation held for human inspection. "
            "No NGO assignment has been made."
        )
        tracking = "PENDING_HUMAN_INSPECTION"

    elif state.get("matched_ngo") is None:

        message = (
            "No suitable NGO found. Donation awaits reassignment."
        )
        tracking = "PENDING_NGO_MATCH"

    else:

        message = (
            f"Donation matched with {state['matched_ngo']}. "
            "Delivery planning completed. "
            "Actual delivery is not performed by this demo."
        )
        tracking = "MATCHED_AND_PLANNED"

    return {
        "notification_message": message,
        "tracking_status": tracking,
        "event_log": ["Notification and tracking status updated."]
    }


# ============================================================
# 5. LANGGRAPH WORKFLOW AND CONDITIONAL ROUTING
# ============================================================

def route_after_collection(state: FoodState):

    if state.get("collection_status") == "VALID":
        return "continue"

    return "stop"


def route_after_safety(state: FoodState):

    if state.get("safety_status") == "PRELIMINARILY_ELIGIBLE":
        return "continue"

    return "stop"


def route_after_matching(state: FoodState):

    if state.get("matched_ngo"):
        return "continue"

    return "stop"


workflow = StateGraph(FoodState)

workflow.add_node(
    "collection_agent",
    food_collection_agent
)

workflow.add_node(
    "safety_agent",
    food_safety_agent
)

workflow.add_node(
    "ngo_matching_agent",
    ngo_matching_agent
)

workflow.add_node(
    "logistics_agent",
    logistics_agent
)

workflow.add_node(
    "notification_agent",
    notification_tracking_agent
)

workflow.add_edge(START, "collection_agent")

workflow.add_conditional_edges(
    "collection_agent",
    route_after_collection,
    {
        "continue": "safety_agent",
        "stop": "notification_agent"
    }
)

workflow.add_conditional_edges(
    "safety_agent",
    route_after_safety,
    {
        "continue": "ngo_matching_agent",
        "stop": "notification_agent"
    }
)

workflow.add_conditional_edges(
    "ngo_matching_agent",
    route_after_matching,
    {
        "continue": "logistics_agent",
        "stop": "notification_agent"
    }
)

workflow.add_edge(
    "logistics_agent",
    "notification_agent"
)

workflow.add_edge(
    "notification_agent",
    END
)

app = workflow.compile()


# ============================================================
# 6. TEST DATA GENERATION
# ============================================================

def create_test_donation(
    provider,
    food_type,
    quantity,
    condition,
    expiry_hours=6,
    simulate_api_failure=False
):

    now = datetime.now()

    return {
        "provider": provider,
        "food_type": food_type,
        "quantity_kg": quantity,
        "condition": condition,
        "preparation_time": (
            now - timedelta(hours=1)
        ).isoformat(),
        "expiry_time": (
            now + timedelta(hours=expiry_hours)
        ).isoformat(),
        "pickup_location": "Hubballi",
        "latitude": 15.3647,
        "longitude": 75.1240,
        "simulate_api_failure": simulate_api_failure
    }


# ============================================================
# 7. POSITIVE AND NEGATIVE TEST CASES
# ============================================================

TEST_CASES = [

    {
        "name": "TEST 1 - Fresh Rice Donation",
        "data": create_test_donation(
            "City Restaurant",
            "rice",
            20,
            "fresh"
        ),
        "expected": {
            "collection_status": "VALID",
            "safety_status": "PRELIMINARILY_ELIGIBLE",
            "tracking_status": "MATCHED_AND_PLANNED"
        }
    },

    {
        "name": "TEST 2 - Spoiled Vegetables",
        "data": create_test_donation(
            "Green Leaf Restaurant",
            "vegetables",
            10,
            "spoiled"
        ),
        "expected": {
            "collection_status": "VALID",
            "safety_status": "REJECTED",
            "tracking_status": "REJECTED"
        }
    },

    {
        "name": "TEST 3 - Unknown Food Condition",
        "data": create_test_donation(
            "Sunrise Bakery",
            "bread",
            5,
            "unknown"
        ),
        "expected": {
            "collection_status": "VALID",
            "safety_status": "MANUAL_INSPECTION",
            "tracking_status": "PENDING_HUMAN_INSPECTION"
        }
    },

    {
        "name": "TEST 4 - Zero Quantity",
        "data": create_test_donation(
            "Test Restaurant",
            "rice",
            0,
            "fresh"
        ),
        "expected": {
            "collection_status": "INVALID",
            "tracking_status": "REJECTED"
        }
    },

    {
        "name": "TEST 5 - Negative Quantity",
        "data": create_test_donation(
            "Test Restaurant",
            "rice",
            -5,
            "fresh"
        ),
        "expected": {
            "collection_status": "INVALID",
            "tracking_status": "REJECTED"
        }
    },

    {
        "name": "TEST 6 - Expired Food",
        "data": create_test_donation(
            "Old Stock Restaurant",
            "rice",
            10,
            "fresh",
            expiry_hours=-2
        ),
        "expected": {
            "collection_status": "INVALID",
            "tracking_status": "REJECTED"
        }
    },

    {
        "name": "TEST 7 - No Suitable NGO",
        "data": create_test_donation(
            "Special Food Restaurant",
            "seafood",
            10,
            "fresh"
        ),
        "expected": {
            "collection_status": "VALID",
            "safety_status": "PRELIMINARILY_ELIGIBLE",
            "matched_ngo": None,
            "tracking_status": "PENDING_NGO_MATCH"
        }
    },

    {
        "name": "TEST 8 - Quantity Exceeds NGO Capacity",
        "data": create_test_donation(
            "Large Restaurant",
            "rice",
            100,
            "fresh"
        ),
        "expected": {
            "collection_status": "VALID",
            "safety_status": "PRELIMINARILY_ELIGIBLE",
            "matched_ngo": None,
            "tracking_status": "PENDING_NGO_MATCH"
        }
    },

    {
        "name": "TEST 9 - Gemini API Failure",
        "data": create_test_donation(
            "API Failure Test Restaurant",
            "rice",
            10,
            "fresh",
            simulate_api_failure=True
        ),
        "expected": {
            "collection_status": "VALID",
            "safety_status": "MANUAL_INSPECTION",
            "tracking_status": "PENDING_HUMAN_INSPECTION"
        }
    }
]


# ============================================================
# 8. DISPLAY RESULTS AND VERIFY EXPECTED BEHAVIOUR
# ============================================================

def run_test_case(test_case):

    print("\n" + "=" * 65)
    print(test_case["name"])
    print("=" * 65)

    try:
        result = app.invoke(test_case["data"])

        print("\n--- AGENT RESULTS ---")

        print(
            "Collection:",
            result.get("collection_status", "NOT RUN")
        )

        print(
            "Collection Message:",
            result.get("collection_message", "")
        )

        print(
            "Safety:",
            result.get("safety_status", "NOT RUN")
        )

        print(
            "Safety Message:",
            result.get("safety_message", "")
        )

        print(
            "Matched NGO:",
            result.get("matched_ngo") or "NONE"
        )

        print(
            "NGO Message:",
            result.get("ngo_match_message", "")
        )

        print(
            "Distance:",
            result.get("distance_km", 0),
            "km"
        )

        print(
            "Estimated Delivery:",
            result.get("estimated_delivery_hours", 0),
            "hours"
        )

        print(
            "Urgency:",
            result.get("urgency", "NOT CALCULATED")
        )

        print(
            "Logistics Message:",
            result.get("logistics_message", "")
        )

        print(
            "Notification:",
            result.get("notification_message", "")
        )

        print(
            "Tracking Status:",
            result.get("tracking_status", "")
        )

        print("\n--- TEST VERIFICATION ---")

        passed = True

        for key, expected_value in test_case["expected"].items():

            actual_value = result.get(key)

            if actual_value == expected_value:
                print(f"PASS: {key}")
            else:
                print(
                    f"FAIL: {key} | "
                    f"Expected: {expected_value} | "
                    f"Actual: {actual_value}"
                )
                passed = False

        print(
            "\nTEST RESULT:",
            "PASSED" if passed else "FAILED"
        )

        return passed

    except Exception as error:

        print("\nTEST EXECUTION ERROR:", str(error))
        print("TEST RESULT: FAILED")

        return False


# ============================================================
# 9. RUN COMPLETE SYSTEM TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("*" * 65)
    print("AGENTIC AI-BASED SMART FOOD RESCUE SYSTEM")
    print("FIVE-AGENT INTEGRATED WORKFLOW")
    print("*" * 65)

    results = []

    for test_case in TEST_CASES:
        results.append(
            run_test_case(test_case)
        )

    total = len(results)
    passed = sum(results)
    failed = total - passed

    print("\n" + "=" * 65)
    print("FINAL SYSTEM TEST SUMMARY")
    print("=" * 65)

    print("Total Test Cases:", total)
    print("Passed:", passed)
    print("Failed:", failed)

    if failed == 0:
        print("\nALL TEST CASES PASSED!")
    else:
        print("\nSOME TEST CASES NEED ATTENTION.")

    print("\nSYSTEM TESTING COMPLETED.")