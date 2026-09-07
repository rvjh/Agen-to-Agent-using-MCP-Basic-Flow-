from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------
# State
# ---------------------------------------------------------

class HotelState(TypedDict):
    location: str
    max_price: int
    nights: int
    hotels: list


# ---------------------------------------------------------
# Fake hotel database
# ---------------------------------------------------------

HOTELS = [
    {
        "name": "Bengaluru Grand Hotel",
        "location": "Bengaluru",
        "price": 2500,
        "rating": 4.3,
    },
    {
        "name": "MG Road Business Hotel",
        "location": "Bengaluru",
        "price": 3200,
        "rating": 4.5,
    },
    {
        "name": "Koramangala Comfort Inn",
        "location": "Bengaluru",
        "price": 1800,
        "rating": 4.1,
    },
    {
        "name": "Indiranagar Premium Hotel",
        "location": "Bengaluru",
        "price": 4500,
        "rating": 4.7,
    },
    {
        "name": "Mumbai Central Hotel",
        "location": "Mumbai",
        "price": 4000,
        "rating": 4.2,
    },
]


# ---------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------

def search_hotels(state: HotelState):

    location = state["location"].lower()
    max_price = state["max_price"]

    matching_hotels = []

    for hotel in HOTELS:

        if (
            hotel["location"].lower() == location
            and hotel["price"] <= max_price
        ):
            matching_hotels.append(hotel)

    # Highest rating first
    matching_hotels.sort(
        key=lambda x: x["rating"],
        reverse=True,
    )

    return {
        "hotels": matching_hotels
    }


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

builder = StateGraph(HotelState)

builder.add_node(
    "search_hotels",
    search_hotels,
)

builder.add_edge(
    START,
    "search_hotels",
)

builder.add_edge(
    "search_hotels",
    END,
)


# ---------------------------------------------------------
# THIS is the variable server.py imports
# ---------------------------------------------------------

hotel_graph = builder.compile()
