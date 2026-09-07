import json
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from fastmcp import Client
from dotenv import load_dotenv

load_dotenv()




# ---------------------------------------------------------
# MCP server URL
# ---------------------------------------------------------

MCP_SERVER_URL = "http://127.0.0.1:8001/mcp"


# ---------------------------------------------------------
# State
# ---------------------------------------------------------

class TravelState(TypedDict):
    user_request: str
    hotel_request: dict
    hotel_result: Any
    final_answer: str


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# ---------------------------------------------------------
# Node 1
# Understand user's request
# ---------------------------------------------------------

def understand_request(
    state: TravelState,
):

    user_request = state["user_request"]

    print()
    print("===================================")
    print("TRAVEL AGENT")
    print("===================================")

    print("User request:")
    print(user_request)

    prompt = f"""
You are a travel assistant.

Extract hotel search information from the user's request.

User request:

{user_request}

Return ONLY valid JSON.

The JSON must have exactly these fields:

{{
    "location": "city",
    "max_price": 5000,
    "nights": 2
}}
"""

    response = model.invoke(prompt)

    content = response.content

    # Remove possible markdown formatting
    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    hotel_request = json.loads(content)

    print()
    print("Travel Agent understood:")
    print(hotel_request)

    return {
        "hotel_request": hotel_request
    }


# ---------------------------------------------------------
# Node 2
# Call Hotel Agent through MCP
# ---------------------------------------------------------

async def call_hotel_agent(
    state: TravelState,
):

    request = state["hotel_request"]

    print()
    print("===================================")
    print("A2A-LIKE AGENT COMMUNICATION")
    print("===================================")

    print("Travel Agent -> Hotel Agent")
    print()
    print(request)

    # -----------------------------------------
    # Connect to FastMCP
    # -----------------------------------------

    async with Client(MCP_SERVER_URL) as client:

        # -------------------------------------
        # Call MCP tool
        # -------------------------------------

        result = await client.call_tool(
            "find_hotels",
            request,
        )

    print()
    print("Hotel Agent -> Travel Agent")
    print()
    print(result)

    return {
        "hotel_result": result
    }


# ---------------------------------------------------------
# Node 3
# Generate final response
# ---------------------------------------------------------

def create_final_answer(
    state: TravelState,
):

    request = state["hotel_request"]
    hotel_result = state["hotel_result"]

    prompt = f"""
You are a travel assistant.

The user requested:

Location:
{request["location"]}

Maximum price per night:
₹{request["max_price"]}

Number of nights:
{request["nights"]}

The Hotel Agent returned:

{hotel_result}

Create a concise response for the user.

For every hotel include:

- Hotel name
- Price per night
- Rating
- Estimated total for the requested number of nights

Do not invent information.
"""

    response = model.invoke(prompt)

    return {
        "final_answer": response.content
    }


# ---------------------------------------------------------
# Build Travel LangGraph
# ---------------------------------------------------------

builder = StateGraph(TravelState)


builder.add_node(
    "understand_request",
    understand_request,
)


builder.add_node(
    "call_hotel_agent",
    call_hotel_agent,
)


builder.add_node(
    "create_final_answer",
    create_final_answer,
)


# START
builder.add_edge(
    START,
    "understand_request",
)


# Understand -> Hotel Agent
builder.add_edge(
    "understand_request",
    "call_hotel_agent",
)


# Hotel Agent -> Final answer
builder.add_edge(
    "call_hotel_agent",
    "create_final_answer",
)


# Final -> END
builder.add_edge(
    "create_final_answer",
    END,
)


# ---------------------------------------------------------
# Compile
# ---------------------------------------------------------

travel_graph = builder.compile()
