# LangGraph + FastMCP Multi-Agent Travel Assistant

An agent-to-agent (A2A-style) demo built using **LangGraph**, **FastMCP**, and **OpenAI**. The system coordinates two independent agent processes: a client-facing Travel Agent that parses user requests, and a specialized Hotel Agent exposed over HTTP via the Model Context Protocol (MCP).

---

## Architecture Overview

```text
                    USER
                      | (Natural Language)
                      v
             +------------------+
             |   Travel Agent   |
             |    (LangGraph)   |
             +------------------+
                      |
                      | MCP over HTTP (Streamable HTTP)
                      v
             +------------------+
             |     FastMCP      |
             |   (MCP Server)   |
             +------------------+
                      |
                      v
             +------------------+
             |   Hotel Agent    |
             |    (LangGraph)   |
             +------------------+
                      |
                      v
                Hotel Database
```

* **Process 1 (Hotel Agent + FastMCP):** Runs an MCP server on `http://127.0.0.1:8001` exposing the `find_hotels` tool powered by a LangGraph node.
* **Process 2 (Travel Agent):** Parses natural language inputs, extracts search parameters using `gpt-4o-mini`, calls the remote Hotel Agent over MCP, and returns a formatted recommendation.

---

## Directory Structure

```text
a2a-langgraph-demo/
│
├── .env
├── requirements.txt
├── README.md
│
├── hotel_agent/
│   ├── __init__.py
│   ├── graph.py
│   └── server.py
│
└── travel_agent/
    ├── __init__.py
    ├── graph.py
    └── main.py
```

---

## Prerequisites & Installation

### 1. Set Up Virtual Environment

```bash
# Windows
python -m venv myenv1
.\myenv1\Scripts\activate

# macOS / Linux
python3 -m venv myenv1
source myenv1/bin/activate
```

### 2. Install Dependencies

Create `requirements.txt`:
```text
langgraph
langchain
langchain-openai
fastmcp
python-dotenv
```

Install via pip:
```bash
pip install -U -r requirements.txt
```

### 3. Configure Environment Variables

Create `.env` in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Code Implementation

### `hotel_agent/graph.py`

```python
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
builder.add_node("search_hotels", search_hotels)
builder.add_edge(START, "search_hotels")
builder.add_edge("search_hotels", END)

# Expose compiled graph variable imported by server.py
hotel_graph = builder.compile()
```

---

### `hotel_agent/server.py`

```python
from fastmcp import FastMCP
from .graph import hotel_graph

# ---------------------------------------------------------
# Create MCP server
# ---------------------------------------------------------
mcp = FastMCP("Hotel Agent")

# ---------------------------------------------------------
# MCP Tool
# ---------------------------------------------------------
@mcp.tool
def find_hotels(
    location: str,
    max_price: int,
    nights: int,
) -> dict:
    """
    Find hotels for a location and maximum price per night.
    """
    print()
    print("===================================")
    print("HOTEL AGENT")
    print("===================================")
    print(f"Location : {location}")
    print(f"Max price: ₹{max_price}")
    print(f"Nights   : {nights}")

    # -----------------------------------------
    # Call LangGraph
    # -----------------------------------------
    result = hotel_graph.invoke(
        {
            "location": location,
            "max_price": max_price,
            "nights": nights,
            "hotels": [],
        }
    )
    print()
    print("Hotels found:", len(result["hotels"]))
    return {
        "location": location,
        "max_price": max_price,
        "nights": nights,
        "hotels": result["hotels"],
    }

# ---------------------------------------------------------
# Start server
# ---------------------------------------------------------
if __name__ == "__main__":
    print()
    print("===================================")
    print("HOTEL AGENT MCP SERVER")
    print("===================================")
    print("Starting on http://127.0.0.1:8001")
    print()
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8001,
    )
```

---

### `travel_agent/graph.py`

```python
import json
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from fastmcp import Client

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
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# ---------------------------------------------------------
# Node 1: Understand user's request
# ---------------------------------------------------------
def understand_request(state: TravelState):
    user_request = state["user_request"]
    print()
    print("===================================")
    print("TRAVEL AGENT")
    print("===================================")
    print("User request:")
    print(user_request)
    
    prompt = f"""You are a travel assistant.
Extract hotel search information from the user's request.
User request:
{user_request}

Return ONLY valid JSON.
The JSON must have exactly these fields:
{{
    "location": "city",
    "max_price": 5000,
    "nights": 2
}}"""
    response = model.invoke(prompt)
    content = response.content
    content = content.replace("```json", "").replace("```", "").strip()
    hotel_request = json.loads(content)
    print()
    print("Travel Agent understood:")
    print(hotel_request)
    return {"hotel_request": hotel_request}

# ---------------------------------------------------------
# Node 2: Call Hotel Agent through MCP
# ---------------------------------------------------------
async def call_hotel_agent(state: TravelState):
    request = state["hotel_request"]
    print()
    print("===================================")
    print("A2A-LIKE AGENT COMMUNICATION")
    print("===================================")
    print("Travel Agent -> Hotel Agent")
    print()
    print(request)
    
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "find_hotels",
            request,
        )
        
    print()
    print("Hotel Agent -> Travel Agent")
    print()
    print(result)
    return {"hotel_result": result}

# ---------------------------------------------------------
# Node 3: Generate final response
# ---------------------------------------------------------
def create_final_answer(state: TravelState):
    request = state["hotel_request"]
    hotel_result = state["hotel_result"]
    prompt = f"""You are a travel assistant.
The user requested:
Location: {request["location"]}
Maximum price per night: ₹{request["max_price"]}
Number of nights: {request["nights"]}

The Hotel Agent returned:
{hotel_result}

Create a concise response for the user.
For every hotel include:
- Hotel name
- Price per night
- Rating
- Estimated total for the requested number of nights
Do not invent information."""
    response = model.invoke(prompt)
    return {"final_answer": response.content}

# ---------------------------------------------------------
# Build Travel LangGraph
# ---------------------------------------------------------
builder = StateGraph(TravelState)
builder.add_node("understand_request", understand_request)
builder.add_node("call_hotel_agent", call_hotel_agent)
builder.add_node("create_final_answer", create_final_answer)

builder.add_edge(START, "understand_request")
builder.add_edge("understand_request", "call_hotel_agent")
builder.add_edge("call_hotel_agent", "create_final_answer")
builder.add_edge("create_final_answer", END)

travel_graph = builder.compile()
```

---

### `travel_agent/main.py`

```python
import asyncio
import os
from dotenv import load_dotenv
from .graph import travel_graph

load_dotenv()

async def main():
    user_request = """
    Find me a hotel in Bengaluru for 2 nights.
    My budget is 5000 rupees per night.
    """
    print()
    print("===================================")
    print("USER")
    print("===================================")
    print(user_request)

    result = await travel_graph.ainvoke(
        {
            "user_request": user_request,
            "hotel_request": {},
            "hotel_result": {},
            "final_answer": "",
        }
    )

    print()
    print("===================================")
    print("FINAL RESPONSE")
    print("===================================")
    print(result["final_answer"])

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Execution Guide

### 1. Test the Hotel Graph Standalone
Ensure compilation and import work without error:
```bash
python -c "from hotel_agent.graph import hotel_graph; print(hotel_graph.invoke({'location':'Bengaluru','max_price':5000,'nights':2,'hotels':[]}))"
```

### 2. Start the Hotel Agent MCP Server
In **Terminal 1**:
```bash
python -m hotel_agent.server
```
Leave this process running on `http://127.0.0.1:8001`.

### 3. Run the Travel Agent
In **Terminal 2**:
```bash
python -m travel_agent.main
```

---

## Architectural Note: MCP vs. A2A

* **Current Architecture (MCP as Tool Interface):**  
  `Travel Agent ──(MCP Tool Call)──> Hotel Agent Server ──> LangGraph`  
  FastMCP exposes the `find_hotels` function as a standard tool endpoint.

* **Target Multi-Agent Architecture (Peer-to-Peer A2A):**  
  `Travel Agent <──(A2A Protocol Negotiation)──> Hotel Agent <──(MCP)──> Internal DB / Tools`  
  In a formal A2A system, agents converse and delegate tasks via an agent protocol, while MCP remains the interface between each individual agent and its local or external tool suite.
