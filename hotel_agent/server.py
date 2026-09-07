from fastmcp import FastMCP

from .graph import hotel_graph


# ---------------------------------------------------------
# Create MCP server
# ---------------------------------------------------------

mcp = FastMCP(
    "Hotel Agent"
)


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
