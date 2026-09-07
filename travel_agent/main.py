import asyncio
import os

from dotenv import load_dotenv

from .graph import travel_graph


# ---------------------------------------------------------
# Load .env
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

async def main():

    user_request = """
    Find me a hotel in Bengaluru for 3 nights.
    My budget is 4000 rupees per night.
    """

    print()
    print("===================================")
    print("USER")
    print("===================================")

    print(user_request)

    # -----------------------------------------
    # Run Travel Agent
    # -----------------------------------------

    result = await travel_graph.ainvoke(
        {
            "user_request": user_request,
            "hotel_request": {},
            "hotel_result": {},
            "final_answer": "",
        }
    )

    # -----------------------------------------
    # Print answer
    # -----------------------------------------

    print()
    print("===================================")
    print("FINAL RESPONSE")
    print("===================================")

    print(
        result["final_answer"]
    )


if __name__ == "__main__":
    asyncio.run(main())
