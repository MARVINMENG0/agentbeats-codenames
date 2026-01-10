import argparse
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)

from pydantic import BaseModel
from adk import FunctionTool

class Clue(BaseModel):
    word: str
    number: int

def submit_clue(word: str, number: int) -> Clue:
    return Clue(word=word, number=number)

clue_tool = FunctionTool(func=submit_clue)

def main():
    parser = argparse.ArgumentParser(description="Run the A2A debater agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()

    root_agent = Agent(
        name="debater",
        model="gemini-2.0-flash",
        description="Spymaster in a game of Codenames.",
        instruction="You are the spymaster in a game of codenames. You will be given a dictionary of words where the values can be red, blue, neutral, or assassin. /You will be assigned either to the red team or blue team. Whichever team you are assigned, you need to get your field agents to guess the words in the list that match your team color. You do this by giving a single word as a clue followed by a number. The clue should relate to some of the words that match your color. The number indicates how many of the words your clue relates to. You need your field agents to finish guessing your team's words before the other team does. You want to avoid having your field agents guess the assassin word, because then you automatically lose. The neutral colors do not benefit anyone, but if your team guesses a neutral word their turn ends. You will also be given a log of the game history including the clues that both you and the opponent spymaster have given, as well as the guesses both teams have made for each turn. You may use this to help understand your field agents' thinking and construct a better clue. You must answer only by calling the submit_clue tool. Do not output prose.",
        tools=[clue_tool]
    )

    agent_card = AgentCard(
        name="spymaster",
        description='Spymaster in a game of Codenames.',
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()