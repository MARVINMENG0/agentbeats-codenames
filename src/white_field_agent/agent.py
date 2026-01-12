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
from google.adk.tools import FunctionTool
from typing import Literal

class Guess(BaseModel):
    word: str
    explanation: str
    action: Literal["guess", "pass"]

def submit_guess(word: str, explanation: str, action: str) -> Guess:
    return Guess(word=word, explanation=explanation, action=action)

guess_tool = FunctionTool(func=submit_guess)

def main():
    parser = argparse.ArgumentParser(description="Run the A2A debater agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--team", type=str, help="red or blue team in codenames")
    args = parser.parse_args()

    root_agent = Agent(
        name="field_agent",
        model="gemini-2.0-flash",
        description="Participates in a debate.",
        instruction=f"You are a field agent in the game of codenames. You will be on the {args.team} team. You will receive a list of possible words to choose from. You will receive a clue from your spymaster in the form of a single word and a single number. The number indicates the number of words their clue relates to. The word will relate to that number of words in the word list you are given. Your task is to use the clue and the game history to guess one of the words from the word list that belongs to your team. You will also be given a game log consisting of the clues that your spymaster and the opponent spymaster have given, as well as the guesses you and the opponent field agents' have made. Provide a single word as an answer, as well as an explanation of why you chose that word. You must answer only by calling the submit_guess tool. Do not output prose.",
        tools=[guess_tool]
    )

    agent_card = AgentCard(
        name="field_agent",
        description='Participates in a debate.',
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