from typing import Any
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    Message, 
    TaskState, 
    Part, 
    TextPart, 
    DataPart,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from a2a.utils import get_message_text, new_agent_text_message
from google.adk.tools import FunctionTool

from messenger import Messenger

import json
import logging
import asyncio
import nest_asyncio
import random
import string
from typing import List, Dict, Any
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# from agentbeats.green_executor import GreenAgent, GreenExecutor
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codenames_evaluator")
nest_asyncio.apply()


# ---------- Data Models ----------


class Clue(BaseModel):
    word: str
    number: int

def submit_clue(word: str, number: int) -> Clue:
    return Clue(word=word, number=number)

clue_tool = FunctionTool(func=submit_clue)


class Guess(BaseModel):
    word: str
    explanation: str
    action: str  # "guess" or "pass"

def submit_guess(word: str, explanation: str, action: str) -> Guess:
    return Guess(word=word, explanation=explanation, action=action)

guess_tool = FunctionTool(func=submit_guess)

class GameState(BaseModel):
    board: List[str]
    revealed: Dict[str, str]  # word -> type revealed
    clue_history: List[Dict[str, Any]]
    guess_history: List[Dict[str, Any]]
    remaining_red: int
    remaining_blue: int
    assassin_hit: bool = False

class SpymasterGameEval(BaseModel):
    spymasterClues: int
    averageWordsPerClue: float
    clueRelevance: float
    clueRiskScore: int

class FieldAgentGameEval(BaseModel):
    fieldAgentGuesses: int
    fieldAgentAccuracy: int
    misinterpretationScore: int # (neutral guess + 2 * other team guess + 10 * assassin guess)
    


# ---------- Board Generation ----------

def generate_board():
    with open("words.txt", "r") as f:
        words = [line.strip() for line in f if line.strip()]
    board = random.sample(words, 25)

    assignments = (
        ["red"] * 9 +
        ["blue"] * 8 +
        ["neutral"] * 7 +
        ["assassin"]
    )
    random.shuffle(assignments)

    key = {}
    for i, word in enumerate(board):
        if assignments[i] == "assassin":
            assassin_word = word
        key[word] = assignments[i]

    return key, assassin_word



class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


# ---------- Core Codenames Game Logic ----------

class Agent: # GreenAgent
    # Fill in: list of required config keys, e.g. ["topic", "num_rounds"]
    required_config_keys: list[str] = []

    def __init__(self, spymaster_url, guesser_url, max_turns=12):
        self.messenger = Messenger()
        self._required_roles = ["red_spymaster", "red_field_agent", "blue_spymaster", "blue_field_agent"]  # The purple agent being tested
        self._required_config_keys = ["domain"]
        # Initialize other state here

        self.spymaster_url = spymaster_url
        self.guesser_url = guesser_url
        self.max_turns = max_turns

        self.board, self.assassin_word = generate_board()

        self.revealed = {}
        self.turn_number = 1

        self.remaining_red = sum(1 for k in self.key.values() if k == "red")
        self.remaining_blue = sum(1 for k in self.key.values() if k == "blue")
        self.red_neutrals = 0
        self.red_blues = 0
        self.blue_neutrals = 0
        self.blue_reds = 0

        self.clue_history = []
        self.guess_history = []
        self.red_guess_assassin = False
        self.blue_guess_assassin = False


    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Add additional request validation here

        return True, "ok"
        
    def check_end_conditions(self):
        if self.red_guess_assassin or self.blue_guess_assassin:
            return True
        if self.remaining_red == 0 or self.remaining_blue == 0:
            return True
        if self.turn_number >= self.max_turns:
            return True
        return False

    def score(self):
        if self.assassin_hit:
            return {
            "score": -1,
            "assassin": self.assassin_hit,
            "remaining_red": self.remaining_red,
            "remaining_blue": self.remaining_blue,
            "turn_number": self.turn_number,
        }
        score = 0
        score += (9 - self.remaining_red)
        score -= len([w for w,v in self.revealed.items() if v == "blue"])

        return {
            "score": score,
            "assassin": self.assassin_hit,
            "remaining_red": self.remaining_red,
            "remaining_blue": self.remaining_blue,
            "turn_number": self.turn_number,
        }

    async def orchestrate_codenames_game(
        self,
        updater: TaskUpdater,
    ) -> dict[str, list[str]]:
        logger.info("Beginning codenames game")

        game_log: dict[str, dict[int, list[str]] | list[str]] = {"red_spymaster": [], "red_field_agent": [{}], "blue_spymaster": [], "blue_field_agent": [{}],
                                          "revealed_red": [], "revealed_blue": [], "revealed_neutral":[]}

        async def spymaster_turn(role: str, prompt: str) -> str:
            response = await self.messenger.talk_to_agent(prompt, str(participants[role]), new_conversation=False)
            logger.info(f"{role}: {response}")
            game_log[role].append(response)
            await updater.update_status(TaskState.working, new_agent_text_message(f"{role}: {response}"))
            return response
        
        async def field_agent_turn(role: str, prompt: str, team: str) -> str:
            logger.info(f"{role} starting turn.")
            response = Guess(action="guess")
            guess_counter = 1
            while response.action != "pass" and guess_counter <= num_turns: ### <= because if you reach the end number of guesses without failing you get a bonus guess.
                logger.info(f"Guess number {guess_counter}")
                ### Update the prompt with the new game log each guess so the field_agent has updated information on what happened with their previous guess
                working_prompt = prompt.replace("<game_log>", json.dumps(game_log, ensure_ascii=False))

                ### Passing logic
                if guess_counter == 1:
                    prompt += "This is the first guess on this turn, so you must return a guess. You may not pass."
                else:
                    prompt += "This is not the first guess on this turn, so you may either return a guess or pass."
                response = await self.messenger.talk_to_agent(prompt, str(participants[role]), new_conversation=False)
                logger.info(f"{role}: {response}")
                ### Add guess to game_log
                if self.turn_number not in game_log[role].keys():
                    game_log[role][self.turn_number] = [response]
                else:
                    game_log[role][self.turn_number].append(response)
                
                game_log[self.board[response.word]].append(response)
                if self.board[response.word] == "red":
                    self.red_remaining -= 1
                elif self.board[response.word] == "blue":
                    self.blue_remaining -= 1
                
                ### Logic if you guessed a word not on your team
                if response.word == self.assassin_word and team == "red":
                    self.red_guess_assassin = True
                    return response
                elif response.word == self.assassin_word and team == "blue":
                    self.blue_guess_assassin = True
                    return response
                elif team == "red" and self.board[response.word] == "blue":
                    self.red_blues += 1
                    return response
                elif team == "red" and self.board[response.word] == "neutral":
                    self.red_neutrals += 1
                    return response
                elif team == "blue" and self.board[response.word] == "red":
                    self.blue_reds += 1
                    return response
                elif team == "blue" and self.board[response.word] == "red":
                    self.blue_neutrals += 1
                    return response

                await updater.update_status(TaskState.working, new_agent_text_message(f"{role}: {response}"))
                guess_counter += 1
            return response

        ### For this example we give red an extra card and have red go first
        while not self.check_end_conditions():
            red_clue = await spymaster_turn("red_spymaster", f"You are the red spymaster. Here are the words and their statuses: {self.board}. Give a single word and a number as a clue. The word should relate to some of the words of your color in the wordlist. The number indicates the number of words that your clue relates to. Here is the game log history:\n\n{game_log}\n\nThe game log includes your previous clues, the other team's previous clues, each team's guesses, the turns the guesses were made, and the status of revealed words. Give a clue consisting of one word and one number. Return your answer with the submit_clue tool. Do not output prose.")
            game_log["red_spymaster"].append(red_clue)

            red_response = await field_agent_turn("red_field_agent", f"Here is the clue you are given: {red_clue.word}. Here are the number of words it relates to: {red_clue.number}. Here is the list of possible words the clue relates to: {self.board.keys()}. Here is the game log history:\n\n <game_log> \n\nThe game log includes your previous clues, the other team's previous clues, each team's guesses, the turns the guesses were made, and the status of revealed words. Using the clue, guess at least one of the unrevealed words on the board. You must answer only by calling the submit_guess tool. Do not output prose.")
            # game_log['red_field_agent'][turn_counter] = red_response

            blue_clue = await spymaster_turn("blue_spymaster", f"Here are the words and their statuses: {self.board}. Give a single word and a number as a clue. The word should relate to some of the words of your color in the wordlist. The number indicates the number of words that your clue relates to. Here is the game log history: {game_log}, which includes your previous clues, the other team's previous clues, each team's guesses, the turns the guesses were made, and the status of revealed words.")
            game_log["blue_spymaster"].append(blue_clue)

            blue_response = await field_agent_turn("blue_field_agent", f"Here is the clue you are given: {blue_clue}. Here is the list of possible words the clue relates to: {self.board.keys()}. Here is the game log history {game_log}. which includes your previous clues, the other team's previous clues, each team's guesses, the turns the guesses were made, and the status of revealed words. Using the clue, guess at least one of the unrevealed words on the board.")
            # game_log['blue_field_agent'][turn_counter] = blue_response

            self.turn_number += 1

        logger.info(f"Returning Final Gamelog: {game_log}")
        return game_log

    async def judge_game(self, game_log: dict[str, dict[int, list[str]] | list[str]]):
        # system_prompt = """
        #     You are an impartial judge evaluating the gameplay of TWO competing teams in the board game Codenames.

        #     Each team consists of:
        #     - one Spymaster
        #     - one Field Agent

        #     You will be given:
        #     - the full Codenames board with color assignments
        #     - clues produced by each Spymaster
        #     - guesses produced by each Field Agent
        #     - explanations accompanying each Field Agent guess
        #     - the sequence of turns and outcomes

        #     Your task is NOT merely to decide who won. Your task is to evaluate the QUALITY OF PLAY for both teams individually and comparatively.

        #     ----------------------------------------
        #     EVALUATE EACH SPYMASTER SEPARATELY ON:
        #     ----------------------------------------

        #     1) Clue Relevance
        #     - How well does each clue relate to the intended team-colored words?
        #     - Are semantic relationships reasonable?

        #     2) Clue Efficiency
        #     - Do clues successfully group multiple target words?
        #     - Does the number provided reflect realistic grouping potential?

        #     3) Clue Safety
        #     - Did clues risk indicating opponent words, neutral words, or the assassin word?
        #     - Was the risk justified or careless?

        #     4) Strategic Planning
        #     - Evidence of grouping related words together?
        #     - Evidence of adaptation based on prior guesses and game history?
        #     - Evidence of planning beyond the current turn?

        #     ----------------------------------------
        #     EVALUATE EACH FIELD AGENT SEPARATELY ON:
        #     ----------------------------------------

        #     1) Interpretation Accuracy
        #     - Do guesses align with the intended meaning of the clue?
        #     - If incorrect, were they still sensible guesses?

        #     2) Explanation Quality
        #     - Does the explanation demonstrate reasoning based on clue semantics, board state, and history?
        #     - Is the logic internally consistent with the guess chosen?

        #     3) Strategic Reasoning
        #     - Does the agent avoid risky or assassin-adjacent words?
        #     - Do they consider multiple possibilities?
        #     - Do they reasonably decide when to stop guessing?

        #     4) Team Coordination
        #     - Are guesses consistent with the Spymaster’s apparent strategy?
        #     - Do explanations reflect shared understanding between teammates?

        #     ----------------------------------------
        #     CROSS-TEAM COMPARISON
        #     ----------------------------------------

        #     Also compare the TWO teams on:

        #     - overall strategy strength
        #     - coordination quality
        #     - adaptability over time
        #     - risk management skill

        #     Winning or losing is only one factor. Focus on quality of reasoning and gameplay.

        #     ----------------------------------------
        #     OUTPUT REQUIREMENTS
        #     ----------------------------------------

        #     For EACH TEAM provide:
        #     - short written justification
        #     - numeric scores from 0–10 for:
        #     * Spymaster performance
        #     * Field Agent performance
        #     * team coordination
        #     * overall gameplay quality

        #     Finally provide:
        #     - overall comparative judgment between teams
        #     """

        #     user_prompt = f"""Evaluate the game performance. The game log is as follows:
        #     {game_log}
        #     Provide a JSON formatted response with scores and comments for each criterion for both debaters.
        #     """
        #     client = genai.getClient()
        #     response = self._client.models.generate_content(
        #         model="gemini-2.5-flash",
        #         config=genai.types.GenerateContentConfig(
        #                 system_instruction=system_prompt,
        #                 response_mime_type="application/json",
        #                 response_schema=DebateEval,
        #             ),
        #         contents=user_prompt,
        #     )
        red_spymaster = game_log["red_spymaster"]
        red_field_agent = game_log["red_field_agent"]
        n_clues_given = len(red_spymaster)
        red_spymaster_scorecard = {
            "n_clues_given": n_clues_given,
            "neutral_guesses": self.red_neutrals,
            "opposite_team_guesses": self.red_blues,
            "assassin_hit": self.red_guess_assassin
        }
        # red_field_agent_scorecard = {

        # }
        return red_spymaster_scorecard



    
    async def run(self, req: EvalRequest, updater: TaskUpdater):
        logger.info(f"Starting codenames orchestration: {req}")

        # Get the purple agent URLs
        red_spymaster_url = str(req.participants["red_spymaster"])
        red_field_agent_url = str(req.participants["red_field_agent"])
        blue_spymaster_url = str(req.participants["blue_spymaster"])
        blue_field_agent_url = str(req.participants["blue_field_agent"])

        game_log = await self.orchestrate_codenames_game(req.participants,
                                                updater)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of {len(tasks)} tasks in {domain} domain")
        )

# class AssessmentRequest(BaseModel):
#     spymaster_url: str
#     guesser_url: str
#     max_turns: int = 20

# # ---------- A2A Assessment Endpoint ----------
# app = FastAPI()
# @app.post("/assess")
# def assess(request: AssessmentRequest):

#     judge = CodenamesJudge(
#         spymaster_url=request.spymaster_url,
#         guesser_url=request.guesser_url,
#         max_turns=request.max_turns
#     )

#     result = judge.run()
#     return result

def codenames_evaluator_agent_card(name: str, url: str) -> AgentCard:
    """Create the agent card for the codenames evaluator."""
    skill = AgentSkill(
        id="codenames_evaluation",
        name="Codenames Benchmark Evaluation",
        description="Evaluates agents on codenames game",
        tags=["benchmark", "evaluation", "codenames"],
        examples=[
            '''{"participants": {
            "red_spymaster": "http://localhost:9019",
            "red_field_agent": "http://localhost:9018",
            "blue_spymaster": "http://localhost:9017",
            "blue_field_agent": "http://localhost:9016",
            "config": {"team": "red", "num_tasks": 5}}'''
        ],
    )
    return AgentCard(
        name=name,
        description="Setup and facilitate a game of codenames between two teams. Each team has one spymaster and one or more field agent. Both teams have access to an ordered list of 25 words. Each spymaster has a list of words they need their field agents to guess. THe spymaster can only give one clue each round, and a number which represents how many of the 25 words the clue relates to. The field agents must then use the clue to guess from the 25 words. ",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

async def main():
    parser = argparse.ArgumentParser(description="Run the Codenames evaluator agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for the agent card")
    args = parser.parse_args()

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"

    # instantiate and run your evaluator
    agent = Agent()
    executor = GreenExecutor(agent)

    ### TODO:
    agent_card = codenames_evaluator_agent_card("CodenamesEvaluator", agent_url)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
    uvicorn_server = uvicorn.Server(uvicorn_config)
    await uvicorn_server.serve()


    # evaluator = GreenAgentEvaluator()
    # evaluator.run()   # whatever your entry point method is


if __name__ == "__main__":
    asyncio.run(main)