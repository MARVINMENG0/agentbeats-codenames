import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, default=9009, help="Port to bind the server"
    )
    parser.add_argument(
        "--card-url", type=str, help="URL to advertise in the agent card"
    )
    args = parser.parse_args()

    role = os.getenv("ROLE", None)
    if role is None:
        raise ValueError("ROLE environment variable must be set to 'green' or 'white'.")

# lazy importing so API keys are only needed for the chosen role
    if role == "green":
        from . import green
        green.start(host=args.host, port=args.port, card_url=args.card_url)
    elif role == "spymaster":
        from . import white_spymaster
        white_spymaster.start(host=args.host, port=args.port, card_url=args.card_url)
    else:
        from . import white_field_agent
        white_field_agent.start(host=args.host, port=args.port, card_url=args.card_url)
        
if __name__ == "__main__":
    main()