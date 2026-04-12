"""
main.py — CLI fallback for testing the AutoStream Aaru agent
without starting the FastAPI server.
"""
import uuid
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import compiled_graph
from agent.state import LeadProfile

load_dotenv()


def run_cli():
    """Interactive CLI session with Aaru."""
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    print("\n" + "═" * 60)
    print("  AutoStream · Aaru Agent  (CLI Mode)")
    print("  Session:", session_id[:8] + "...")
    print("  Type 'quit' or 'exit' to end the session.")
    print("═" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nAaru: Thanks for chatting — hope to see you on AutoStream soon! 👋")
            break

        input_state = {
            "messages": [HumanMessage(content=user_input)],
            "session_id": session_id,
        }

        try:
            final_state = compiled_graph.invoke(input_state, config=config)
        except Exception as e:
            print(f"\n[Error] Agent raised an exception: {e}\n")
            continue

        # Print the last AI message
        ai_response = ""
        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage):
                ai_response = msg.content
                break

        print(f"\nAaru: {ai_response}\n")

        # Print debug metadata
        intent = final_state.get("last_intent", "—")
        confidence = final_state.get("last_confidence", 0.0)
        reasoning = final_state.get("last_reasoning", "—")
        lead_captured = final_state.get("lead_captured", False)
        turn = final_state.get("turn_count", 0)

        profile: LeadProfile = final_state.get("lead_profile", LeadProfile())

        print(f"  [intent={intent} | conf={confidence:.2f} | turn={turn}]")
        print(f"  [reasoning: {reasoning}]")
        if profile.name or profile.email or profile.platform:
            print(f"  [lead profile → name={profile.name} | email={profile.email} | platform={profile.platform}]")
        if lead_captured:
            print("  ✅ Lead captured and saved to SQLite!")
        print()


if __name__ == "__main__":
    run_cli()
