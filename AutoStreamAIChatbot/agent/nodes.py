import os
import re
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

from agent.state import AgentState, LeadProfile, IntentClassification
from agent.prompts import (
    SYSTEM_PROMPT,
    INTENT_PROMPT,
    RAG_PROMPT,
    LEAD_COLLECTION_PROMPT,
)
from agent.tools import retrieve_knowledge, mock_lead_capture
from memory.db import init_db, save_lead

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)

intent_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY"),
).with_structured_output(IntentClassification)


def _build_history(messages: list, n: int = 6) -> str:
    """Return last n messages as a formatted conversation string."""
    recent = messages[-n:] if len(messages) > n else messages
    lines = []
    for m in recent:
        role = "User" if isinstance(m, HumanMessage) else "Aaru"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def _last_human(messages: list) -> str:
    """Return the content of the most recent HumanMessage."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def _extract_field(text: str, field: str) -> str | None:
    """
    Simple regex extractor for name, email, and platform from user text.
    Returns the extracted value or None.
    """
    if field == "email":
        match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
        return match.group(0) if match else None

    if field == "platform":
        platforms = ["youtube", "instagram", "tiktok", "twitter", "facebook",
                     "twitch", "linkedin", "snapchat", "pinterest", "reddit"]
        text_lower = text.lower()
        for p in platforms:
            if p in text_lower:
                return p.capitalize()
        return None

    if field == "name":
        patterns = [
            r"(?:i'?m|i am|my name is|it'?s|call me|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    return None



def session_init_node(state: AgentState) -> dict:
    """
    Initialise session on first turn. Generates session_id, sets up
    SQLite tables, and resets all counters.
    """
    init_db()
    session_id = state.get("session_id") or str(uuid.uuid4())
    return {
        "session_id": session_id,
        "turn_count": state.get("turn_count", 0),
        "lead_captured": state.get("lead_captured", False),
        "intent_history": state.get("intent_history", []),
        "lead_profile": state.get("lead_profile", LeadProfile()),
        "current_node": "session_init_node",
        "last_intent": state.get("last_intent"),
        "last_confidence": state.get("last_confidence"),
        "last_reasoning": state.get("last_reasoning"),
    }


def intent_classifier_node(state: AgentState) -> dict:
    """
    Classify current user message intent using Gemini structured output.
    Pre-fills lead_profile.platform if a platform is detected early.
    """
    messages = state.get("messages", [])
    user_message = _last_human(messages)
    conversation_history = _build_history(messages)

    prompt = INTENT_PROMPT.format(
        conversation_history=conversation_history,
        user_message=user_message,
    )

    result: IntentClassification = intent_llm.invoke(prompt)

    profile: LeadProfile = state.get("lead_profile", LeadProfile())

    if result.detected_platform and not profile.platform:
        profile = profile.model_copy(update={"platform": result.detected_platform})

    if result.detected_plan_interest:
        profile = profile.model_copy(update={"plan_interest": result.detected_plan_interest})

    intent_history = list(state.get("intent_history", []))
    intent_history.append(result.intent)

    return {
        "lead_profile": profile,
        "intent_history": intent_history,
        "last_intent": result.intent,
        "last_confidence": result.confidence,
        "last_reasoning": result.reasoning,
        "current_node": "intent_classifier_node",
    }

def greeter_node(state: AgentState) -> dict:
    greeting_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "The user has just greeted you. Respond warmly, introduce yourself as Aaru, "
        "and mention in one sentence what AutoStream does for content creators. "
        "Keep it to 2-3 sentences max."
    )
    response = llm.invoke(greeting_prompt)
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=response.content))
    return {
        "messages": messages,
        "turn_count": state.get("turn_count", 0) + 1,
        "current_node": "greeter_node",
        "last_intent": state.get("last_intent"),
        "last_confidence": state.get("last_confidence"),
        "last_reasoning": state.get("last_reasoning"),
    }


def rag_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    user_message = _last_human(messages)

    rag_context = retrieve_knowledge.invoke({"query": user_message})

    return {
        "rag_context": rag_context,
        "current_node": "rag_node",
        "last_intent": state.get("last_intent"),
        "last_confidence": state.get("last_confidence"),
        "last_reasoning": state.get("last_reasoning"),
    }


def response_generator_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    user_message = _last_human(messages)
    conversation_history = _build_history(messages)
    rag_context = state.get("rag_context", "No context retrieved.")

    prompt = RAG_PROMPT.format(
        rag_context=rag_context,
        conversation_history=conversation_history,
        user_message=user_message,
    )

    response = llm.invoke(prompt)
    messages = list(messages)
    messages.append(AIMessage(content=response.content))

    return {
        "messages": messages,
        "turn_count": state.get("turn_count", 0) + 1,
        "current_node": "response_generator_node",
        "last_intent": state.get("last_intent"),
        "last_confidence": state.get("last_confidence"),
        "last_reasoning": state.get("last_reasoning"),
    }

def lead_collector_node(state: AgentState) -> dict:
    messages = state.get("messages", [])
    user_message = _last_human(messages)
    conversation_history = _build_history(messages)
    profile: LeadProfile = state.get("lead_profile", LeadProfile())

    updates: dict = {}

    if not profile.name:
        extracted_name = _extract_field(user_message, "name")
        if extracted_name:
            updates["name"] = extracted_name

    if not profile.email:
        extracted_email = _extract_field(user_message, "email")
        if extracted_email:
            updates["email"] = extracted_email

    if not profile.platform:
        extracted_platform = _extract_field(user_message, "platform")
        if extracted_platform:
            updates["platform"] = extracted_platform

    if updates:
        profile = profile.model_copy(update=updates)


    if profile.is_complete:
        profile = profile.model_copy(update={"collection_stage": "complete"})
        return {
            "lead_profile": profile,
            "turn_count": state.get("turn_count", 0) + 1,
            "current_node": "lead_collector_node",
            "last_intent": state.get("last_intent"),
            "last_confidence": state.get("last_confidence"),
            "last_reasoning": state.get("last_reasoning"),
        }

    profile = profile.model_copy(update={"collection_stage": "collecting"})

    prompt = LEAD_COLLECTION_PROMPT.format(
        missing_fields=", ".join(profile.missing_fields),
        name=profile.name or "Not yet collected",
        email=profile.email or "Not yet collected",
        platform=profile.platform or "Not yet collected",
        conversation_history=conversation_history,
        user_message=user_message,
    )

    response = llm.invoke(prompt)
    messages = list(messages)
    messages.append(AIMessage(content=response.content))

    return {
        "messages": messages,
        "lead_profile": profile,
        "turn_count": state.get("turn_count", 0) + 1,
        "current_node": "lead_collector_node",
        "last_intent": state.get("last_intent"),
        "last_confidence": state.get("last_confidence"),
        "last_reasoning": state.get("last_reasoning"),
    }


def tool_executor_node(state: AgentState) -> dict:
    if state.get("lead_captured"):
        return {}
    profile: LeadProfile = state.get("lead_profile")
    if not profile or not profile.is_complete:
        return {}

    if not (profile.name and profile.email and profile.platform):
        return {}
    mock_lead_capture.invoke({
        "name": profile.name,
        "email": profile.email,
        "platform": profile.platform,
    })

    save_lead(
        state["session_id"],
        profile.name,
        profile.email,
        profile.platform,
        profile.plan_interest,
    )
    
    print(f"Lead captured successfully: {profile.name}, {profile.email}, {profile.platform}")

    confirmation = (
        f"You're all set, {profile.name}! I've got everything I need. "
        f"Welcome to AutoStream — our team will be in touch at "
        f"{profile.email} shortly. Excited to have you on board!"
    )

    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=confirmation))

    return {
        "lead_captured": True,
        "messages": messages,
        "turn_count": state.get("turn_count", 0) + 1,
        "current_node": "tool_executor_node",
        "last_intent": "hard_lead",
        "last_confidence": state.get("last_confidence"),
        "last_reasoning": state.get("last_reasoning"),
    }
