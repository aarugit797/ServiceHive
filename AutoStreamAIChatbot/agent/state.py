from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class IntentClassification(BaseModel):
    intent: Literal[
        "greeting",
        "inquiry_general",
        "inquiry_specific",
        "hard_lead"
    ] = Field(description="Classified intent of the user message")

    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence 0.0-1.0. Below 0.6 triggers clarify_node."
    )

    detected_platform: Optional[str] = Field(
        default=None,
        description=(
            "Any creator platform mentioned anywhere in the message "
            "(YouTube, Instagram, TikTok etc). Extract even if casual."
        )
    )

    detected_plan_interest: Optional[Literal["basic", "pro", "unknown"]] = Field(
        default=None,
        description="Which plan the user seems interested in, if mentioned."
    )

    reasoning: str = Field(
        description="One sentence explaining why this intent was chosen."
    )


class LeadProfile(BaseModel):
    name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    platform: Optional[str] = Field(
        default=None,
        description=(
            "Pre-filled from detected_platform if user mentioned it "
            "in any earlier turn. Never ask again if already set."
        )
    )
    plan_interest: Optional[str] = Field(default=None)
    collection_stage: Literal["not_started", "collecting", "complete"]=Field(default="not_started")

    @property
    def missing_fields(self) -> List[str]:
        missing = []
        if not self.name: missing.append("name")
        if not self.email: missing.append("email")
        if not self.platform: missing.append("platform")
        return missing

    @property
    def is_complete(self) -> bool:
        return len(self.missing_fields) == 0


class AgentState(TypedDict):
    messages: List[BaseMessage]       
    intent_history: List[str]        
    lead_profile: LeadProfile         
    rag_context: Optional[str]        
    current_node: str                 
    session_id: str                  
    turn_count: int
    lead_captured: bool               
    last_intent: Optional[str]        
    last_confidence: Optional[float]  
    last_reasoning: Optional[str]    
