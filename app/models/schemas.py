from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class TestType(str, Enum):
    ability = "A"
    biodata_sjt = "B"
    competencies = "C"
    development = "D"
    exercises = "E"
    knowledge = "K"
    personality = "P"
    simulations = "S"


class Intent(str, Enum):
    clarify_needed = "clarify_needed"
    recommend = "recommend"
    refine = "refine"
    compare = "compare"
    off_topic = "off_topic"
    injection = "injection"


class HiringContext(BaseModel):
    """
    Structured version of whatever the user has told us so far. Everything
    downstream (retriever, reranker, formatter) works off this instead of
    raw chat text - keeps the rest of the pipeline dumb and predictable.
    """
    role: Optional[str] = None
    seniority: Optional[str] = None
    experience_years: Optional[str] = None
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    personality_required: Optional[bool] = None
    cognitive_required: Optional[bool] = None
    assessment_type_preference: List[TestType] = Field(default_factory=list)
    max_duration_minutes: Optional[int] = None
    language: Optional[str] = None
    industry: Optional[str] = None
    remote_required: Optional[bool] = None
    other_constraints: List[str] = Field(default_factory=list)

    intent: Intent = Intent.clarify_needed
    comparison_targets: List[str] = Field(default_factory=list)
    has_enough_context: bool = False
    missing_info: List[str] = Field(default_factory=list)
    reasoning: str = ""

    def to_query_text(self) -> str:
        """Flatten into a single string for the retriever to embed."""
        parts = []
        if self.role:
            parts.append(self.role)
        if self.seniority:
            parts.append(self.seniority)
        parts.extend(self.technical_skills)
        parts.extend(self.soft_skills)
        if self.personality_required:
            parts.append("personality behavior assessment")
        if self.cognitive_required:
            parts.append("cognitive ability reasoning assessment")
        parts.extend(t.name for t in self.assessment_type_preference)
        parts.extend(self.other_constraints)
        return " ".join(parts) if parts else (self.role or "")


class CatalogItem(BaseModel):
    id: str
    name: str
    url: str
    test_type: List[TestType]
    description: str
    job_levels: List[str] = Field(default_factory=list)
    remote_testing: bool = False
    adaptive_irt: bool = False
    duration_minutes: Optional[int] = None
