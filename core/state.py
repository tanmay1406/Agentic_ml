"""
core/state.py

===============================================================================
                                  SynapseAI
===============================================================================

Centralized shared memory (Blackboard Architecture) for the SynapseAI
multi-agent system.

Every agent (Supervisor, EDA, Data Preparation, Modeling, Critic, etc.)
reads from and writes to this shared state.

The AgentState acts as the single source of truth throughout the entire
machine learning workflow.

Architecture:

                ┌─────────────────────┐
                │     Supervisor      │
                └──────────┬──────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     EDA Agent      Prep Agent      Model Agent
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    Critic Agent
                           │
                           ▼
                    Shared AgentState

Features
--------
• Centralized shared memory (Blackboard Architecture)
• Type-safe state management using Pydantic
• Automatic execution and decision logging
• Model leaderboard tracking
• Retry protection against infinite agent loops
• Thread-safe updates
• Persistent checkpointing (save/load)
• Production-ready serialization

Every agent communicates only through AgentState, making SynapseAI
modular, extensible, and easy to debug.
===============================================================================
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from pydantic import PrivateAttr
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# Helper Models
# ============================================================


class CandidateModel(BaseModel):
    """Stores information about one trained model."""

    name: str
    algorithm: str
    score: float
    metrics: Dict[str, float] = Field(default_factory=dict)
    model_path: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_time: Optional[float] = None


class ExecutionLog(BaseModel):
    """Stores execution history of every agent."""

    timestamp: str
    agent: str
    action: str
    status: str
    output: Optional[str] = None   # FIX: was silently missing — log_execution
    error: Optional[str] = None    #      was passing 'details' which has no
                                   #      matching field here, so it was dropped


class DecisionLog(BaseModel):
    """Stores important decisions made by agents."""

    timestamp: str
    agent: str
    decision: str


class ErrorLog(BaseModel):
    """Stores runtime errors."""

    timestamp: str
    message: str


# ============================================================
# Agent State
# ============================================================


class AgentState(BaseModel):
    """
    Shared memory for the SynapseAI multi-agent platform.

    AgentState serves as the communication hub between all autonomous
    agents. Each agent reads from the current state, performs its task,
    and writes its outputs back into the shared state.

    Workflow:

        User
          │
          ▼
      Supervisor Agent
          │
          ▼
      EDA Agent
          │
          ▼
      Data Preparation Agent
          │
          ▼
      Feature Engineering Agent
          │
          ▼
      Model Training Agent
          │
          ▼
      Critic Agent
          │
          ▼
      Supervisor Agent
          │
      Retry / Finish

    This implementation follows the Blackboard Architecture pattern,
    allowing agents to collaborate without direct dependencies.
    """

    model_config = ConfigDict(validate_assignment=True)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_path: str
    target_column: str

    dataset_name: Optional[str] = None
    dataset_shape: Optional[List[int]] = None

    problem_type: Optional[str] = None
    feature_types: Dict[str, str] = Field(default_factory=dict)
    class_distribution: Dict[str, int] = Field(default_factory=dict)

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    eda_report: Optional[str] = None
    eda_summary: Dict[str, Any] = Field(default_factory=dict)

    data_quality_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    # --------------------------------------------------------
    # Data Preparation
    # --------------------------------------------------------

    cleaned_dataset_path: Optional[str] = None

    preprocessing_pipeline: List[str] = Field(default_factory=list)

    preprocessing_report: Optional[str] = None

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    feature_dataset_path: Optional[str] = None

    feature_report: Optional[str] = None

    feature_engineering_steps: List[str] = Field(default_factory=list)

    selected_features: List[str] = Field(default_factory=list)

    # --------------------------------------------------------
    # Modeling
    # --------------------------------------------------------

    candidate_models: List[CandidateModel] = Field(default_factory=list)

    leaderboard: List[CandidateModel] = Field(default_factory=list)

    best_model: Optional[str] = None

    best_model_path: Optional[str] = None

    leaderboard_path: Optional[str] = None

    metrics_path: Optional[str] = None

    model_report: Optional[str] = None
    # reporting
    report_path: Optional[str] = None

    summary_path: Optional[str] = None

    best_score: Optional[float] = None

    # --------------------------------------------------------
    # Critic
    # --------------------------------------------------------

    critic_feedback: List[str] = Field(default_factory=list)

    overfitting_detected: bool = False

    retry_required: bool = False

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    execution_history: List[ExecutionLog] = Field(default_factory=list)

    decisions: List[DecisionLog] = Field(default_factory=list)

    errors: List[ErrorLog] = Field(default_factory=list)

    warnings: List[str] = Field(default_factory=list)

    generated_code: List[str] = Field(default_factory=list)

    sandbox_logs: List[str] = Field(default_factory=list)

    # --------------------------------------------------------
    # Workflow
    # --------------------------------------------------------

    current_phase: str = "EDA"

    current_agent: str = "Supervisor"

    completed: bool = False

    retry_count: int = 0

    max_retries: int = 3

    # FIX: datetime.utcnow() is deprecated in Python 3.12+
    #      Use datetime.now(timezone.utc) instead
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    workflow_id: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    )

    # --------------------------------------------------------
    # Internal (not serialized)
    # --------------------------------------------------------

    _lock: Lock = PrivateAttr(default_factory=Lock)

    # ========================================================
    # Generic Functions
    # ========================================================

    def update(self, field: str, value: Any) -> None:
        """Thread-safe update of any state field."""

        with self._lock:

            if not hasattr(self, field):
                raise AttributeError(f"Unknown field '{field}'")

            setattr(self, field, value)

    def get(self, field: str) -> Any:
        """Retrieve any field."""

        return getattr(self, field)

    # ========================================================
    # Logging
    # ========================================================

    def log_execution(
        self,
        agent: str,
        action: str,
        status: str,
        details: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:

        with self._lock:

            self.execution_history.append(
                ExecutionLog(
                    timestamp=datetime.now(IST).isoformat(),
                    agent=agent,
                    action=action,
                    status=status,
                    output=details,   # FIX: was `details=details` — ExecutionLog
                    error=error,      #      has no 'details' field, only 'output'
                )                     #      so the value was silently dropped
            )

    def log_decision(self, agent: str, decision: str) -> None:

        with self._lock:

            self.decisions.append(
                DecisionLog(
                    timestamp=datetime.now(IST).isoformat(),
                    agent=agent,
                    decision=decision,
                )
            )

    def log_error(self, message: str) -> None:

        with self._lock:

            self.errors.append(
                ErrorLog(
                    timestamp=datetime.now(IST).isoformat(),
                    message=message,
                )
            )

    # ========================================================
    # Model Management
    # ========================================================

    def add_candidate_model(self, model: CandidateModel) -> None:
        """Add a trained model."""

        with self._lock:

            self.candidate_models.append(model)

    def update_leaderboard(self, model: CandidateModel) -> None:
        """Update leaderboard and automatically select the best model."""

        with self._lock:

            self.leaderboard.append(model)

            self.leaderboard.sort(
                key=lambda x: x.score,
                reverse=True,
            )

            best = self.leaderboard[0]

            self.best_model = best.name
            if best.model_path is not None:
                self.best_model_path = best.model_path
            self.best_score = best.score

    # ========================================================
    # Retry Control
    # ========================================================

    def increment_retry(self) -> bool:
        """
        Increase retry count.

        Returns
        -------
        True
            Retry is allowed.

        False
            Maximum retry count exceeded.
        """

        with self._lock:

            self.retry_count += 1

            return self.retry_count <= self.max_retries

    def reset_retry(self) -> None:

        with self._lock:

            self.retry_count = 0

    # ========================================================
    # Workflow
    # ========================================================

    def next_phase(self, phase: str):

        self.current_phase = phase

    def next_agent(self, agent: str):

        self.current_agent = agent

    def mark_completed(self):

        self.completed = True

    # ========================================================
    # Persistence
    # ========================================================

    def save(self, path: str = "workspace/state.json") -> None:
        """Save state as JSON."""

        path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            self.model_dump_json(indent=4),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str):
        """Restore state from JSON."""

        return cls.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )

    # ========================================================
    # Utilities
    # ========================================================

    def summary(self) -> Dict[str, Any]:
        """Return a lightweight summary."""

        return {
            "dataset": self.dataset_name,
            "phase": self.current_phase,
            "agent": self.current_agent,
            "best_model": self.best_model,
            "best_score": self.best_score,
            "retry_count": self.retry_count,
            "completed": self.completed,
        }