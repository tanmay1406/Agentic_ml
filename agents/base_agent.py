"""
agents/base_agent.py

===============================================================================
                                SynapseAI
===============================================================================

Abstract base class for every autonomous agent.

All agents inherit from BaseAgent and automatically gain:

✓ Shared AgentState access
✓ LLM communication
✓ Secure code execution
✓ Logging
✓ Error handling
✓ Retry support

===============================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from core.state import AgentState
from core.sandbox import Sandbox, ExecutionResult
from core.llm_client import LLMClient


class BaseAgent(ABC):
    """
    Base class for every SynapseAI agent.
    """

    def __init__(
        self,
        state: AgentState,
        llm: LLMClient,
        sandbox: Sandbox,
    ):

        self.state = state
        self.llm = llm
        self.sandbox = sandbox

        self.name = self.__class__.__name__

    # ==========================================================
    # Main Entry Point
    # ==========================================================

    @abstractmethod
    def run(self) -> AgentState:
        """
        Execute this agent.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    # ==========================================================
    # LLM Helpers
    # ==========================================================

    def ask_llm(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        """
        Generate a normal text response.
        """

        self.log(f"Sending prompt to {self.llm.model}")

        return self.llm.generate(
            prompt,
            system_prompt,
        )

    def generate_code(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        """
        Generate executable Python code.
        """

        self.log("Generating Python code")

        return self.llm.generate_code(
            prompt,
            system_prompt,
        )

    # ==========================================================
    # Sandbox
    # ==========================================================

    def execute_code(
        self,
        code: str,
    ) -> ExecutionResult:
        """
        Execute Python code safely.
        """

        self.log("Executing generated code")

        return self.sandbox.execute(code)

    # ==========================================================
    # Logging
    # ==========================================================

    def log(
        self,
        message: str,
    ) -> None:

        self.state.log_execution(
            agent=self.name,
            action=message,
            status="INFO",
        )

    def warning(
        self,
        message: str,
    ) -> None:

        self.state.warnings.append(
            f"[{self.name}] {message}"
        )

    def error(
        self,
        message: str,
    ) -> None:

        self.state.log_error(message)

    # ==========================================================
    # Workflow Helpers
    # ==========================================================

    def set_phase(
        self,
        phase: str,
    ):

        self.state.next_phase(phase)

    def complete(self):

        self.state.next_agent("Supervisor")

    # ==========================================================
    # Retry
    # ==========================================================

    def retry_allowed(self) -> bool:

        return self.state.increment_retry()

    def reset_retry(self):

        self.state.reset_retry()