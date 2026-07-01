"""
agents/supervisor.py
 
===============================================================================
                                SynapseAI
===============================================================================
 
Supervisor Agent
 
The Supervisor orchestrates the entire AutoML workflow.
 
Responsibilities
----------------
✓ Decide which agent executes next
✓ Handle retries
✓ Detect workflow completion
✓ Recover from failures
✓ Log important decisions
✓ Maintain execution order
 
===============================================================================
"""
 
from __future__ import annotations
 
from agents.base_agent import BaseAgent
 
 
class SupervisorAgent(BaseAgent):
    """
    Central orchestrator for SynapseAI.
    """
 
    PHASE_ROUTING = {
        "START": "EDAAgent",
        "EDA": "EDAAgent",
        "PREPROCESSING": "PrepAgent",
        "FEATURE_ENGINEERING": "FeatureAgent",
        "MODELING": "ModelAgent",
        "CRITIC": "CriticAgent",
        "REPORT": "ReportAgent",
        "FINISHED": None,
    }
 
    def run(self):
 
        self.log("Supervisor evaluating workflow")
 
        # ----------------------------------------------------
        # Check for workflow completion
        # ----------------------------------------------------
 
        if self.state.completed:
 
            self.log("Workflow already completed.")
 
            return self.state
 
        # ----------------------------------------------------
        # Retry handling
        # ----------------------------------------------------
 
        if self.state.retry_required:
 
            if self.retry_allowed():
 
                self.log(
                    f"Retrying phase: {self.state.current_phase}"
                )
 
                self.state.retry_required = False
 
            else:
 
                self.error(
                    "Maximum retry limit exceeded."
                )
 
                self.state.mark_completed()
 
                return self.state
 
        # ----------------------------------------------------
        # Determine next agent
        # ----------------------------------------------------
 
        phase = self.state.current_phase
 
        next_agent = self.PHASE_ROUTING.get(phase)
 
        if next_agent is None:
 
            self.log("Workflow finished.")
 
            self.state.mark_completed()
 
            return self.state
 
        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------
 
        self.state.next_agent(next_agent)
 
        self.state.log_decision(
            agent=self.name,
            decision=f"Routing execution to {next_agent}",
        )
 
        self.log(f"Next agent: {next_agent}")
 
        return self.state