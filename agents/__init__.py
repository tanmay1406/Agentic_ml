"""
agents/__init__.py

Agent Registry

Every new agent should be registered here.

This keeps main.py clean and allows the Supervisor
to instantiate agents dynamically.

IMPORTANT: SupervisorAgent MUST be registered — main.py's
loop starts with current_agent = "Supervisor".
"""

from agents.eda_agent import EDAAgent
from agents.supervisor import SupervisorAgent
from agents.prep_agent import PrepAgent
from agents.feature_agent import FeatureAgent
from agents.critic_agent import CriticAgent   # BUG FIX: was 'synapseai.agents.critic_agent'
from agents.model_agent import ModelAgent     # BUG FIX: was 'synapseai.agents.model_agent'
from agents.report_agent import ReportAgent  # BUG FIX: was 'synapseai.agents.report_agent'


AGENT_REGISTRY = {

    # Supervisor must always be present
    "Supervisor": SupervisorAgent,

    "EDAAgent": EDAAgent,

    "PrepAgent": PrepAgent,

    "FeatureAgent": FeatureAgent,

    "ModelAgent": ModelAgent,

    "CriticAgent": CriticAgent,

    "ReportAgent": ReportAgent,
}