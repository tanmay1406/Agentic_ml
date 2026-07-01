"""
main.py

===============================================================================
                                SynapseAI
===============================================================================

Entry point for the SynapseAI multi-agent AutoML system.

Example
-------

python main.py \
    --dataset workspace/data/raw/iris.csv \
    --target species \
    --model qwen3:8b

===============================================================================
"""

from __future__ import annotations

import argparse

from core.state import AgentState
from core.llm_client import LLMClient
from core.sandbox import Sandbox
from agents import AGENT_REGISTRY


# =============================================================================
# Argument Parser
# =============================================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="SynapseAI - Autonomous Machine Learning System"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset (.csv)",
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name",
    )

    parser.add_argument(
        "--model",
        default="qwen3:8b",
        help="LLM model (default: qwen3:8b)",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum agent steps before aborting (default: 50)",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_arguments()

    print("=" * 70)
    print("  SynapseAI — Autonomous Machine Learning")
    print("=" * 70)
    print(f"  Dataset : {args.dataset}")
    print(f"  Target  : {args.target}")
    print(f"  Model   : {args.model}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Shared Components
    # -------------------------------------------------------------------------

    import os
    dataset_abs_path = os.path.abspath(args.dataset)

    state = AgentState(
        dataset_path=dataset_abs_path,
        target_column=args.target,
    )

    llm = LLMClient(model=args.model)

    sandbox = Sandbox()

    # -------------------------------------------------------------------------
    # Execution Loop
    #
    # Flow:
    #   Supervisor → EDAAgent → Supervisor → PrepAgent →
    #   Supervisor → FeatureAgent → ... → complete
    #
    # The Supervisor reads current_phase and sets current_agent.
    # Each agent does its work and hands control back to the Supervisor.
    # The loop continues until state.completed is True or max_steps hit.
    # -------------------------------------------------------------------------

    step = 0

    while not state.completed:

        step += 1

        if step > args.max_steps:
            print(
                f"\n[Abort] Exceeded {args.max_steps} steps — "
                "possible infinite loop. Saving state."
            )
            break

        current = state.current_agent

        print(f"\n[Step {step}] ── {current}")

        agent_cls = AGENT_REGISTRY.get(current)

        if agent_cls is None:
            # An agent was routed to that isn't registered yet.
            print(
                f"\n[Paused] Agent '{current}' is not yet registered "
                "in AGENT_REGISTRY.\n"
                "Implement and register it to continue the pipeline."
            )
            break

        agent = agent_cls(state, llm, sandbox)

        # BUG FIX: original code had no exception handling.
        # If agent.run() raised an unhandled exception, state.save()
        # was never reached, losing all progress.
        # try/finally guarantees a checkpoint is always written,
        # even on crash, so the run can be resumed or debugged.
        try:
            agent.run()
        except Exception as e:
            print(f"\n[ERROR] {current} raised an unhandled exception: {e}")
            state.log_error(
                f"Unhandled exception in {current}: {e}"
            )
        finally:
            # Always checkpoint — never lose work
            state.save()

        print(
            f"         phase={state.current_phase} | "
            f"next={state.current_agent} | "
            f"retry={state.retry_count}"
        )

    # -------------------------------------------------------------------------
    # Final Report
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)

    if state.completed:
        print("  Workflow completed successfully.")
    else:
        print("  Workflow stopped before completion.")

    print("=" * 70)
    print()

    summary = state.summary()

    for key, value in summary.items():
        print(f"  {key:<15}: {value}")

    print()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    main()