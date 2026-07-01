"""
agents/report_agent.py

===============================================================================
                                SynapseAI
===============================================================================

Report Agent — ALL BUGS FIXED IN THIS VERSION

Bugs fixed
----------
Bug 10a  `to_markdown()` requires the `tabulate` library which is not a
         mandatory dependency. Missing tabulate crashed the entire report
         generation with an ImportError. Now falls back gracefully to
         `to_string()` if tabulate is unavailable.

Bug 10b  `save_report()` and `save_summary()` methods existed but were
         never called from `run()`. The run() method duplicated their logic
         inline, leading to two code paths that could diverge. run() now
         delegates to those helpers — single responsibility, no duplication.

Bug 10c  Stale comment in run() said "Add these fields to AgentState if
         they don't exist" for report_path and summary_path. Both fields
         already exist in state.py — the comment was misleading and has
         been removed.

Bug 10d  build_markdown() rendered None as the literal string "None" for
         best_model and best_score when neither had been set (e.g. if the
         modeling stage failed). These now display a human-readable fallback.

===============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from agents.base_agent import BaseAgent


class ReportAgent(BaseAgent):
    """
    Generates the final AutoML report.
    """

    REPORT_DIR = Path("workspace/reports")
    REPORT_FILE = "report.md"
    SUMMARY_FILE = "summary.json"

    def __init__(
        self,
        state,
        llm,
        sandbox,
    ):

        super().__init__(state, llm, sandbox)

        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)

        self.report_path = self.REPORT_DIR / self.REPORT_FILE
        self.summary_path = self.REPORT_DIR / self.SUMMARY_FILE

        self.leaderboard = None
        self.metrics = {}

    # ==========================================================
    # Loading
    # ==========================================================

    def load_artifacts(self):
        """
        Load leaderboard and metrics from disk if available.

        Both checks are defensive — neither leaderboard_path nor
        metrics_path are guaranteed to exist if an upstream agent failed.
        """

        if self.state.leaderboard_path:

            path = Path(self.state.leaderboard_path)

            if path.exists():
                self.leaderboard = pd.read_csv(path)
            else:
                self.warning(
                    f"leaderboard_path is set but file does not exist: "
                    f"{self.state.leaderboard_path}"
                )

        if self.state.metrics_path:

            path = Path(self.state.metrics_path)

            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            else:
                self.warning(
                    f"metrics_path is set but file does not exist: "
                    f"{self.state.metrics_path}"
                )

    # ==========================================================
    # Markdown Report
    # ==========================================================

    def _format_leaderboard(self) -> str:
        """
        BUG 10a FIX — to_markdown() requires the optional `tabulate`
        library. If it isn't installed, fall back to to_string() which
        only needs pandas. Always return a non-empty string.
        """

        if self.leaderboard is None:
            return "No leaderboard available."

        try:
            return self.leaderboard.to_markdown(index=False)

        except ImportError:
            # tabulate not installed — use plain text table
            self.warning(
                "tabulate not installed; leaderboard formatted as plain text. "
                "Install with: pip install tabulate"
            )
            return self.leaderboard.to_string(index=False)

        except Exception as e:
            self.warning(f"Unable to format leaderboard: {e}")
            return self.leaderboard.to_string(index=False)

    def build_markdown(self) -> str:

        leaderboard = self._format_leaderboard()

        critic = (
            "\n".join(self.state.critic_feedback)
            if self.state.critic_feedback
            else "None"
        )

        warnings_text = (
            "\n".join(self.state.warnings)
            if self.state.warnings
            else "None"
        )

        errors_text = (
            "\n".join(error.message for error in self.state.errors)
            or "None"
        )

        execution_history = "\n".join(
            f"- {log.agent}: {log.action}"
            for log in self.state.execution_history
        )

        # BUG 10d FIX — best_model and best_score may be None if the
        # modeling stage failed or was skipped. Render a clear fallback
        # rather than the raw Python string "None".
        best_model_display = self.state.best_model or "Not available"
        best_score_display = (
            f"{self.state.best_score:.6f}"
            if self.state.best_score is not None
            else "Not available"
        )

        report = f"""# SynapseAI AutoML Report

Generated
---------
{datetime.now().isoformat()}

---

# Dataset

Dataset Path

{self.state.dataset_path}

Target Column

{self.state.target_column}

Problem Type

{self.state.problem_type or "Not determined"}

---

# Exploratory Data Analysis

{self.state.eda_report or "Not Available"}

---

# Data Preparation

{self.state.preprocessing_report or "Not Available"}

---

# Feature Engineering

{self.state.feature_report or "Not Available"}

---

# Model Training

{self.state.model_report or "Not Available"}

---

# Best Model

Model

{best_model_display}

Score

{best_score_display}

---

# Leaderboard

{leaderboard}

---

# Critic Feedback

{critic}

---

# Warnings

{warnings_text}

---

# Errors

{errors_text}

---

# Execution History

{execution_history}

"""

        return report

    # ==========================================================
    # JSON Summary
    # ==========================================================

    def build_summary(self) -> dict:

        return {
            "dataset_path": self.state.dataset_path,
            "target_column": self.state.target_column,
            "problem_type": self.state.problem_type,
            "best_model": self.state.best_model,
            "best_score": self.state.best_score,
            "leaderboard_path": self.state.leaderboard_path,
            "metrics_path": self.state.metrics_path,
            "report_path": str(self.report_path),
            "critic_feedback": self.state.critic_feedback,
            "warnings": self.state.warnings,
            "errors": [error.model_dump() for error in self.state.errors],
            "execution_steps": len(self.state.execution_history),
            "completed": True,
        }

    # ==========================================================
    # Save Functions
    # ==========================================================

    def save_report(self, extra_section: str = "") -> None:
        """
        Build and write the markdown report to disk.

        Parameters
        ----------
        extra_section
            Optional additional markdown content appended after the
            main report (e.g. an LLM executive summary).
        """

        markdown = self.build_markdown()

        if extra_section:
            markdown += extra_section

        self.report_path.write_text(markdown, encoding="utf-8")

        self.log(f"Markdown report saved to {self.report_path}")

    def save_summary(self) -> None:
        """Build and write the JSON summary to disk."""

        summary = self.build_summary()

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        self.log(f"Summary saved to {self.summary_path}")

    # ==========================================================
    # Optional LLM Executive Summary
    # ==========================================================

    def generate_llm_summary(self) -> str | None:

        critic_text = (
            "\n".join(self.state.critic_feedback)
            if self.state.critic_feedback
            else "No critic feedback available."
        )

        prompt = f"""
You are SynapseAI.

Write a concise executive summary of this AutoML run.

Problem Type:
{self.state.problem_type or "Unknown"}

Best Model:
{self.state.best_model or "Unknown"}

Best Score:
{self.state.best_score if self.state.best_score is not None else "Unknown"}

Critic Feedback:
{critic_text}

Return a professional summary in less than 250 words.
"""

        try:

            summary = self.ask_llm(
                prompt=prompt,
                system_prompt="You are an expert ML consultant.",
            )

            return summary

        except Exception as e:

            self.warning(f"Unable to generate LLM summary: {e}")

            return None

    # ==========================================================
    # Main Execution
    # ==========================================================

    def run(self):

        self.log("Generating final AutoML report.")

        # ------------------------------------------------------
        # Load generated artifacts
        # ------------------------------------------------------

        self.load_artifacts()

        # ------------------------------------------------------
        # Generate optional LLM executive summary
        # ------------------------------------------------------

        llm_summary = self.generate_llm_summary()

        executive_section = ""

        if llm_summary:
            executive_section = f"""

---

# Executive Summary

{llm_summary}

"""

        # ------------------------------------------------------
        # BUG 10b FIX — save_report() and save_summary() were defined
        # but never called. run() duplicated their logic inline, creating
        # two code paths that could diverge silently.
        # Now delegates to helpers: single source of truth.
        # ------------------------------------------------------

        self.save_report(extra_section=executive_section)
        self.save_summary()

        # ------------------------------------------------------
        # BUG 10c FIX — removed stale comment saying "Add these fields
        # to AgentState if they don't exist". Both report_path and
        # summary_path already exist in state.py.
        # ------------------------------------------------------

        self.state.report_path = str(self.report_path)
        self.state.summary_path = str(self.summary_path)

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------

        self.state.log_execution(
            agent=self.name,
            action="Generated final AutoML report",
            status="SUCCESS",
            details=(
                f"Report: {self.report_path}\n"
                f"Summary: {self.summary_path}"
            ),
        )

        self.state.log_decision(
            agent=self.name,
            decision=(
                "Pipeline completed successfully. "
                "Final report generated."
            ),
        )

        self.log("ReportAgent completed successfully.")

        # ------------------------------------------------------
        # Finish Workflow
        # ------------------------------------------------------

        self.set_phase("FINISHED")
        self.complete()
        self.reset_retry()
        self.state.mark_completed()

        return self.state