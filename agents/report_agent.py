"""
agents/report_agent.py

===============================================================================
                                SynapseAI
===============================================================================

Report Agent

Responsibilities
----------------
✓ Collect outputs from every agent
✓ Generate final AutoML report
✓ Save markdown report
✓ Save JSON summary
✓ Update AgentState
✓ Mark workflow as completed

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

        super().__init__(
            state,
            llm,
            sandbox,
        )

        self.REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.report_path = (
            self.REPORT_DIR /
            self.REPORT_FILE
        )

        self.summary_path = (
            self.REPORT_DIR /
            self.SUMMARY_FILE
        )

        self.leaderboard = None

        self.metrics = {}

    # ==========================================================
    # Loading
    # ==========================================================

    def load_artifacts(self):

        """
        Load leaderboard and metrics if available.
        """

        if self.state.leaderboard_path:

            path = Path(
                self.state.leaderboard_path
            )

            if path.exists():

                self.leaderboard = pd.read_csv(
                    path
                )

        if self.state.metrics_path:

            path = Path(
                self.state.metrics_path
            )

            if path.exists():

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    self.metrics = json.load(f)

    # ==========================================================
    # Markdown Report
    # ==========================================================

    def build_markdown(self):

        leaderboard = (
            self.leaderboard.to_markdown(index=False)
            if self.leaderboard is not None
            else "No leaderboard available."
        )

        critic = (
            "\n".join(
                self.state.critic_feedback
            )
            if self.state.critic_feedback
            else "None"
        )

        warnings = (
            "\n".join(self.state.warnings)
            if self.state.warnings
            else "None"
        )

        errors = "\n".join(

            error.message

            for error in self.state.errors

        ) or "None"

        execution_history = "\n".join(

            f"- {log.agent}: {log.action}"

            for log in self.state.execution_history

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

{self.state.problem_type}

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

{self.state.best_model}

Score

{self.state.best_score}

---

# Leaderboard

{leaderboard}

---

# Critic Feedback

{critic}

---

# Warnings

{warnings}

---

# Errors

{errors}

---

# Execution History

{execution_history}

"""

        return report
    
    # ==========================================================
    # JSON Summary
    # ==========================================================

    def build_summary(self):

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

            "errors": [

                error.model_dump()

                for error in self.state.errors

            ],

            "execution_steps": len(

                self.state.execution_history

            ),

            "completed": True,

        }

    # ==========================================================
    # Save Functions
    # ==========================================================

    def save_report(self):

        markdown = self.build_markdown()

        self.report_path.write_text(

            markdown,

            encoding="utf-8",

        )

        self.log(

            f"Markdown report saved to {self.report_path}"

        )

    def save_summary(self):

        summary = self.build_summary()

        with open(

            self.summary_path,

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(

                summary,

                f,

                indent=4,

            )

        self.log(

            f"Summary saved to {self.summary_path}"

        )

    # ==========================================================
    # Optional LLM Executive Summary
    # ==========================================================

    def generate_llm_summary(self):

        prompt = f"""
You are SynapseAI.

Write a concise executive summary of this AutoML run.

Problem Type:
{self.state.problem_type}

Best Model:
{self.state.best_model}

Best Score:
{self.state.best_score}

Critic Feedback:
{chr(10).join(self.state.critic_feedback)}

Return a professional summary in less than 250 words.
"""

        try:

            summary = self.ask_llm(

                prompt=prompt,

                system_prompt="You are an expert ML consultant.",

            )

            return summary

        except Exception as e:

            self.warning(

                f"Unable to generate LLM summary: {e}"

            )

            return None
        
        # ==========================================================
    # Main Execution
    # ==========================================================

    def run(self):

        self.log(
            "Generating final AutoML report."
        )

        # ------------------------------------------------------
        # Load generated artifacts
        # ------------------------------------------------------

        self.load_artifacts()

        # ------------------------------------------------------
        # Generate optional executive summary
        # ------------------------------------------------------

        llm_summary = self.generate_llm_summary()

        # ------------------------------------------------------
        # Build markdown report
        # ------------------------------------------------------

        markdown = self.build_markdown()

        if llm_summary:

            markdown += f"""

---

# Executive Summary

{llm_summary}

"""

        self.report_path.write_text(
            markdown,
            encoding="utf-8",
        )

        self.log(
            f"Report written to {self.report_path}"
        )

        # ------------------------------------------------------
        # Save JSON summary
        # ------------------------------------------------------

        summary = self.build_summary()

        with open(
            self.summary_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary,
                f,
                indent=4,
            )

        self.log(
            f"Summary written to {self.summary_path}"
        )

        # ------------------------------------------------------
        # Store report paths in AgentState
        # ------------------------------------------------------

        # Add these fields to AgentState if they don't exist:
        #
        # report_path: Optional[str] = None
        # summary_path: Optional[str] = None

        self.state.report_path = str(
            self.report_path
        )

        self.state.summary_path = str(
            self.summary_path
        )

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------

        self.state.log_execution(
            agent=self.name,
            action="Generated final AutoML report",
            status="SUCCESS",
            details=f"Report: {self.report_path}\nSummary: {self.summary_path}",
        )

        self.state.log_decision(
            agent=self.name,
            decision=(
                "Pipeline completed successfully. "
                "Final report generated."
            ),
        )

        self.log(
            "ReportAgent completed successfully."
        )

        # ------------------------------------------------------
        # Finish Workflow
        # ------------------------------------------------------

        self.set_phase("FINISHED")

        self.complete()

        self.reset_retry()

        self.state.mark_completed()

        return self.state