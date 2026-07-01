"""
agents/feature_agent.py

===============================================================================
                                SynapseAI
===============================================================================

Feature Engineering Agent — ALL BUGS FIXED IN THIS VERSION

Bugs fixed
----------
Bug 2  Workflow: success path called self.state.next_agent("ModelAgent")
       directly, bypassing the Supervisor entirely. Now routes through
       self.complete() → Supervisor → ModelAgent.

Bug 3  Missing self.complete() in three error paths. Without it,
       current_agent never reverted to "Supervisor", causing the main loop
       to re-run FeatureAgent directly on every retry.

Bug 7  No guard against cleaned_dataset_path being None. If PrepAgent
       failed, None was silently interpolated into the LLM prompt, producing
       hallucinated code that referenced a non-existent file path.

Bug 11 Bypassed BaseAgent helpers: used self.llm.generate_code() and
       self.sandbox.execute() directly. Correct pattern is
       self.generate_code() and self.execute_code(), which add standardised
       logging and allow future cross-cutting concerns (metrics, tracing) to
       be added in one place.

===============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

from agents.base_agent import BaseAgent


class FeatureAgent(BaseAgent):
    """
    Autonomous Feature Engineering Agent.

    Workflow
    --------
    1. Guard: verify cleaned dataset exists.
    2. Ask the LLM to generate feature engineering code.
    3. Execute generated code inside the Sandbox.
    4. Verify engineered dataset was produced.
    5. Update AgentState.
    6. Route back to Supervisor.
    """

    OUTPUT_DIR = "workspace/data/processed"
    OUTPUT_FILE = "feature_dataset.csv"

    def run(self):

        self.log("Starting feature engineering.")

        # ==========================================================
        # BUG 7 FIX — Guard: cleaned_dataset_path must not be None
        #
        # cleaned_dataset_path is set by PrepAgent on success.
        # If PrepAgent failed (or was skipped), it stays None.
        # Without this check, None gets interpolated into the LLM
        # prompt as the string "None", causing the generated code to
        # try to open a file literally called "None".
        # ==========================================================

        if not self.state.cleaned_dataset_path:

            self.error(
                "cleaned_dataset_path is None. "
                "PrepAgent may not have run or may have failed. "
                "Cannot proceed with feature engineering."
            )

            self.state.retry_required = True

            self.complete()   # route back to Supervisor
            return self.state

        dataset_path = self.state.cleaned_dataset_path
        target_column = self.state.target_column

        eda_report = (
            self.state.eda_report
            or "No EDA report available."
        )

        preprocessing_report = (
            self.state.preprocessing_report
            or "No preprocessing report available."
        )

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        feature_dataset_path = str(
            Path(self.OUTPUT_DIR) / self.OUTPUT_FILE
        )

        # ==========================================================
        # Prompt
        # ==========================================================

        prompt = f"""
You are SynapseAI's autonomous Feature Engineering Agent.

Your objective is to improve the predictive performance of the dataset.

Dataset
-------
Path:
{dataset_path}

Target Column
-------------
{target_column}

EDA Report
----------
{eda_report}

Preprocessing Report
--------------------
{preprocessing_report}

Write ONLY executable Python code.

Before writing code, internally reason about the dataset and determine
whether feature engineering is beneficial.

Do NOT output your reasoning.

Generate Python code that performs the following tasks.

1. Import all required libraries.

2. Load the cleaned dataset.

3. Inspect every feature before making decisions.

4. Detect and remove features that are:

   • Constant

   • Near constant

   • Duplicate

   • Highly correlated

   • Identifier columns (if appropriate)

5. Analyze skewed numerical features.

   Decide whether transformations such as:

   • Log

   • Square Root

   • Box-Cox

   are beneficial.

6. Analyze datetime columns.

   If useful, create features such as:

   • Year

   • Month

   • Day

   • Weekday

7. Decide whether interaction features would improve the dataset.

8. Decide whether polynomial features are useful.

   Only generate them when beneficial.

9. Perform feature selection.

   Decide automatically among methods such as:

   • Variance Threshold

   • Mutual Information

   • SelectKBest

   • Recursive Feature Elimination

   • Tree-based Feature Importance

10. Preserve the target column.

11. Preserve meaningful feature names whenever possible.

12. Avoid unnecessary feature engineering.

13. Use random_state=42 whenever randomness is involved.

14. Validate the dataset before applying transformations.

Raise descriptive Python exceptions if:

• dataset is empty

• target column is missing

• duplicate column names exist

15. Save the engineered dataset to:

{feature_dataset_path}

16. Print the following report.

====================================

FEATURE ENGINEERING REPORT

====================================

Original Dataset Shape :

Final Dataset Shape :

Removed Features :

Created Features :

Feature Selection Method :

Feature Transformations :

Output Path :

====================================

Rules

- Return ONLY executable Python code.

- Do NOT use markdown.

- Do NOT access the internet.

- Do NOT execute shell commands.

- Do NOT use subprocess.

- Do NOT use bare open() calls.
  Use pathlib.Path for all file writing:
      from pathlib import Path
      Path(output_path).write_text(content, encoding='utf-8')

- Never modify the original dataset.

- Always preserve the target column.

- Use only pandas, numpy and scikit-learn.

- Explain important feature engineering decisions using Python comments.
"""

        # ==========================================================
        # Generate Feature Engineering Code
        #
        # BUG 11 FIX — use self.generate_code() not self.llm.generate_code()
        # The base-class helper adds a standardised log entry and ensures
        # that any future cross-cutting concern (tracing, metrics, retry
        # policy) only needs to be added in one place.
        # ==========================================================

        try:

            generated_code = self.generate_code(
                prompt,
                "You are an expert Feature Engineering specialist "
                "for machine learning.",
            )

        except Exception as e:

            self.error(f"LLM generation failed: {e}")
            self.state.retry_required = True
            self.complete()   # BUG 3 FIX — route back to Supervisor
            return self.state

        self.state.generated_code.append(generated_code)

        # ==========================================================
        # Execute Generated Code
        #
        # BUG 11 FIX — use self.execute_code() not self.sandbox.execute()
        # Same reasoning: consistent logging through the base-class helper.
        # ==========================================================

        result = self.execute_code(generated_code)

        # ==========================================================
        # Check Execution Result
        # ==========================================================

        if not result.success:

            self.error(
                f"Feature engineering failed.\n\n{result.stderr}"
            )

            self.state.retry_required = True

            self.state.log_decision(
                agent=self.name,
                decision="Feature engineering failed. Retry required.",
            )

            self.complete()   # BUG 3 FIX
            return self.state

        # ==========================================================
        # Verify Output Dataset
        # ==========================================================

        if not os.path.exists(feature_dataset_path):

            self.error(
                "Feature engineering completed but no output "
                "dataset was generated."
            )

            self.state.retry_required = True

            self.state.log_decision(
                agent=self.name,
                decision="Feature dataset missing after execution.",
            )

            self.complete()   # BUG 3 FIX
            return self.state

        # ==========================================================
        # Update AgentState
        # ==========================================================

        self.state.feature_dataset_path = feature_dataset_path
        self.state.feature_report = result.stdout

        self.state.log_execution(
            agent=self.name,
            action="Feature engineering code generated and executed",
            status="SUCCESS",
            details=result.stdout,
        )

        self.state.log_decision(
            agent=self.name,
            decision=(
                "Feature engineering completed successfully "
                "and dataset stored for model training."
            ),
        )

        self.log(
            f"Feature engineered dataset saved to {feature_dataset_path}"
        )

        self.log("Feature engineering completed successfully.")

        # ==========================================================
        # Workflow Progression
        #
        # BUG 2 FIX — was:
        #   self.state.current_phase = "MODELING"   (direct mutation)
        #   self.state.next_agent("ModelAgent")     (bypasses Supervisor)
        #
        # Every agent MUST route through the Supervisor via self.complete().
        # The Supervisor reads current_phase from PHASE_ROUTING and decides
        # the next agent. Bypassing it skips retry-limit enforcement and all
        # future Supervisor logic.
        # ==========================================================

        self.set_phase("MODELING")
        self.complete()    # sets current_agent = "Supervisor"
        self.reset_retry()

        return self.state