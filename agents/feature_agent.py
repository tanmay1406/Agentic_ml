"""
agents/feature_agent.py

===============================================================================
                                SynapseAI
===============================================================================

Feature Engineering Agent

Responsibilities
----------------
✓ Analyze cleaned dataset
✓ Generate feature engineering code using the LLM
✓ Execute generated code inside the Sandbox
✓ Save engineered dataset
✓ Update AgentState
✓ Log execution

Unlike the PrepAgent, this agent is responsible for improving the
predictive power of the dataset through intelligent feature engineering.

The LLM should reason about the dataset and choose appropriate feature
engineering techniques instead of blindly applying transformations.

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
    1. Read cleaned dataset.
    2. Analyze feature characteristics.
    3. Ask the LLM to generate feature engineering code.
    4. Execute generated code inside the Sandbox.
    5. Save engineered dataset.
    6. Update AgentState.
    """

    OUTPUT_DIR = "workspace/data/processed"
    OUTPUT_FILE = "feature_dataset.csv"

    def run(self):

        self.log("Starting feature engineering.")

        dataset_path = self.state.cleaned_dataset_path
        target_column = self.state.target_column

        eda_report = (
            self.state.eda_report
            or "No EDA report available."
        )

        preprocessing_report = (
            getattr(self.state, "preprocessing_report", None)
            or "No preprocessing report available."
        )

        os.makedirs(
            self.OUTPUT_DIR,
            exist_ok=True,
        )

        feature_dataset_path = str(
            Path(self.OUTPUT_DIR)
            / self.OUTPUT_FILE
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

- Do NOT use bare open() calls. Use pathlib.Path for all file writing:
    from pathlib import Path
    Path(output_path).write_text(content, encoding='utf-8')

- Never modify the original dataset.

- Always preserve the target column.

- Use only pandas, numpy and scikit-learn.

- Explain important feature engineering decisions using Python comments.
"""

        # ==========================================================
        # Generate Feature Engineering Code
        # ==========================================================

        self.log(
            "Generating feature engineering code using LLM."
        )

        try:

            generated_code = self.llm.generate_code(
                prompt=prompt,
                system_prompt=(
                    "You are an expert Feature Engineering "
                    "specialist for machine learning."
                ),
            )

        except Exception as e:

            self.error(
                f"LLM generation failed: {e}"
            )

            self.state.retry_required = True

            # BUG FIX: was missing self.complete() — without it the main loop
            # re-runs FeatureAgent directly, bypassing the Supervisor's retry
            # limit and routing logic entirely.
            self.complete()

            return self.state

        self.state.generated_code.append(
            generated_code
        )

        self.log(
            "Executing feature engineering code "
            "inside Sandbox."
        )

        result = self.sandbox.execute(
            script=generated_code,
        )

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

            # BUG FIX: was missing self.complete() — same Supervisor bypass issue
            self.complete()

            return self.state

        # ==========================================================
        # Verify Output Dataset
        # ==========================================================

        if not os.path.exists(feature_dataset_path):

            self.error(
                "Feature engineering completed but no output dataset was generated."
            )

            self.state.retry_required = True

            self.state.log_decision(
                agent=self.name,
                decision="Feature dataset missing after execution.",
            )

            # BUG FIX: was missing self.complete() — same Supervisor bypass issue
            self.complete()

            return self.state

        # ==========================================================
        # Update AgentState
        # ==========================================================

        self.state.feature_dataset_path = feature_dataset_path

        # Save generated report (stdout)
        self.state.feature_report = result.stdout

        # Store generated code
        self.state.log_execution(
            agent=self.name,
            action="Generated feature engineering code",
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
            f"Feature engineered dataset saved to "
            f"{feature_dataset_path}"
        )

        self.log(
            "Feature engineering completed successfully."
        )

        # ==========================================================
        # Workflow Progression
        # ==========================================================

        # BUG FIX: was:
        #   self.state.current_phase = "MODELING"   ← direct field mutation
        #   self.state.next_agent("ModelAgent")     ← bypasses Supervisor entirely
        #
        # Every agent must hand control back to the Supervisor via self.complete(),
        # which sets current_agent = "Supervisor". The Supervisor then reads
        # current_phase from PHASE_ROUTING to decide the next agent.
        # Routing directly to ModelAgent skips retry-limit checks and any
        # future Supervisor logic (e.g. conditional branching, logging).

        self.set_phase("MODELING")
        self.complete()    # sets current_agent = "Supervisor"
        self.reset_retry()

        return self.state