"""
agents/prep_agent.py

===============================================================================
                                SynapseAI
===============================================================================

Data Preparation Agent

Responsibilities
----------------
✓ Analyze EDA findings
✓ Generate preprocessing code using the LLM
✓ Execute generated code inside the Sandbox
✓ Save cleaned dataset
✓ Update AgentState
✓ Log execution

This agent NEVER performs preprocessing itself.
Instead, it asks the LLM to generate executable Python code and safely
executes that code inside the sandbox.

===============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

from agents.base_agent import BaseAgent


class PrepAgent(BaseAgent):
    """
    Autonomous Data Preparation Agent.

    Workflow
    --------
    1. Read dataset path and EDA report.
    2. Ask the LLM to generate preprocessing code.
    3. Execute the generated code.
    4. Save cleaned dataset.
    5. Update AgentState.
    """

    OUTPUT_DIR = "workspace/data/processed"
    OUTPUT_FILE = "cleaned_dataset.csv"

    def run(self):

        self.log("Starting data preparation.")

        dataset_path = self.state.dataset_path
        target_column = self.state.target_column
        eda_report = self.state.eda_report or "No EDA report available."

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        cleaned_dataset_path = str(
            Path(self.OUTPUT_DIR) / self.OUTPUT_FILE
        )

        # =====================================================
        # Prompt
        # =====================================================

        prompt = f"""
You are SynapseAI's autonomous Data Preparation Agent.

Your task is to intelligently preprocess a dataset based on the dataset characteristics and the EDA findings.

Dataset
-------
Path:
{dataset_path}

Target Column:
{target_column}

EDA Report
----------
{eda_report}

Write ONLY executable Python code.

Your objective is to prepare the dataset for machine learning while preserving as much useful information as possible.

The generated code should:

1. Import all required libraries.

2. Load the dataset using pandas.

3. Analyze the dataset before making preprocessing decisions.

4. Remove duplicate rows if necessary.

5. Handle missing values.

   Decide the best strategy for every column.

   Possible strategies include:

   • Remove rows
   • Remove columns
   • Mean Imputation
   • Median Imputation
   • Mode Imputation
   • Forward Fill
   • Backward Fill

   Choose the strategy based on:

   • Percentage of missing values
   • Feature type
   • Information contained in the feature

6. Handle categorical features.

   Decide automatically whether to use:

   • Label Encoding
   • One-Hot Encoding
   • Ordinal Encoding

   based on the characteristics of each feature.

7. Handle numerical features.

   Decide whether scaling is necessary.

   If scaling is beneficial, choose between:

   • StandardScaler
   • MinMaxScaler
   • RobustScaler

   Explain your reasoning inside Python comments.

8. Detect outliers.

   Decide whether they should be:

   • kept
   • removed
   • clipped

   Use suitable statistical techniques.

9. Remove useless columns such as IDs only if appropriate.

10. Ensure the target column is never modified incorrectly.

11. Use pathlib.Path to create directories. Do NOT use os.makedirs.

12. Save the cleaned dataset to:

{cleaned_dataset_path}

13. Print a preprocessing summary including:

• Missing value strategy
• Encoding strategy
• Scaling strategy
• Outlier handling
• Final dataset shape

Rules

- Return ONLY executable Python code.
- Do NOT use markdown.
- Do NOT include explanations outside Python.
- Python comments are allowed and encouraged to explain preprocessing decisions.
- The generated code must run without requiring manual editing.
- Use pathlib.Path for all file and directory operations, NOT os.

Before applying any preprocessing technique, briefly explain in Python comments WHY you selected that preprocessing strategy.

Example:

# Median imputation selected because this numerical feature contains
# significant outliers.

# RobustScaler selected because the EDA detected heavy-tailed distributions.

# One-Hot Encoding selected because this categorical feature is nominal.
"""

        # =====================================================
        # Generate preprocessing code
        # =====================================================

        self.log("Generating preprocessing code using LLM.")

        # Use the base class helper — consistent with all other agents
        generated_code = self.generate_code(
            prompt=prompt,
            system_prompt="You are an expert AutoML preprocessing assistant.",
        )

        self.state.generated_code.append(generated_code)

        self.log("Executing preprocessing code inside Sandbox.")

        # FIX: was sandbox.execute(script_content=..., script_name=...)
        # The Sandbox.execute() only accepts a single positional arg: script
        result = self.execute_code(generated_code)

        # =====================================================
        # Check Execution Result
        # =====================================================

        if not result.success:

            self.error(
                f"Data preparation failed.\n\n{result.stderr}"
            )

            self.state.retry_required = True

            self.state.log_decision(
                agent=self.name,
                decision="Preprocessing failed. Retry required.",
            )

            self.complete()  # → Supervisor handles the retry

            return self.state

        # =====================================================
        # Verify Output Dataset
        # =====================================================

        if not os.path.exists(cleaned_dataset_path):

            self.error(
                "Preprocessing finished but no cleaned dataset was produced."
            )

            self.state.retry_required = True

            self.complete()

            return self.state

        # =====================================================
        # Update AgentState
        # =====================================================

        self.state.cleaned_dataset_path = cleaned_dataset_path

        # Store the preprocessing report for downstream agents
        self.state.preprocessing_report = result.stdout

        self.state.log_execution(
            agent=self.name,
            action="Generated preprocessing code",
            status="SUCCESS",
            details=result.stdout,
        )

        self.state.log_decision(
            agent=self.name,
            decision=(
                "Dataset cleaned successfully and "
                "stored for downstream agents."
            ),
        )

        self.log(f"Cleaned dataset saved to {cleaned_dataset_path}")

        self.log("Data preparation completed successfully.")

        # =====================================================
        # Advance Workflow
        # =====================================================

        self.set_phase("FEATURE_ENGINEERING")
        self.complete()    # → Supervisor routes to FeatureAgent
        self.reset_retry()

        return self.state