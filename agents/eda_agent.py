"""
agents/eda_agent.py

===============================================================================
                                SynapseAI
===============================================================================

Exploratory Data Analysis Agent

Responsibilities
----------------
✓ Inspect dataset
✓ Detect missing values
✓ Detect duplicates
✓ Determine problem type
✓ Summarize dataset
✓ Store findings in AgentState

===============================================================================
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class EDAAgent(BaseAgent):
    """
    Performs exploratory data analysis using LLM-generated Python code.
    """

    def run(self):

        self.log("Starting Exploratory Data Analysis")

        prompt = f"""
You are an expert data scientist.

Dataset
-------
Path:
{self.state.dataset_path}

Target column:
{self.state.target_column}

Write ONLY executable Python code.

The code must:

1. Load the dataset using pandas.

2. Print:
   - Dataset shape
   - Column names
   - Data types

3. Detect:
   - Missing values
   - Duplicate rows

4. Print:
   - Statistical summary

5. Detect whether the task is:
   - Classification
   - Regression

6. Print class distribution if classification.

7. Print feature types:
   Numerical
   Categorical
   Boolean
   Datetime

8. Finish with a concise EDA report.

Do not include explanations.

Do not wrap the code in markdown.
"""

        code = self.generate_code(prompt)

        self.state.generated_code.append(code)

        result = self.execute_code(code)

        # ----------------------------------------------------------
        # Failure path — hand retry control back to Supervisor
        # ----------------------------------------------------------

        if not result.success:

            self.error(result.stderr)

            self.state.retry_required = True

            self.complete()  # → Supervisor will handle the retry

            return self.state

        # ----------------------------------------------------------
        # Success path
        # ----------------------------------------------------------

        self.state.eda_report = result.stdout

        self.log("EDA completed successfully")

        self.state.log_decision(
            agent=self.name,
            decision="EDA report successfully generated.",
        )

        # Advance the workflow so Supervisor routes to PrepAgent next
        self.set_phase("PREPROCESSING")
        self.complete()       # sets current_agent → "Supervisor"
        self.reset_retry()    # clear any previous retry count

        return self.state