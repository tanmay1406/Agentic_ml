"""
agents/model_agent.py

===============================================================================
                                SynapseAI
===============================================================================

Model Training Agent

Responsibilities
----------------
✓ Analyze engineered dataset
✓ Determine the machine learning task
✓ Select appropriate candidate models
✓ Generate model training code using the LLM
✓ Execute generated code inside the Sandbox
✓ Save trained models and evaluation metrics
✓ Update AgentState

Unlike previous agents, this agent is responsible for producing the
highest-performing machine learning model while avoiding overfitting.

The LLM should reason about the dataset and choose suitable algorithms,
evaluation metrics, and validation strategies.

===============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

from agents.base_agent import BaseAgent


class ModelAgent(BaseAgent):
    """
    Autonomous Model Training Agent.

    Workflow
    --------
    1. Read engineered dataset.
    2. Analyze the ML problem.
    3. Generate training code using the LLM.
    4. Execute generated code in the Sandbox.
    5. Save trained models.
    6. Update AgentState.
    """

    MODEL_DIR = "workspace/models"

    def run(self):

        self.log("Starting model training.")

        dataset_path = self.state.feature_dataset_path
        target_column = self.state.target_column

        eda_report = (
            self.state.eda_report
            or "No EDA report available."
        )

        feature_report = (
            self.state.feature_report
            or "No feature engineering report available."
        )

        os.makedirs(
            self.MODEL_DIR,
            exist_ok=True,
        )

        best_model_path = str(
            Path(self.MODEL_DIR) / "best_model.pkl"
        )

        leaderboard_path = str(
            Path(self.MODEL_DIR) / "leaderboard.csv"
        )

        metrics_path = str(
            Path(self.MODEL_DIR) / "metrics.json"
        )

        # ==========================================================
        # Prompt
        # ==========================================================

        prompt = f"""
You are SynapseAI's autonomous Model Training Agent.

Your objective is to produce the highest-performing machine learning
model while avoiding overfitting and data leakage.

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

Feature Engineering Report
--------------------------
{feature_report}

Write ONLY executable Python code.

Before generating code, internally determine:

• Whether the problem is classification or regression.
• Which candidate models are appropriate.
• Which validation strategy should be used.
• Which evaluation metric should be optimized.

Do NOT output your reasoning.

Generate Python code that performs the following tasks.

1. Import all required libraries.

2. Load the engineered dataset.

3. Validate the dataset.

Raise descriptive exceptions if:

• Dataset is empty.
• Target column does not exist.
• Target column contains invalid values.

4. Automatically determine whether the task is:

• Binary Classification
• Multi-class Classification
• Regression

5. Split the dataset into training and testing sets.

Use random_state=42.

6. Select multiple suitable candidate models.

Examples include:

Classification

• Logistic Regression
• Random Forest
• Gradient Boosting
• XGBoost
• LightGBM
• CatBoost
• Support Vector Machine
• KNN
• MLP

Regression

• Linear Regression
• ElasticNet
• Random Forest
• Gradient Boosting
• XGBoost
• LightGBM
• CatBoost
• Support Vector Regression
• MLP Regressor

Only choose models appropriate for the detected problem.

7. Store the candidate models inside a dictionary.

8. Train every model using a loop.

Do NOT duplicate training code.

9. Use cross-validation whenever appropriate.

10. Automatically choose evaluation metrics.

Examples:

Classification

• Accuracy
• Precision
• Recall
• F1
• ROC AUC

Regression

• RMSE
• MAE
• R²

11. Record the score of every model.

12. Save every trained model individually using joblib.

13. Create a leaderboard dataframe containing:

• Model Name
• Evaluation Metric
• Score
• Training Time

14. Sort the leaderboard by best score.

15. Save the leaderboard to:

{leaderboard_path}

16. Save the best-performing model to:

{best_model_path}

17. Save a structured metrics file to:

{metrics_path}

IMPORTANT — Do NOT use bare open() calls for file writing.
Use pathlib.Path instead:

    import json
    from pathlib import Path
    Path('{metrics_path}').write_text(
        json.dumps(metrics_dict, indent=4),
        encoding='utf-8',
    )

The JSON should contain:

• problem_type
• models_trained
• best_model
• best_score
• evaluation_metric

18. Print the following report.

====================================

MODEL TRAINING REPORT

====================================

Problem Type :

Models Trained :

Evaluation Metric :

Best Model :

Best Score :

Leaderboard Saved :

Best Model Saved :

Metrics Saved :

====================================

Rules

- Return ONLY executable Python code.

- Do NOT use markdown.

- Do NOT use shell commands.

- Do NOT use subprocess.

- Do NOT access the internet.

- Do NOT use bare open() calls. Use pathlib.Path for file writing.

- Avoid data leakage.

- Never modify the original dataset.

- Preserve reproducibility using random_state=42.

- Use Python comments to explain important modelling decisions.
"""

        # ==========================================================
        # Generate Model Training Code
        # ==========================================================

        self.log(
            "Generating model training code using LLM."
        )

        try:

            generated_code = self.llm.generate_code(
                prompt=prompt,
                system_prompt=(
                    "You are an expert machine learning engineer "
                    "specializing in automated model selection."
                ),
            )

        except Exception as e:

            self.error(
                f"LLM generation failed: {e}"
            )

            self.state.retry_required = True

            # BUG FIX: was missing self.complete() — without it the main loop
            # re-runs ModelAgent directly, bypassing the Supervisor entirely.
            self.complete()

            return self.state

        self.state.generated_code.append(
            generated_code
        )

        self.log(
            "Executing model training code inside Sandbox."
        )

        result = self.sandbox.execute(
            script=generated_code,
        )

        # ==========================================================
        # Check Execution Result
        # ==========================================================

        if not result.success:

            self.error(
                f"Model training failed.\n\n{result.stderr}"
            )

            self.state.retry_required = True

            self.state.log_decision(
                agent=self.name,
                decision="Model training failed. Retry required.",
            )

            # BUG FIX: was missing self.complete() — same Supervisor bypass issue
            self.complete()

            return self.state

        # ==========================================================
        # Verify Generated Artifacts
        # ==========================================================

        missing_files = []

        if not os.path.exists(best_model_path):
            missing_files.append(best_model_path)

        if not os.path.exists(leaderboard_path):
            missing_files.append(leaderboard_path)

        if not os.path.exists(metrics_path):
            missing_files.append(metrics_path)

        if missing_files:

            self.error(
                "Model training completed but the following "
                f"artifacts were not generated:\n{missing_files}"
            )

            self.state.retry_required = True

            self.state.log_decision(
                agent=self.name,
                decision="Model artifacts missing after execution.",
            )

            # BUG FIX: was missing self.complete() — same Supervisor bypass issue
            self.complete()

            return self.state

        # ==========================================================
        # Update AgentState
        # ==========================================================

        self.state.best_model_path = best_model_path

        self.state.leaderboard_path = leaderboard_path

        self.state.metrics_path = metrics_path

        self.state.model_report = result.stdout

        # ----------------------------------------------------------
        # Read metrics.json (if available)
        # ----------------------------------------------------------

        try:

            import json

            with open(metrics_path, "r") as f:

                metrics = json.load(f)

            self.state.problem_type = metrics.get(
                "problem_type",
                self.state.problem_type,
            )

            self.state.best_model = metrics.get(
                "best_model",
                self.state.best_model,
            )

            self.state.best_score = metrics.get(
                "best_score",
                self.state.best_score,
            )

        except Exception as e:

            self.log(
                f"Unable to parse metrics.json: {e}"
            )

        # ==========================================================
        # Logging
        # ==========================================================

        self.state.log_execution(
            agent=self.name,
            action="Generated model training code",
            status="SUCCESS",
            details=result.stdout,
        )

        self.state.log_decision(
            agent=self.name,
            decision=(
                "Model training completed successfully. "
                "Best model selected and saved."
            ),
        )

        self.log(
            f"Best model saved to {best_model_path}"
        )

        self.log(
            "Model training completed successfully."
        )

        # ==========================================================
        # Workflow Progression
        # ==========================================================

        self.set_phase("CRITIC")
        self.complete()          # current_agent = "Supervisor"
        self.reset_retry()

        return self.state