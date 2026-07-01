"""
agents/model_agent.py

===============================================================================
                                SynapseAI
===============================================================================

Model Training Agent — ALL BUGS FIXED IN THIS VERSION

Bugs fixed
----------
Bug 3  Missing self.complete() in three error paths. Without it,
       current_agent never reverted to "Supervisor", causing the main loop
       to re-run ModelAgent directly on every retry, bypassing retry-limit
       enforcement and all Supervisor routing logic.

Bug 11 Bypassed BaseAgent helpers: used self.llm.generate_code() and
       self.sandbox.execute() directly. Correct pattern is
       self.generate_code() and self.execute_code() for consistent logging.

Bug 12 State persistence gap: after training, the in-memory
       state.leaderboard (List[CandidateModel]) and state.candidate_models
       were never populated even though the leaderboard CSV existed on disk.
       ModelAgent now reads the leaderboard CSV after execution and syncs
       it into the in-memory state so downstream agents can access models
       without touching the filesystem directly.

===============================================================================
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from agents.base_agent import BaseAgent
from core.state import CandidateModel


class ModelAgent(BaseAgent):
    """
    Autonomous Model Training Agent.

    Workflow
    --------
    1. Read engineered dataset.
    2. Generate training code using the LLM.
    3. Execute generated code in the Sandbox.
    4. Verify artifacts exist on disk.
    5. Sync leaderboard into in-memory AgentState.
    6. Route back to Supervisor.
    """

    MODEL_DIR = "workspace/models"

    def run(self):

        self.log("Starting model training.")

        # ==========================================================
        # Guard: feature_dataset_path must not be None
        #
        # If FeatureAgent failed or was skipped, feature_dataset_path
        # stays None. Without this check, None gets interpolated into
        # the LLM prompt as the string "None", causing the generated
        # code to try to open a file literally called "None".
        # ==========================================================

        if not self.state.feature_dataset_path:

            self.error(
                "feature_dataset_path is None. "
                "FeatureAgent may not have run or may have failed. "
                "Cannot proceed with model training."
            )

            self.state.retry_required = True
            self.complete()
            return self.state

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

        os.makedirs(self.MODEL_DIR, exist_ok=True)

        best_model_path = str(Path(self.MODEL_DIR).resolve() / "best_model.pkl")
        leaderboard_path = str(Path(self.MODEL_DIR).resolve() / "leaderboard.csv")
        metrics_path = str(Path(self.MODEL_DIR).resolve() / "metrics.json")

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

14. Sort the leaderboard by best score (descending).

15. Save the leaderboard to:

{leaderboard_path}

16. Save the best-performing model to:

{best_model_path}

17. Save a structured metrics file to:

{metrics_path}

IMPORTANT: Do NOT use bare open() calls for file writing.
Use pathlib.Path instead:

    import json
    from pathlib import Path
    Path('{metrics_path}').write_text(
        json.dumps(metrics_dict, indent=4),
        encoding='utf-8',
    )

The JSON must contain these exact keys:

• problem_type
• models_trained
• best_model
• best_score
• evaluation_metric
• train_score      (training set score of the best model — for overfitting detection)

18. Print the following report.

====================================

MODEL TRAINING REPORT

====================================

Problem Type :

Models Trained :

Evaluation Metric :

Best Model :

Best Score :

Train Score :

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
        #
        # BUG 11 FIX — was self.llm.generate_code() (direct call).
        # Use self.generate_code() which adds standardised logging
        # and makes future cross-cutting concerns (tracing, retry
        # policy, cost tracking) easy to add in one place.
        # ==========================================================

        try:

            generated_code = self.generate_code(
                prompt,
                "You are an expert machine learning engineer "
                "specialising in automated model selection.",
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
        # BUG 11 FIX — was self.sandbox.execute(script=generated_code).
        # Use self.execute_code() for consistent logging.
        # ==========================================================

        result = self.execute_code(generated_code)

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

            self.complete()   # BUG 3 FIX
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

            self.complete()   # BUG 3 FIX
            return self.state

        # ==========================================================
        # Update AgentState — paths and report
        # ==========================================================

        self.state.best_model_path = best_model_path
        self.state.leaderboard_path = leaderboard_path
        self.state.metrics_path = metrics_path
        self.state.model_report = result.stdout

        # ==========================================================
        # Read metrics.json — populate scalar fields in state
        # ==========================================================

        try:

            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            self.state.problem_type = metrics.get(
                "problem_type", self.state.problem_type
            )
            self.state.best_model = metrics.get(
                "best_model", self.state.best_model
            )
            self.state.best_score = metrics.get(
                "best_score", self.state.best_score
            )

        except Exception as e:
            self.log(f"Unable to parse metrics.json: {e}")

        # ==========================================================
        # BUG 12 FIX — Sync leaderboard into in-memory AgentState
        #
        # Before this fix, state.leaderboard was always an empty list
        # even though leaderboard.csv existed on disk.
        # Downstream code that accessed state.leaderboard directly
        # (e.g. future agents, dashboards, monitoring hooks) would see
        # no models at all.
        #
        # This reads the leaderboard CSV and populates:
        #   state.candidate_models   — all trained models
        #   state.leaderboard        — sorted by score (via update_leaderboard)
        #   state.best_model         — automatically updated
        #   state.best_score         — automatically updated
        # ==========================================================

        try:

            lb_df = pd.read_csv(leaderboard_path)

            for _, row in lb_df.iterrows():

                model = CandidateModel(
                    name=str(row.get("Model Name", "unknown")),
                    algorithm=str(row.get("Model Name", "unknown")),
                    score=float(row.get("Score", 0.0)),
                    training_time=float(row.get("Training Time", 0.0)),
                )

                self.state.candidate_models.append(model)
                self.state.update_leaderboard(model)

            self.log(
                f"In-memory leaderboard populated with "
                f"{len(lb_df)} models."
            )

        except Exception as e:

            self.log(
                f"Unable to sync leaderboard into state: {e}. "
                "Disk-based access via leaderboard_path will still work."
            )

        # ==========================================================
        # Logging
        # ==========================================================

        self.state.log_execution(
            agent=self.name,
            action="Model training code generated and executed",
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

        self.log(f"Best model saved to {best_model_path}")
        self.log("Model training completed successfully.")

        # ==========================================================
        # Workflow Progression
        # ==========================================================

        self.set_phase("CRITIC")
        self.complete()          # current_agent = "Supervisor"
        self.reset_retry()

        return self.state