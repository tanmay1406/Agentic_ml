"""
Prompt templates for SynapseAI agents.
"""

EDA_PROMPT = """
You are an expert data scientist.

Dataset:
{dataset}

Target Column:
{target}

Write ONLY executable Python code.

Tasks:
1. Load dataset using pandas.
2. Print dataset shape.
3. Print column names.
4. Detect missing values.
5. Detect duplicates.
6. Print descriptive statistics.
7. Detect whether the task is classification or regression.
8. Print a concise EDA summary.

Return only Python code.
"""


PREPROCESS_PROMPT = """
TODO
"""


FEATURE_PROMPT = """
TODO
"""


MODEL_PROMPT = """
TODO
"""


CRITIC_PROMPT = """
TODO
"""


REPORT_PROMPT = """
TODO
"""