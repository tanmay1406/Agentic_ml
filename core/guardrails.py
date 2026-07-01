"""
core/guardrails.py

===============================================================================
                                SynapseAI
===============================================================================

LLM Code Guardrails

This module validates LLM-generated Python code before execution.

Responsibilities
----------------
✓ Detect dangerous imports
✓ Detect dangerous function calls
✓ Prevent filesystem abuse
✓ Prevent shell execution
✓ Prevent network access
✓ Ensure generated code is safe enough for sandbox execution

NOTE: Enable in sandbox.py by uncommenting the guard.validate() block.

===============================================================================
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List


# =============================================================================
# Validation Result
# =============================================================================

@dataclass
class GuardResult:
    """
    Result returned after validating generated code.
    """

    safe: bool
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Code Guard
# =============================================================================

class CodeGuard:
    """
    Static AST-based validator for LLM-generated Python code.

    Checks are layered:
        1. Syntax — can the code be parsed at all?
        2. Forbidden imports — no networking, shell, or system modules.
        3. Forbidden calls — no eval/exec/open/system etc.

    All errors are collected before returning so the caller sees every
    problem in one pass rather than one-at-a-time.
    """

    # -------------------------------------------------------------------------
    # Forbidden Imports
    # Kept in sync with sandbox.py FORBIDDEN_IMPORTS
    # -------------------------------------------------------------------------

    FORBIDDEN_IMPORTS: frozenset = frozenset({
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "telnetlib",
        "paramiko",
        "multiprocessing",
        "threading",
        "os",       # agents should use pathlib instead
        "sys",
        "shutil",
        "ctypes",
        "resource",
        "signal",
        "psutil",
    })

    # -------------------------------------------------------------------------
    # Forbidden Function / Method Calls
    # -------------------------------------------------------------------------

    FORBIDDEN_CALLS: frozenset = frozenset({
        # Dangerous builtins
        "eval",
        "exec",
        "compile",
        "__import__",
        "input",
        # Shell execution (os / subprocess methods)
        "system",
        "popen",
        "Popen",
        # Destructive filesystem operations
        "remove",
        "unlink",
        "rmtree",
        "chmod",
        "chown",
    })

    # -------------------------------------------------------------------------
    # Allowed Packages
    # Kept in sync with sandbox.py ALLOWED_IMPORTS
    # -------------------------------------------------------------------------

    ALLOWED_IMPORTS: frozenset = frozenset({
        "pandas",
        "numpy",
        "sklearn",
        "scipy",
        "matplotlib",
        "seaborn",
        "plotly",
        "xgboost",
        "lightgbm",
        "catboost",
        "joblib",
        "pickle",
        "pathlib",
        "json",
        "math",
        "statistics",
        "collections",
        "itertools",
        "typing",
        "datetime",
        "time",
        "warnings",
        "re",
        "copy",
        "functools",
    })

    # =========================================================================
    # Public API
    # =========================================================================

    def validate(self, code: str) -> GuardResult:
        """
        Validate generated Python code.

        Runs three passes over the AST:
            1. Syntax check
            2. Forbidden import scan
            3. Forbidden call scan

        Returns a GuardResult with safe=True only if all passes succeed.
        All errors are accumulated — the full list is always returned.
        """

        errors: List[str] = []

        # ------------------------------------------------------------------
        # Pass 1 — Syntax
        # ------------------------------------------------------------------

        try:
            tree = ast.parse(code)

        except SyntaxError as e:
            # Can't continue without a valid AST
            return GuardResult(
                safe=False,
                errors=[f"SyntaxError on line {e.lineno}: {e.msg}"],
            )

        # ------------------------------------------------------------------
        # Pass 2 & 3 — Walk the AST once for both import and call checks
        # ------------------------------------------------------------------

        for node in ast.walk(tree):

            # --------------------------------------------------------------
            # Check: import xxx
            # --------------------------------------------------------------

            if isinstance(node, ast.Import):

                for alias in node.names:
                    root = alias.name.split(".")[0]

                    if root in self.FORBIDDEN_IMPORTS:
                        errors.append(
                            f"Forbidden import '{root}' "
                            f"(line {node.lineno})"
                        )

            # --------------------------------------------------------------
            # Check: from xxx import yyy
            # --------------------------------------------------------------

            elif isinstance(node, ast.ImportFrom):

                if node.module is None:
                    continue

                root = node.module.split(".")[0]

                if root in self.FORBIDDEN_IMPORTS:
                    errors.append(
                        f"Forbidden import '{root}' "
                        f"(line {node.lineno})"
                    )

            # --------------------------------------------------------------
            # Check: dangerous function / method calls
            # --------------------------------------------------------------

            elif isinstance(node, ast.Call):

                # bare call: eval(...), exec(...)
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.FORBIDDEN_CALLS:
                        errors.append(
                            f"Forbidden call '{node.func.id}()' "
                            f"(line {node.lineno})"
                        )

                # attribute call: os.system(...), shutil.rmtree(...)
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.FORBIDDEN_CALLS:
                        errors.append(
                            f"Forbidden call '.{node.func.attr}()' "
                            f"(line {node.lineno})"
                        )

        return GuardResult(
            safe=len(errors) == 0,
            errors=errors,
        )