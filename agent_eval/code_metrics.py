"""Deterministic static metrics for generated code (no LLM needed).

Cheap, reproducible complements to the optional LLM judge: size, cyclomatic
complexity (McCabe: 1 + decision points), and max nesting depth.
"""

from __future__ import annotations

import ast
import difflib

_DECISIONS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.ExceptHandler,
              ast.With, ast.AsyncWith, ast.Assert, ast.comprehension, ast.match_case)
_NESTING = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try,
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Match)


def _depth(node: ast.AST, level: int = 0) -> int:
    best = level
    for child in ast.iter_child_nodes(node):
        best = max(best, _depth(child, level + 1 if isinstance(child, _NESTING) else level))
    return best


def analyze(code: str) -> dict | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, _DECISIONS):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    sloc = sum(1 for line in code.splitlines() if line.strip() and not line.strip().startswith("#"))
    return {
        "sloc": sloc,
        "cyclomatic_complexity": complexity,
        "max_nesting": _depth(tree),
        "num_functions": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)),
    }


def edit_ratio(before: str, after: str) -> float:
    """0.0 = identical, 1.0 = completely rewritten (line-level diff)."""
    return round(1.0 - difflib.SequenceMatcher(None, before.splitlines(), after.splitlines()).ratio(), 4)
