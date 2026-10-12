"""
Restricted equation parser.

User equations are parsed with Python's ``ast`` module in ``eval`` mode and
validated against a strict allowlist (numbers, state/parameter names,
+ - * / **, unary +/-, and a fixed set of math functions). Nothing the user
types is ever passed to ``eval``/``exec``. A validated tree is evaluated by
a small interpreter (``_evaluate``) against either a NumPy or a TensorFlow
function table, so one expression yields both the NumPy ``rhs`` and the
TF-native ``rhs_tf`` that ``adalib.CallableODESystem`` needs for L-BFGS.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Sequence

MAX_EXPR_LEN = 2000
MAX_NODES = 500
MAX_POW_LITERAL = 100

# name -> (numpy attr, tensorflow attr, LaTeX command)
FUNCTIONS: Dict[str, tuple] = {
    "sin": ("sin", "math.sin", r"\sin"),
    "cos": ("cos", "math.cos", r"\cos"),
    "tan": ("tan", "math.tan", r"\tan"),
    "asin": ("arcsin", "math.asin", r"\arcsin"),
    "acos": ("arccos", "math.acos", r"\arccos"),
    "atan": ("arctan", "math.atan", r"\arctan"),
    "sinh": ("sinh", "math.sinh", r"\sinh"),
    "cosh": ("cosh", "math.cosh", r"\cosh"),
    "tanh": ("tanh", "math.tanh", r"\tanh"),
    "exp": ("exp", "math.exp", r"\exp"),
    "log": ("log", "math.log", r"\ln"),
    "sqrt": ("sqrt", "math.sqrt", r"\sqrt"),
    "abs": ("abs", "math.abs", None),
}
CONSTANTS: Dict[str, float] = {"pi": math.pi, "e": math.e}
RESERVED = set(FUNCTIONS) | set(CONSTANTS) | {"t"}

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
_UNARYOPS = {ast.UAdd: "+", ast.USub: "-"}


class EquationError(ValueError):
    """Validation error with the offending equation index (or None)."""

    def __init__(self, message: str, index: int | None = None, field_name: str | None = None):
        super().__init__(message)
        self.index = index
        self.field_name = field_name


@dataclass
class ParsedEquation:
    state: str
    source: str
    tree: ast.AST
    symbols: set = field(default_factory=set)
    latex: str = ""


@dataclass
class ParsedSystem:
    state_names: List[str]
    parameter_names: List[str]
    equations: List[ParsedEquation]
    control_names: List[str] = field(default_factory=list)

    def rhs_numpy(self) -> Callable:
        """``rhs(t, x, u, p)``. Works point-wise (``x`` of shape (n_state,)) and
        batched (``x`` of shape (n_state, N), ``u`` of shape (n_control, N)) —
        adalib's generic operator/MPC paths call it batched."""
        import numpy as np

        eqs = self.equations
        states, params, controls = self.state_names, self.parameter_names, self.control_names
        table = _numpy_table()

        def rhs(t, x, u=None, p=None):
            env = _env(states, params, x, p)
            _bind_controls(env, controls, u)
            vals = [np.asarray(_evaluate(eq.tree, env, table), dtype=float) for eq in eqs]
            # constant right-hand sides must still match a batched state's shape
            shape = np.broadcast_shapes(np.shape(x[0]), *[v.shape for v in vals])
            return np.stack([np.broadcast_to(v, shape) for v in vals]).astype(float)

        return rhs

    def rhs_tensorflow(self) -> Callable:
        """``rhs_tf(var_list, i, u, p)`` returning ``dy_i/dt - f_i(y)``."""
        eqs = self.equations
        states, params, controls = self.state_names, self.parameter_names, self.control_names
        table = _tf_table()

        def rhs_tf(var_list, i, u=None, p=None):
            import tensorflow as tf

            dtype = var_list[0][0].dtype
            y = [var_list[k][0] for k in range(len(states))]
            # numbers -> exact constants in the solver dtype (tf.cast of a Python
            # float would go through float32); tensors (trainable) -> tf.cast
            def to_tf(v):
                return tf.constant(float(v), dtype) if _is_scalar_number(v) else tf.cast(v, dtype)

            env = _env(states, params, y, p, cast=to_tf)
            _bind_controls(env, controls, u, cast=to_tf)
            f_i = _evaluate(eqs[i].tree, env, table)
            y_i_t = var_list[i][1]
            # constant RHS (e.g. dx/dt = 2) must still broadcast to the grid
            return y_i_t - (f_i + tf.zeros_like(y_i_t))

        return rhs_tf


# ── public API ─────────────────────────────────────────────────────────

def validate_identifier(name: str, kind: str) -> str:
    name = (name or "").strip()
    if not name.isidentifier() or not name.isascii():
        raise EquationError(f"{kind} name '{name}' is not a valid identifier (letters, digits, _; not starting with a digit).",
                            field_name=kind)
    if name in RESERVED:
        raise EquationError(f"{kind} name '{name}' is reserved (function, constant or time variable).", field_name=kind)
    if name.startswith("_"):
        raise EquationError(f"{kind} name '{name}' must not start with '_'.", field_name=kind)
    return name


def parse_system(state_names: Sequence[str], parameter_names: Sequence[str],
                 equations: Sequence[str], control_names: Sequence[str] = ()) -> ParsedSystem:
    states = [validate_identifier(s, "State variable") for s in state_names]
    params = [validate_identifier(p, "Parameter") for p in parameter_names]
    controls = [validate_identifier(c, "Control input") for c in control_names]
    if not states:
        raise EquationError("Define at least one state variable.")
    dup = _first_duplicate(states + params + controls)
    if dup:
        raise EquationError(f"Name '{dup}' is used more than once among states, parameters and controls.")
    if len(equations) != len(states):
        raise EquationError(f"{len(states)} state variable(s) but {len(equations)} equation(s); "
                            "each state needs exactly one dx/dt equation.")
    known = set(states) | set(params) | set(controls)
    parsed = []
    for i, (state, src) in enumerate(zip(states, equations)):
        tree = parse_expression(src, known, index=i)
        parsed.append(ParsedEquation(state=state, source=src.strip(), tree=tree,
                                     symbols=_names(tree), latex=to_latex(tree)))
    return ParsedSystem(states, params, parsed, controls)


def parse_expression(src: str, known: set, index: int | None = None) -> ast.AST:
    src = (src or "").strip()
    if not src:
        raise EquationError("Equation is empty.", index)
    if len(src) > MAX_EXPR_LEN:
        raise EquationError(f"Equation is longer than {MAX_EXPR_LEN} characters.", index)
    src = src.replace("^", "**")
    try:
        tree = ast.parse(src, mode="eval").body
    except SyntaxError as exc:
        col = f" near column {exc.offset}" if exc.offset else ""
        raise EquationError(f"Invalid syntax{col}: {exc.msg}.", index) from None
    count = sum(1 for _ in ast.walk(tree))
    if count > MAX_NODES:
        raise EquationError(f"Equation is too complex ({count} nodes, limit {MAX_NODES}).", index)
    _check(tree, known, index)
    return tree


# ── validation ─────────────────────────────────────────────────────────

def _check(node: ast.AST, known: set, index):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise EquationError(f"Only numeric literals are allowed (got {node.value!r}).", index)
        return
    if isinstance(node, ast.Name):
        if node.id == "t":
            raise EquationError(
                "Explicit time dependence ('t') is not supported: adalib 0.1.0's TF residual "
                "interface (rhs_tf) does not receive t, so only autonomous ODEs can be solved.", index)
        if node.id in FUNCTIONS:
            raise EquationError(f"'{node.id}' is a function; call it like {node.id}(x).", index)
        if node.id not in known and node.id not in CONSTANTS:
            raise EquationError(f"Undefined symbol '{node.id}'. Declare it as a state variable or parameter.", index)
        return
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise EquationError(f"Operator '{type(node.op).__name__}' is not allowed; use + - * / ^.", index)
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) \
                and isinstance(node.right.value, (int, float)) and abs(node.right.value) > MAX_POW_LITERAL:
            raise EquationError(f"Exponent {node.right.value} is too large (limit {MAX_POW_LITERAL}).", index)
        _check(node.left, known, index)
        _check(node.right, known, index)
        return
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOPS:
            raise EquationError("Only unary + and - are allowed.", index)
        _check(node.operand, known, index)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
            allowed = ", ".join(sorted(FUNCTIONS))
            raise EquationError(f"Function '{name}' is not allowed. Allowed: {allowed}.", index)
        if node.keywords or len(node.args) != 1:
            raise EquationError(f"{node.func.id}() takes exactly one argument.", index)
        _check(node.args[0], known, index)
        return
    raise EquationError(f"Unsupported expression element: {type(node).__name__}.", index)


# ── evaluation ─────────────────────────────────────────────────────────

def _evaluate(node: ast.AST, env: dict, table: dict):
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise EquationError(f"No value supplied for '{node.id}' (control inputs need a value in this module).")
    if isinstance(node, ast.BinOp):
        a = _evaluate(node.left, env, table)
        b = _evaluate(node.right, env, table)
        op = type(node.op)
        if op is ast.Add:
            return a + b
        if op is ast.Sub:
            return a - b
        if op is ast.Mult:
            return a * b
        if op is ast.Div:
            return a / b
        return a ** b
    if isinstance(node, ast.UnaryOp):
        v = _evaluate(node.operand, env, table)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call):
        arg = _evaluate(node.args[0], env, table)
        if isinstance(arg, float):
            # constant sub-expression: keep it a weakly-typed Python float so it
            # adopts the solver dtype (tf.sqrt(2.0) would be a float32 tensor)
            return float(_numpy_table()[node.func.id](arg))
        return table[node.func.id](arg)
    raise EquationError(f"Unsupported expression element: {type(node).__name__}.")


def _env(states, params, x, p, cast=None) -> dict:
    """Bind states and parameters. ``p`` may be a dict or a sequence ordered
    like ``params`` (adalib's reference solver passes ``np.asarray(params)``)."""
    env = {name: x[k] for k, name in enumerate(states)}
    if p is None:
        return env
    items = p.items() if isinstance(p, dict) else zip(params, list(p))
    for name, value in items:
        # TF path: values may be trainable tensors (adalib inverse passes
        # InverseParameter.constrained) -- cast without float() to keep gradients
        env[name] = cast(value) if cast else float(value)
    return env


def _bind_controls(env: dict, controls, u, cast=None) -> None:
    """Bind control inputs from ``u`` (sequence, or array of shape (n_control, N))."""
    if not controls:
        return
    if u is None:
        # bound lazily so an equation that never uses a control still evaluates
        return
    for k, name in enumerate(controls):
        v = u[k]
        env[name] = cast(float(v)) if cast else (float(v) if _is_scalar(v) else v)


def _is_scalar_number(v) -> bool:
    import numbers
    import numpy as np
    return isinstance(v, (numbers.Number, np.number)) and not isinstance(v, bool)


def _is_scalar(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=1)
def _numpy_table() -> dict:
    import numpy as np
    return {k: getattr(np, v[0]) for k, v in FUNCTIONS.items()}


def _tf_table() -> dict:
    import tensorflow as tf
    table = {}
    for k, v in FUNCTIONS.items():
        mod, attr = v[1].split(".")
        table[k] = getattr(getattr(tf, mod), attr)
    return table


# ── helpers ────────────────────────────────────────────────────────────

def _names(tree: ast.AST) -> set:
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _first_duplicate(names):
    seen = set()
    for n in names:
        if n in seen:
            return n
        seen.add(n)
    return None


_PREC = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2, ast.Pow: 4}
_GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
          "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
          "Gamma", "Delta", "Theta", "Lambda", "Xi", "Sigma", "Phi", "Psi", "Omega"}


def latex_symbol(name: str) -> str:
    base, _, sub = name.partition("_")
    head = "\\" + base if base in _GREEK else (base if len(base) == 1 else r"\mathrm{" + base + "}")
    return f"{head}_{{{sub}}}" if sub else head


def to_latex(node: ast.AST, parent_prec: int = 0) -> str:
    if isinstance(node, ast.Constant):
        return f"{node.value:g}" if isinstance(node.value, float) else str(node.value)
    if isinstance(node, ast.Name):
        return r"\pi" if node.id == "pi" else latex_symbol(node.id)
    if isinstance(node, ast.UnaryOp):
        inner = to_latex(node.operand, 3)
        s = ("-" if isinstance(node.op, ast.USub) else "+") + inner
        # parent_prec 2 = left factor of a product: "-\gamma R" needs no parentheses
        return f"\\left({s}\\right)" if parent_prec > 2 else s
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        prec = _PREC[op]
        if op is ast.Div:
            s = r"\frac{" + to_latex(node.left) + "}{" + to_latex(node.right) + "}"
        elif op is ast.Pow:
            s = to_latex(node.left, 5) + "^{" + to_latex(node.right) + "}"
        else:
            right_prec = prec + 1 if op in (ast.Sub, ast.Mult) else prec
            sym = {ast.Add: " + ", ast.Sub: " - ", ast.Mult: r" \, "}[op]
            right = to_latex(node.right, right_prec)
            if isinstance(node.right, ast.UnaryOp) and op in (ast.Add, ast.Sub):
                right = f"\\left({right}\\right)"      # "a - (-b)", not "a - -b"
            s = to_latex(node.left, prec) + sym + right
        return f"\\left({s}\\right)" if prec < parent_prec else s
    if isinstance(node, ast.Call):
        name = node.func.id
        arg = to_latex(node.args[0])
        if name == "sqrt":
            return r"\sqrt{" + arg + "}"
        if name == "abs":
            return r"\left|" + arg + r"\right|"
        return FUNCTIONS[name][2] + r"\left(" + arg + r"\right)"
    return ast.unparse(node)
