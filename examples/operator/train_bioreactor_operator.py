"""
Operator training: Fed-batch bioreactor.
This example delegates to the legacy main_train.py with the correct env vars.
Run from adalib_project/ root:
    python examples/operator/train_bioreactor_operator.py
"""
import subprocess, sys, os

legacy_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'legacy', 'operator_mpc_original', 'cstr_mpc_op')
env = os.environ.copy()
env['PROBLEM'] = 'bioreactor'
env['BASIS']   = 'lpa'

subprocess.run(
    [sys.executable, 'main_train.py'],
    cwd=legacy_dir, env=env, check=True
)
