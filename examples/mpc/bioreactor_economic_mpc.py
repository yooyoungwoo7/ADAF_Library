"""
Economic MPC: Fed-batch bioreactor.
This example delegates to the legacy main_mpc_bioreactor.py.
Run from adalib_project/ root:
    python examples/mpc/bioreactor_economic_mpc.py
or with custom args:
    python examples/mpc/bioreactor_economic_mpc.py --n_pred 20 --w_ss_track 2.0
"""
import subprocess, sys, os

legacy_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'legacy', 'operator_mpc_original', 'cstr_mpc_op')
env = os.environ.copy()
env['PROBLEM'] = 'bioreactor'
env['BASIS']   = 'lpa'

args = sys.argv[1:]
subprocess.run(
    [sys.executable, 'main_mpc_bioreactor.py'] + args,
    cwd=legacy_dir, env=env, check=True
)
