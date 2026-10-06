"""
Tracking MPC: CSTR.
Run from adalib_project/ root:
    python examples/mpc/cstr_tracking_mpc.py [--T_ref 136.0]
"""
import subprocess, sys, os

legacy_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'legacy', 'operator_mpc_original', 'cstr_mpc_op')
env = os.environ.copy()
env['PROBLEM'] = 'cstr_mpc'
env['BASIS']   = 'lpa'

subprocess.run(
    [sys.executable, 'main_mpc_cstr.py'] + sys.argv[1:],
    cwd=legacy_dir, env=env, check=True
)
