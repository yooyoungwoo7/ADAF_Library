"""
Tracking MPC: Triple-tank.
Run from adalib_project/ root:
    python examples/mpc/triple_tank_tracking_mpc.py [--h3_target 150.0]
"""
import subprocess, sys, os

legacy_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'legacy', 'operator_mpc_original', 'cstr_mpc_op')
env = os.environ.copy()
env['PROBLEM'] = 'triple_tank_mpc'
env['BASIS']   = 'lpa'

subprocess.run(
    [sys.executable, 'main_mpc_triple_tank.py'] + sys.argv[1:],
    cwd=legacy_dir, env=env, check=True
)
