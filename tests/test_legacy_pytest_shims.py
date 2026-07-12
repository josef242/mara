"""Collect the legacy tests that were ALREADY pytest-style but lived outside
any suite. Star-import makes pytest collect their test_ functions here, with
tests/conftest.py providing absolute sys.path (their own `sys.path.insert(0,
".")` hacks are cwd-dependent and harmless once ours is in place).

(test_row_center.py was shimmed here until 2026-07-13; it now lives in
tests/ and is collected natively.)
"""
from test_body_lr_controller_auto import *  # noqa: F401,F403  (common_fsdp2, 12 tests)
