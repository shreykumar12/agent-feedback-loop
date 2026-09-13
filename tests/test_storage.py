# WHY THIS FILE EXISTS:
#   Regression detection and the dashboard read from SQLite, so writes must
#   round-trip correctly. Uses a temp DB so tests never touch real run history.
#
# WHAT IT NEEDS:
#   - A pytest fixture that points config.DB_PATH at tmp_path.
#   - init_db is idempotent (safe to call twice).
#   - create_run / save_attempt / save_result then read back identical data.
#   - JSON test_results survive the round trip.

import pytest


@pytest.mark.skip(reason="TODO: implement once storage exists")
def test_init_db_idempotent():
    pass


@pytest.mark.skip(reason="TODO")
def test_result_round_trip():
    pass
