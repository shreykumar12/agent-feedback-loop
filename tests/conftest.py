import pytest

from agent_eval import config


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point storage at a throwaway SQLite file so tests never touch real history."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", path)
    from agent_eval import storage

    storage.init_db()
    return path
