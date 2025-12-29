import pathlib
import sys
import types

from fastapi.testclient import TestClient


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_PATH = str(_REPO_ROOT / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)


def _install_fake_rag_module(monkeypatch, *, answer_value="mocked", raise_exc: Exception | None = None):
    rag_pkg = types.ModuleType("rag")
    rag_answer_mod = types.ModuleType("rag.rag_answer")

    def rag_answer(question: str):
        if raise_exc is not None:
            raise raise_exc
        return answer_value

    rag_answer_mod.rag_answer = rag_answer

    monkeypatch.setitem(sys.modules, "rag", rag_pkg)
    monkeypatch.setitem(sys.modules, "rag.rag_answer", rag_answer_mod)


def _import_api_app(monkeypatch, *, answer_value="mocked", raise_exc: Exception | None = None):
    _install_fake_rag_module(monkeypatch, answer_value=answer_value, raise_exc=raise_exc)

    # Ensure we import a fresh module instance after setting sys.modules.
    sys.modules.pop("flask.api", None)

    from flask.api import app  # noqa: E402

    return app


def test_root_ok(monkeypatch):
    app = _import_api_app(monkeypatch)
    client = TestClient(app)

    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"message": "GenAI RAG API is running", "version": "1.0.0"}


def test_health_ok(monkeypatch):
    app = _import_api_app(monkeypatch)
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}


def test_ask_ok(monkeypatch):
    app = _import_api_app(monkeypatch, answer_value="Paris")
    client = TestClient(app)

    res = client.post("/ask", json={"question": "What is the capital of France?"})
    assert res.status_code == 200
    assert res.json() == {"answer": "Paris"}


def test_ask_missing_question_400(monkeypatch):
    app = _import_api_app(monkeypatch)
    client = TestClient(app)

    res = client.post("/ask", json={})
    assert res.status_code == 400
    assert res.json() == {"detail": "No question provided"}


def test_ask_rag_failure_500(monkeypatch):
    app = _import_api_app(monkeypatch, raise_exc=RuntimeError("boom"))
    client = TestClient(app)

    res = client.post("/ask", json={"question": "hi"})
    assert res.status_code == 500
    assert res.json() == {"detail": "boom"}
