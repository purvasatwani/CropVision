import pytest


@pytest.mark.parametrize("path", ["/", "/detect", "/history", "/about", "/api/status"])
def test_pages_load(client_no_model, path):
    assert client_no_model.get(path).status_code == 200


def test_unknown_page_is_404(client_no_model):
    assert client_no_model.get("/nope").status_code == 404
    assert client_no_model.get("/results/999").status_code == 404


def test_detect_page_warns_when_model_missing(client_no_model):
    assert b"Model inference is not configured" in client_no_model.get("/detect").data


def test_empty_history_message(client_no_model):
    assert b"No scans yet" in client_no_model.get("/history").data
