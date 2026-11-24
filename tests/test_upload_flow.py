import io

from conftest import png_bytes


def test_predict_refused_when_model_missing(client_no_model):
    r = client_no_model.post("/api/predict", data={"image": (png_bytes(), "leaf.png")})
    assert r.status_code == 503 and r.json["code"] == "model_not_configured"
    assert b"No scans yet" in client_no_model.get("/history").data  # nothing stored


def test_bad_extension(client_stub):
    r = client_stub.post("/api/predict", data={"image": (io.BytesIO(b"x"), "evil.exe")})
    assert r.status_code == 415


def test_corrupt_image_rejected(client_stub):
    r = client_stub.post("/api/predict", data={"image": (io.BytesIO(b"not an image"), "leaf.jpg")})
    assert r.status_code == 422 and r.json["code"] == "invalid_image"


def test_missing_file(client_stub):
    assert client_stub.post("/api/predict").status_code == 400


def test_too_large(tmp_path):
    from conftest import StubPredictor, make_app
    app = make_app(tmp_path, StubPredictor())
    app.config["MAX_CONTENT_LENGTH"] = 1000
    r = app.test_client().post("/api/predict", data={"image": (io.BytesIO(b"0" * 5000), "a.jpg")})
    assert r.status_code == 413


def test_full_flow_history_and_delete(client_stub, tmp_path):
    r = client_stub.post("/api/predict", data={"image": (png_bytes(), "leaf.png")})
    assert r.status_code == 200
    url = r.json["redirect"]
    page = client_stub.get(url)
    assert page.status_code == 200 and b"Late blight" in page.data and b"90.0%" in page.data
    assert b"not a substitute" in page.data
    assert b"Late blight" in client_stub.get("/history").data
    uploads = list((tmp_path / "up").glob("*.jpg"))
    assert len(uploads) == 1 and len(uploads[0].stem) == 32  # random name, re-encoded JPEG
    scan_id = r.json["scan_id"]
    assert client_stub.post(f"/history/{scan_id}/delete").status_code == 302
    assert client_stub.get(url).status_code == 404
    assert not list((tmp_path / "up").glob("*.jpg"))
