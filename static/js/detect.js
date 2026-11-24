(function () {
  const script = document.currentScript;
  const MAX_BYTES = Number(script.dataset.maxMb) * 1024 * 1024;
  const enabled = script.dataset.enabled === "true";
  const ALLOWED = ["image/jpeg", "image/png", "image/webp"];
  const $ = (id) => document.getElementById(id);
  const dz = $("dropzone"), fileInput = $("file"), camInput = $("file-camera"), preview = $("preview");
  const errBox = $("error"), btnDetect = $("btn-detect"), meta = $("file-meta");
  let selected = null, stream = null;

  const showError = (m) => { errBox.textContent = m; errBox.classList.remove("d-none"); };
  const clearError = () => errBox.classList.add("d-none");

  function setFile(file) {
    clearError();
    if (!file) return;
    if (!ALLOWED.includes(file.type)) return showError("Unsupported file type. Please choose a JPG, PNG or WebP image.");
    if (file.size === 0) return showError("That file is empty.");
    if (file.size > MAX_BYTES) return showError("That file is too large. The limit is " + (MAX_BYTES / 1048576) + " MB.");
    selected = file;
    const url = URL.createObjectURL(file);
    preview.onerror = () => { reset(); showError("This image could not be displayed. Try another photo."); };
    preview.onload = () => { meta.textContent = file.name + " (" + (file.size / 1024).toFixed(0) + " KB, " + preview.naturalWidth + "×" + preview.naturalHeight + ")"; };
    preview.src = url;
    preview.classList.remove("d-none"); $("dz-empty").classList.add("d-none"); dz.classList.add("has-image");
    $("btn-clear").classList.remove("d-none");
    btnDetect.disabled = !enabled;
  }
  function reset() {
    selected = null; fileInput.value = ""; camInput.value = ""; preview.removeAttribute("src");
    preview.classList.add("d-none"); $("dz-empty").classList.remove("d-none"); dz.classList.remove("has-image");
    $("btn-clear").classList.add("d-none"); btnDetect.disabled = true; meta.textContent = "No image selected yet.";
  }

  $("btn-browse").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  dz.addEventListener("click", () => { if (!selected) fileInput.click(); });
  dz.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && !selected) { e.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  camInput.addEventListener("change", () => setFile(camInput.files[0]));
  $("btn-clear").addEventListener("click", reset);
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  // Camera: live capture where getUserMedia exists (needs https or localhost), else the device camera picker.
  const modalEl = $("camModal"), modal = new bootstrap.Modal(modalEl);
  $("btn-camera").addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return camInput.click();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      $("cam").srcObject = stream; modal.show();
    } catch (err) { camInput.click(); }
  });
  modalEl.addEventListener("hidden.bs.modal", () => { if (stream) stream.getTracks().forEach((t) => t.stop()); stream = null; });
  $("btn-snap").addEventListener("click", () => {
    const v = $("cam"), c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight; c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob((blob) => { modal.hide(); if (blob) setFile(new File([blob], "camera-capture.jpg", { type: "image/jpeg" })); }, "image/jpeg", 0.92);
  });

  btnDetect.addEventListener("click", async () => {
    if (!selected) return;
    clearError();
    btnDetect.disabled = true; $("spinner").classList.remove("d-none"); $("detect-label").textContent = "Analysing…";
    const fd = new FormData(); fd.append("image", selected);
    try {
      const res = await fetch("/api/predict", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "The server could not process this image.");
      window.location.href = data.redirect;
    } catch (err) {
      showError(err.message && err.message !== "Failed to fetch" ? err.message : "Could not reach the CropVision server. Is it still running?");
      btnDetect.disabled = false; $("spinner").classList.add("d-none"); $("detect-label").textContent = "Detect Disease";
    }
  });
})();
