let selectedFile = null;

/* ============================== */
/* DOM ELEMENTS */
/* ============================== */
const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const selectedFileBox = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const detectBtn = document.getElementById("detectBtn");
const loading = document.getElementById("loading");

const resultsSection = document.getElementById("resultsSection");
const resultsContainer = document.getElementById("resultsContainer");
const totalPlates = document.getElementById("totalPlates");

/* ============================== */
/* FILE INPUT LISTENERS */
/* ============================== */
fileInput.addEventListener("change", function () {
    if (this.files.length === 0) return;
    selectFile(this.files[0]);
});

function selectFile(file) {
    const isImage = file.type.startsWith("image/");
    const isVideo = file.type.startsWith("video/");

    if (!isImage && !isVideo) {
        alert("Please select a valid image or video file.");
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    selectedFileBox.classList.remove("hidden");
    detectBtn.disabled = false;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

function removeFile() {
    selectedFile = null;
    fileInput.value = "";
    selectedFileBox.classList.add("hidden");
    detectBtn.disabled = true;
}

/* ============================== */
/* DRAG AND DROP HANDLERS */
/* ============================== */
dropZone.addEventListener("dragover", function (event) {
    event.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", function () {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", function (event) {
    event.preventDefault();
    dropZone.classList.remove("dragover");
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        selectFile(files[0]);
    }
});

/* ============================== */
/* DETECT FILE API CALL */
/* ============================== */
async function detectFile() {
    if (!selectedFile) {
        alert("Please select an image or video.");
        return;
    }

    detectBtn.disabled = true;
    loading.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    resultsContainer.innerHTML = "";

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const endpoint = selectedFile.type.startsWith("image/") ? "/detect/image" : "/detect/video";
        console.log("Sending request to:", endpoint);

        const response = await fetch(endpoint, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Detection process failed.");
        }

        displayResults(data);

    } catch (error) {
        console.error(error);
        alert("Detection failed: " + error.message);
    } finally {
        loading.classList.add("hidden");
        detectBtn.disabled = false;
    }
}

/* ============================== */
/* DISPLAY RESULTS */
/* ============================== */
function displayResults(data) {
    resultsContainer.innerHTML = "";
    let plates = [];

    if (Array.isArray(data.detections)) {
        plates = data.detections;
    } else if (Array.isArray(data.plates)) {
        plates = data.plates;
    }

    // ✅ FIX: Filter out low-confidence / garbage detections
    const MIN_CONFIDENCE = 0.6;
    plates = plates.filter(p => {
        const conf = Number(p.confidence ?? p.detection_confidence ?? 0);
        return conf >= MIN_CONFIDENCE;
    });

    totalPlates.textContent = `${plates.length} Plate${plates.length === 1 ? "" : "s"}`;

    if (plates.length === 0) {
        resultsContainer.innerHTML = `
            <div class="history-empty">
                <div class="empty-icon-3d">🔍</div>
                <h3>No License Plates Detected</h3>
                <p>The neural network couldn't locate a clear number plate in this file.</p>
            </div>
        `;
        resultsSection.classList.remove("hidden");
        return;
    }

    plates.forEach(createResultCard);
    resultsSection.classList.remove("hidden");
}

/* ============================== */
/* CREATE RESULT CARD */
/* ============================== */
function createResultCard(plate) {
    const card = document.createElement("div");
    card.className = "result-card";

    const confidence = Number(plate.confidence ?? plate.detection_confidence ?? 0);
    const confidencePercent = (confidence * 100).toFixed(1);
    const plateText = plate.plate_text ?? plate.text ?? "Unknown";
    const trackId = plate.track_id ?? "N/A";
    const source = plate.source ?? "Uploaded File";

    // ✅ FIX: Comprehensive key mapping + path normalization
    let imageUrl = plate.image_url || plate.plate_image || plate.crop_url || plate.plate_crop ||
                   plate.image_path || plate.crop_path || plate.crop_image ||
                   plate.cropped_image || plate.image || plate.path || plate.url || null;

    if (imageUrl && typeof imageUrl === "string" && imageUrl.trim() !== "") {
        imageUrl = imageUrl.replace(/\\/g, "/");           // Windows \ -> /
        const filename = imageUrl.split("/").pop();         // sirf filename
        imageUrl = "/crops/" + filename;                    // /crops/xyz.jpg
    } else {
        imageUrl = null;
    }

    // ✅ FIX: Better fallback — no "Image Load Failed" text
    let imageHTML = `
        <div class="plate-image-container" style="color:#64748b; font-size:11px; display:flex; align-items:center; justify-content:center; background:#1e293b; height:90px; border-radius:6px;">
            No Preview
        </div>
    `;

    if (imageUrl) {
        imageHTML = `
            <div class="plate-image-container" style="overflow:hidden; border-radius:6px; background:#0f172a;">
                <img class="plate-image"
                     src="${escapeHTML(imageUrl)}"
                     alt="Detected Plate"
                     style="width:100%; height:90px; object-fit:cover; display:block;"
                     onerror="this.onerror=null; this.src='data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%2290%22><rect width=%22200%22 height=%2290%22 fill=%22%230f172a%22/><text x=%2250%25%22 y=%2250%25%22 fill=%22%2364748b%22 font-size=%2212%22 text-anchor=%22middle%22 dy=%22.3em%22>No Preview</text></svg>';">
            </div>
        `;
    }

    card.innerHTML = `
        ${imageHTML}
        <div class="result-info" style="margin-top:10px;">
            <div class="result-label">RECOGNIZED PLATE TEXT</div>
            <div class="plate-number-3d">${escapeHTML(plateText)}</div>
            <div class="result-meta">
                <div class="meta-item">Confidence: <span class="confidence">${confidencePercent}%</span></div>
                <div class="meta-item">Track ID: ${escapeHTML(String(trackId))}</div>
                <div class="meta-item">Source: ${escapeHTML(String(source))}</div>
            </div>
        </div>
    `;

    resultsContainer.appendChild(card);
}

/* ============================== */
/* NAVIGATION & HISTORY */
/* ============================== */
function showSection(sectionId) {
    const uploadSection = document.getElementById("uploadSection");
    const historySection = document.getElementById("historySection");
    const detectionNav = document.getElementById("detectionNav");
    const historyNav = document.getElementById("historyNav");

    uploadSection.classList.add("hidden");
    historySection.classList.add("hidden");
    detectionNav.classList.remove("active");
    historyNav.classList.remove("active");

    if (sectionId === "uploadSection") {
        uploadSection.classList.remove("hidden");
        detectionNav.classList.add("active");
    }

    if (sectionId === "historySection") {
        historySection.classList.remove("hidden");
        historyNav.classList.add("active");
        loadHistory();
    }
}

async function loadHistory() {
    const historyLoading = document.getElementById("historyLoading");
    const historyEmpty = document.getElementById("historyEmpty");
    const historyTableWrapper = document.getElementById("historyTableWrapper");
    const historyTableBody = document.getElementById("historyTableBody");

    historyLoading.classList.remove("hidden");
    historyEmpty.classList.add("hidden");
    historyTableWrapper.classList.add("hidden");

    try {
        const response = await fetch("/detections");
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Failed to load detection logs.");
        }

        let detections = Array.isArray(data) ? data : (Array.isArray(data.detections) ? data.detections : []);

        historyLoading.classList.add("hidden");

        if (detections.length === 0) {
            historyEmpty.classList.remove("hidden");
            return;
        }

        historyTableBody.innerHTML = "";

        detections.forEach(item => {
            const row = document.createElement("tr");
            const confidence = Number(item.confidence || 0);

            row.innerHTML = `
                <td>#${escapeHTML(String(item.id ?? "N/A"))}</td>
                <td class="plate-cell">${escapeHTML(String(item.plate_text ?? "N/A"))}</td>
                <td>${escapeHTML(String(item.track_id ?? "N/A"))}</td>
                <td class="confidence">${(confidence * 100).toFixed(1)}%</td>
                <td>${formatTimestamp(item.timestamp)}</td>
                <td>${escapeHTML(String(item.source ?? "N/A"))}</td>
            `;

            historyTableBody.appendChild(row);
        });

        historyTableWrapper.classList.remove("hidden");

    } catch (error) {
        historyLoading.classList.add("hidden");
        historyEmpty.classList.remove("hidden");
        historyEmpty.innerHTML = `
            <div class="empty-icon-3d">⚠️</div>
            <h3>Could Not Fetch Records</h3>
            <p>${escapeHTML(error.message)}</p>
        `;
    }
}

function formatTimestamp(timestamp) {
    if (!timestamp) return "N/A";
    const date = new Date(timestamp);
    return Number.isNaN(date.getTime()) ? timestamp : date.toLocaleString();
}

function escapeHTML(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}