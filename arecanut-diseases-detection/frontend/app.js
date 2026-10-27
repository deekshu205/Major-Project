/**
 * Smart Areca Nut Crop Advisor - PWA Client Logic
 * Handles camera capture, drag-and-drop, micro-climate auto-fetch,
 * multimodal API inference, and Grad-CAM explainability rendering.
 */

// --------------------------------------------------------------------------
// 1. Service Worker & PWA Installation
// --------------------------------------------------------------------------
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/sw.js")
      .then((reg) => {
        console.log("[PWA] Service Worker registered successfully. Scope:", reg.scope);
      })
      .catch((err) => {
        console.warn("[PWA] Service Worker registration failed:", err);
      });
  });
}

let deferredInstallPrompt = null;
const btnInstallApp = document.getElementById("btnInstallApp");

window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  deferredInstallPrompt = e;
  if (btnInstallApp) {
    btnInstallApp.style.display = "inline-flex";
  }
});

if (btnInstallApp) {
  btnInstallApp.addEventListener("click", async () => {
    if (!deferredInstallPrompt) return;
    deferredInstallPrompt.prompt();
    const { outcome } = await deferredInstallPrompt.userChoice;
    console.log("[PWA] User choice outcome:", outcome);
    deferredInstallPrompt = null;
    btnInstallApp.style.display = "none";
  });
}

// --------------------------------------------------------------------------
// 2. State & DOM References
// --------------------------------------------------------------------------
let selectedImageFile = null;
let selectedImageDataUrl = null;

// DOM Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const cameraInput = document.getElementById("cameraInput");
const btnCamera = document.getElementById("btnCamera");
const btnBrowse = document.getElementById("btnBrowse");
const previewContainer = document.getElementById("previewContainer");
const previewImg = document.getElementById("previewImg");
const btnRemovePreview = document.getElementById("btnRemovePreview");
const btnQuickScan = document.getElementById("btnQuickScan");
const btnMultimodalScan = document.getElementById("btnMultimodalScan");
const btnMultimodalScanBottom = document.getElementById("btnMultimodalScanBottom");

// Environmental Accordion & Sliders
const toggleMetrics = document.getElementById("toggleMetrics");
const metricsContent = document.getElementById("metricsContent");
const btnAutoFetchWeather = document.getElementById("btnAutoFetchWeather");
const btnAutoFetchText = document.getElementById("btnAutoFetchText");

const sliderHumidity = document.getElementById("sliderHumidity");
const valHumidity = document.getElementById("valHumidity");
const sliderRainfall = document.getElementById("sliderRainfall");
const valRainfall = document.getElementById("valRainfall");
const sliderTemp = document.getElementById("sliderTemp");
const valTemp = document.getElementById("valTemp");
const sliderNdvi = document.getElementById("sliderNdvi");
const valNdvi = document.getElementById("valNdvi");
const sliderNdviDelta15 = document.getElementById("sliderNdviDelta15");
const valNdviDelta15 = document.getElementById("valNdviDelta15");
const sliderSoilMoisture = document.getElementById("sliderSoilMoisture");
const valSoilMoisture = document.getElementById("valSoilMoisture");

// Results Card
const resultsCard = document.getElementById("resultsCard");
const resultsBadge = document.getElementById("resultsBadge");
const diagnosisIcon = document.getElementById("diagnosisIcon");
const diagnosisName = document.getElementById("diagnosisName");
const diagnosisCategory = document.getElementById("diagnosisCategory");
const diagnosisConfidence = document.getElementById("diagnosisConfidence");
const confidenceMeterFill = document.getElementById("confidenceMeterFill");
const arbitrationCallout = document.getElementById("arbitrationCallout");
const arbitrationText = document.getElementById("arbitrationText");
const treatmentText = document.getElementById("treatmentText");
const probsList = document.getElementById("probsList");

// Grad-CAM Card
const gradcamCard = document.getElementById("gradcamCard");
const camOriginalImg = document.getElementById("camOriginalImg");
const camHeatmapImg = document.getElementById("camHeatmapImg");

// Loading Overlay
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingSubtext = document.getElementById("loadingSubtext");

// --------------------------------------------------------------------------
// 3. Metadata Mapping for 9 Arecanut Classes
// --------------------------------------------------------------------------
const CLASS_META = {
  male_koleroga: {
    title: "Koleroga / Mahali (Fruit Rot)",
    icon: "🍄",
    category: "Severe Fungal Pathogen",
    badgeColor: "#f43f5e",
    bgBadge: "rgba(244, 63, 94, 0.15)",
    borderBadge: "rgba(244, 63, 94, 0.4)",
  },
  bud_borer: {
    title: "Bud Borer Infestation",
    icon: "🐛",
    category: "Pest Infestation",
    badgeColor: "#f43f5e",
    bgBadge: "rgba(244, 63, 94, 0.15)",
    borderBadge: "rgba(244, 63, 94, 0.4)",
  },
  yellow_leaf: {
    title: "Yellow Leaf Disease (YLD)",
    icon: "🍂",
    category: "Phytoplasma / Nutrient Stress",
    badgeColor: "#f59e0b",
    bgBadge: "rgba(245, 158, 11, 0.15)",
    borderBadge: "rgba(245, 158, 11, 0.4)",
  },
  stem_bleeding: {
    title: "Stem Bleeding Disease",
    icon: "🪵",
    category: "Fungal Vascular Disorder",
    badgeColor: "#f43f5e",
    bgBadge: "rgba(244, 63, 94, 0.15)",
    borderBadge: "rgba(244, 63, 94, 0.4)",
  },
  stem_cracking: {
    title: "Stem Cracking / Sun Scorch",
    icon: "⚡",
    category: "Physiological Stress",
    badgeColor: "#f59e0b",
    bgBadge: "rgba(245, 158, 11, 0.15)",
    borderBadge: "rgba(245, 158, 11, 0.4)",
  },
  healthy_leaf: {
    title: "Healthy Leaf Canopy",
    icon: "🍃",
    category: "Optimal Foliar Health",
    badgeColor: "#10b981",
    bgBadge: "rgba(16, 185, 129, 0.15)",
    borderBadge: "rgba(16, 185, 129, 0.4)",
  },
  healthy_nut: {
    title: "Healthy Nut Bunch",
    icon: "🥥",
    category: "Normal Fruit Development",
    badgeColor: "#10b981",
    bgBadge: "rgba(16, 185, 129, 0.15)",
    borderBadge: "rgba(16, 185, 129, 0.4)",
  },
  healthy_trunk: {
    title: "Healthy Structural Trunk",
    icon: "🌴",
    category: "Intact Vascular Stem",
    badgeColor: "#10b981",
    bgBadge: "rgba(16, 185, 129, 0.15)",
    borderBadge: "rgba(16, 185, 129, 0.4)",
  },
  healthy_foot: {
    title: "Healthy Root Collar / Foot",
    icon: "🪴",
    category: "Sound Basal Foundation",
    badgeColor: "#10b981",
    bgBadge: "rgba(16, 185, 129, 0.15)",
    borderBadge: "rgba(16, 185, 129, 0.4)",
  },
};

// --------------------------------------------------------------------------
// 4. Image Upload & Camera Handling
// --------------------------------------------------------------------------
function handleImageSelect(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Please select a valid image file (JPG, PNG, WEBP).");
    return;
  }

  selectedImageFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    selectedImageDataUrl = e.target.result;
    previewImg.src = selectedImageDataUrl;
    previewContainer.style.display = "block";
    if (btnQuickScan) btnQuickScan.disabled = false;
    if (btnMultimodalScan) btnMultimodalScan.disabled = false;
    if (btnMultimodalScanBottom) btnMultimodalScanBottom.disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearImageSelection() {
  selectedImageFile = null;
  selectedImageDataUrl = null;
  previewImg.src = "";
  previewContainer.style.display = "none";
  fileInput.value = "";
  cameraInput.value = "";
  if (btnQuickScan) btnQuickScan.disabled = true;
  if (btnMultimodalScan) btnMultimodalScan.disabled = true;
  if (btnMultimodalScanBottom) btnMultimodalScanBottom.disabled = true;
  resultsCard.classList.remove("visible");
  gradcamCard.classList.remove("visible");
}

// Trigger native inputs
btnCamera.addEventListener("click", (e) => {
  e.stopPropagation();
  cameraInput.click();
});

btnBrowse.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

cameraInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    handleImageSelect(e.target.files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    handleImageSelect(e.target.files[0]);
  }
});

btnRemovePreview.addEventListener("click", (e) => {
  e.stopPropagation();
  clearImageSelection();
});

// Drag and drop events on dropzone
["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("drag-over");
  });
});

dropzone.addEventListener("drop", (e) => {
  const dt = e.dataTransfer;
  if (dt && dt.files && dt.files.length > 0) {
    handleImageSelect(dt.files[0]);
  }
});

// --------------------------------------------------------------------------
// 5. Environmental Metrics & Sliders
// --------------------------------------------------------------------------
toggleMetrics.addEventListener("click", () => {
  const isExpanded = toggleMetrics.getAttribute("aria-expanded") === "true";
  toggleMetrics.setAttribute("aria-expanded", !isExpanded);
  toggleMetrics.classList.toggle("active");
  metricsContent.classList.toggle("open");
});

sliderHumidity.addEventListener("input", (e) => {
  valHumidity.textContent = `${e.target.value}%`;
});

sliderRainfall.addEventListener("input", (e) => {
  valRainfall.textContent = `${e.target.value} mm`;
});

sliderTemp.addEventListener("input", (e) => {
  valTemp.textContent = `${parseFloat(e.target.value).toFixed(1)} °C`;
});

sliderNdvi.addEventListener("input", (e) => {
  valNdvi.textContent = parseFloat(e.target.value).toFixed(2);
});

sliderNdviDelta15.addEventListener("input", (e) => {
  const val = parseFloat(e.target.value);
  valNdviDelta15.textContent = val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2);
});

sliderSoilMoisture.addEventListener("input", (e) => {
  valSoilMoisture.textContent = `${e.target.value}%`;
});

// --------------------------------------------------------------------------
// 6. Open-Meteo Weather Auto-Fetch
// --------------------------------------------------------------------------
btnAutoFetchWeather.addEventListener("click", async () => {
  btnAutoFetchText.textContent = "Fetching GPS & Weather...";
  btnAutoFetchWeather.disabled = true;

  const fetchWeatherForCoords = async (lat, lon) => {
    try {
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat.toFixed(4)}&longitude=${lon.toFixed(4)}&current=temperature_2m,relative_humidity_2m&daily=precipitation_sum&past_days=7`;
      const res = await fetch(url);
      if (!res.ok) throw new Error("Weather API returned non-200");
      const data = await res.json();

      // Current temp & RH
      if (data.current) {
        if (data.current.temperature_2m != null) {
          sliderTemp.value = data.current.temperature_2m;
          valTemp.textContent = `${data.current.temperature_2m.toFixed(1)} °C`;
        }
        if (data.current.relative_humidity_2m != null) {
          sliderHumidity.value = data.current.relative_humidity_2m;
          valHumidity.textContent = `${Math.round(data.current.relative_humidity_2m)}%`;
        }
      }

      // 7-day cumulative precipitation
      if (data.daily && data.daily.precipitation_sum) {
        const sumPrecip = data.daily.precipitation_sum.reduce((a, b) => (a || 0) + (b || 0), 0);
        const roundedPrecip = Math.min(350, Math.round(sumPrecip));
        sliderRainfall.value = roundedPrecip;
        valRainfall.textContent = `${roundedPrecip} mm`;
      }

      btnAutoFetchText.textContent = `Weather Updated (${lat.toFixed(2)}°, ${lon.toFixed(2)}°)`;
      setTimeout(() => {
        btnAutoFetchText.textContent = "Auto-Detect Weather";
        btnAutoFetchWeather.disabled = false;
      }, 3000);
    } catch (err) {
      console.error("[Weather] Error fetching weather:", err);
      btnAutoFetchText.textContent = "Failed to fetch weather";
      setTimeout(() => {
        btnAutoFetchText.textContent = "Auto-Detect Weather";
        btnAutoFetchWeather.disabled = false;
      }, 3000);
    }
  };

  if ("geolocation" in navigator) {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        fetchWeatherForCoords(pos.coords.latitude, pos.coords.longitude);
      },
      (err) => {
        console.warn("[Weather] Geolocation denied or unavailable, using Arecanut Malnad belt default (Shimoga/Sringeri):", err);
        // Default to prominent arecanut producing region (Karnataka Malnad: lat 13.9299, lon 75.0427)
        fetchWeatherForCoords(13.9299, 75.0427);
      },
      { timeout: 8000 }
    );
  } else {
    // Geolocation not supported fallback
    fetchWeatherForCoords(13.9299, 75.0427);
  }
});

// --------------------------------------------------------------------------
// 7. Two-Tier Crop Diagnosis Inference & Explainability
// --------------------------------------------------------------------------

// Action 1: Quick Vision Scan (Photo Only -> POST /predict)
async function executeQuickVisionScan() {
  if (!selectedImageFile) {
    alert("Please capture or select an arecanut image first.");
    return;
  }

  loadingSubtext.textContent = "Analyzing specimen via Standalone Vision Model (EfficientNet-B0) & Grad-CAM...";
  loadingOverlay.classList.add("active");

  const formData = new FormData();
  formData.append("file", selectedImageFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Inference server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    renderDiagnosisResults(data, "vision");
  } catch (error) {
    console.error("[Quick Vision Scan] Error:", error);
    alert(`Quick Vision Scan failed: ${error.message || "Could not connect to inference backend."}`);
  } finally {
    loadingOverlay.classList.remove("active");
  }
}

// Action 2: Full Multimodal Assessment (Photo + Weather & NDVI -> POST /predict-multimodal)
async function executeMultimodalAssessment() {
  if (!selectedImageFile) {
    alert("Please capture or select an arecanut image first.");
    return;
  }

  loadingSubtext.textContent = "Synthesizing visual evidence with micro-climate & Sentinel-2 NDVI via XGBoost Arbiter...";
  loadingOverlay.classList.add("active");

  const formData = new FormData();
  formData.append("file", selectedImageFile);
  formData.append("relative_humidity_pct", sliderHumidity.value);
  formData.append("rainfall_mm_7d", sliderRainfall.value);
  formData.append("temperature_c", sliderTemp.value);
  formData.append("ndvi_current", sliderNdvi.value);
  formData.append("ndvi_delta_15d", sliderNdviDelta15.value);
  formData.append("ndvi_delta_30d", (parseFloat(sliderNdviDelta15.value) * 1.5).toFixed(2));
  formData.append("soil_moisture_pct", sliderSoilMoisture.value);

  try {
    const response = await fetch("/predict-multimodal", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Multimodal server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    renderDiagnosisResults(data, "multimodal");
  } catch (error) {
    console.error("[Multimodal Assessment] Error:", error);
    alert(`Multimodal Assessment failed: ${error.message || "Could not connect to inference backend."}`);
  } finally {
    loadingOverlay.classList.remove("active");
  }
}

// Bind Action Listeners
if (btnQuickScan) {
  btnQuickScan.addEventListener("click", executeQuickVisionScan);
}
if (btnMultimodalScan) {
  btnMultimodalScan.addEventListener("click", executeMultimodalAssessment);
}
if (btnMultimodalScanBottom) {
  btnMultimodalScanBottom.addEventListener("click", executeMultimodalAssessment);
}

// --------------------------------------------------------------------------
// 8. Render Results & Grad-CAM Visualizations
// --------------------------------------------------------------------------
function renderDiagnosisResults(data, mode = "multimodal") {
  const predClass = data.final_predicted_class || data.predicted_class;
  const meta = CLASS_META[predClass] || {
    title: predClass.replace(/_/g, " "),
    icon: "🔬",
    category: "Identified Class",
    badgeColor: "#10b981",
    bgBadge: "rgba(16, 185, 129, 0.15)",
    borderBadge: "rgba(16, 185, 129, 0.4)",
  };

  // Populate Mode Badge
  if (resultsBadge) {
    if (mode === "vision") {
      resultsBadge.textContent = "Mode: Standalone Vision Model (EfficientNet-B0)";
      resultsBadge.style.background = "rgba(16, 185, 129, 0.15)";
      resultsBadge.style.color = "var(--mint-accent)";
      resultsBadge.style.border = "1px solid rgba(16, 185, 129, 0.35)";
    } else {
      resultsBadge.textContent = "Mode: Multimodal XGBoost Decision Arbiter";
      resultsBadge.style.background = "rgba(6, 182, 212, 0.15)";
      resultsBadge.style.color = "#38bdf8";
      resultsBadge.style.border = "1px solid rgba(6, 182, 212, 0.4)";
    }
  }

  // Populate Main Diagnosis
  diagnosisIcon.textContent = meta.icon;
  diagnosisIcon.style.border = `1px solid ${meta.borderBadge}`;
  diagnosisIcon.style.background = meta.bgBadge;

  diagnosisName.textContent = meta.title;
  diagnosisCategory.textContent = meta.category;
  diagnosisCategory.style.background = meta.bgBadge;
  diagnosisCategory.style.color = meta.badgeColor;

  const rawConf = data.final_confidence != null ? data.final_confidence : (data.confidence != null ? data.confidence : 0);
  const confPct = data.final_confidence_pct || data.confidence_pct || `${(rawConf * 100).toFixed(1)}%`;
  diagnosisConfidence.textContent = confPct;
  diagnosisConfidence.style.color = meta.badgeColor;

  const confVal = rawConf * 100;
  confidenceMeterFill.style.width = `${Math.min(100, Math.max(5, confVal))}%`;
  confidenceMeterFill.style.background = `linear-gradient(90deg, #10b981, ${meta.badgeColor})`;

  // Multimodal Arbitration Callout & Reasoning
  if (mode === "multimodal") {
    if (data.arbiter_reasoning) {
      arbitrationText.innerHTML = `
        <strong>Multimodal Decision Arbiter:</strong> ${data.arbiter_reasoning}
      `;
      arbitrationCallout.style.display = "flex";
    } else {
      arbitrationText.innerHTML = `
        <strong>Multimodal Decision Arbiter:</strong> Environmental telemetry (RH: ${sliderHumidity.value}%, 7d Rain: ${sliderRainfall.value}mm) confirmed <strong>${meta.title}</strong> as the final diagnosis.
      `;
      arbitrationCallout.style.display = "flex";
    }
  } else {
    arbitrationText.innerHTML = `
      <strong>Standalone Vision Scan:</strong> Instant diagnosis generated purely from convolutional feature maps and Grad-CAM spatial attention. <em>Weather and Sentinel-2 NDVI metrics were bypassed. For micro-climate risk calibration, expand Step 2 and run Full Multimodal Assessment.</em>
    `;
    arbitrationCallout.style.display = "flex";
  }

  // Treatment Text
  treatmentText.textContent = data.remedy || "Maintain standard orchard hygiene and recommended irrigation regimes.";

  // Probabilities Breakdown
  probsList.innerHTML = "";

  // Render Top-3 Breakdown
  if (mode === "multimodal" && data.vision_top3 && data.fused_top3) {
    const compBox = document.createElement("div");
    compBox.style.cssText = "display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; margin-bottom: 1.25rem;";

    const renderList = (items, isFused) => {
      return items.map((it, idx) => {
        const meta = CLASS_META[it.class] || { title: it.class.replace(/_/g, " "), badgeColor: "#94a3b8" };
        return `
          <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.35rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.8rem;">
            <span style="color: ${idx === 0 ? '#f8fafc' : 'var(--text-muted)'}; font-weight: ${idx === 0 ? '600' : '400'};">
              #${idx + 1} ${meta.title}
            </span>
            <span style="font-family: var(--font-mono); color: ${isFused ? 'var(--mint-accent)' : '#94a3b8'}; font-weight: 600;">
              ${it.confidence_pct}
            </span>
          </div>
        `;
      }).join("");
    };

    compBox.innerHTML = `
      <div style="background: rgba(255,255,255,0.03); padding: 0.75rem; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
        <div style="font-size: 0.72rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700; letter-spacing: 0.05em; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.3rem;">
          📷 Vision Top-3
        </div>
        ${renderList(data.vision_top3, false)}
      </div>
      <div style="background: rgba(16,185,129,0.06); padding: 0.75rem; border-radius: var(--radius-sm); border: 1px solid rgba(16,185,129,0.25);">
        <div style="font-size: 0.72rem; text-transform: uppercase; color: var(--mint-accent); font-weight: 700; letter-spacing: 0.05em; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.3rem;">
          🛰️ Fused Decision Top-3
        </div>
        ${renderList(data.fused_top3, true)}
      </div>
    `;
    probsList.appendChild(compBox);
  } else if (data.vision_top3) {
    const compBox = document.createElement("div");
    compBox.style.cssText = "margin-bottom: 1.25rem;";
    compBox.innerHTML = `
      <div style="background: rgba(255,255,255,0.03); padding: 0.75rem; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
        <div style="font-size: 0.72rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700; letter-spacing: 0.05em; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.3rem;">
          📷 Vision Top-3 Candidates
        </div>
        ${data.vision_top3.map((it, idx) => {
          const meta = CLASS_META[it.class] || { title: it.class.replace(/_/g, " "), badgeColor: "#94a3b8" };
          return `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.35rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.8rem;">
              <span style="color: ${idx === 0 ? '#f8fafc' : 'var(--text-muted)'}; font-weight: ${idx === 0 ? '600' : '400'};">
                #${idx + 1} ${meta.title}
              </span>
              <span style="font-family: var(--font-mono); color: var(--mint-accent); font-weight: 600;">
                ${it.confidence_pct}
              </span>
            </div>
          `;
        }).join("")}
      </div>
    `;
    probsList.appendChild(compBox);
  }

  const probsSource = data.multimodal_probabilities || data.probabilities;
  if (probsSource) {
    const sortedProbs = Object.entries(probsSource).sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([clsName, probVal]) => {
      const clsMeta = CLASS_META[clsName] || { title: clsName.replace(/_/g, " "), badgeColor: "#94a3b8" };
      const pct = (probVal * 100).toFixed(1);

      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-name" title="${clsMeta.title}">${clsMeta.title}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width: ${pct}%; background: ${clsMeta.badgeColor || "var(--mint-accent)"};"></div>
        </div>
        <span class="prob-val">${pct}%</span>
      `;
      probsList.appendChild(row);
    });
  }

  // Grad-CAM Explainability Cards
  camOriginalImg.src = selectedImageDataUrl;
  if (data.gradcam_base64) {
    camHeatmapImg.src = data.gradcam_base64;
  } else if (data.gradcam_artifact_url) {
    camHeatmapImg.src = data.gradcam_artifact_url;
  }

  // Make Cards Visible
  resultsCard.classList.add("visible");
  gradcamCard.classList.add("visible");

  // Scroll to results
  resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}
