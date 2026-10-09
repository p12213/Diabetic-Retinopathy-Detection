/**
 * Diabetic Retinopathy AI Web Application Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const sourceCanvas = document.getElementById('sourceCanvas');
  const heatmapCanvas = document.getElementById('heatmapCanvas');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  
  const presetBtns = document.querySelectorAll('.preset-btn');
  const filterChips = document.querySelectorAll('.filter-chip');
  const chkLesions = document.getElementById('chkLesions');
  
  const btnDualView = document.getElementById('btnDualView');
  const btnArchitectureView = document.getElementById('btnArchitectureView');
  const visualizerViewport = document.getElementById('visualizerViewport');
  const architectureViewport = document.getElementById('architectureViewport');
  
  const gradCamSlider = document.getElementById('gradCamSlider');
  const gradCamVal = document.getElementById('gradCamVal');
  const leftTag = document.getElementById('leftTag');
  
  const severityBanner = document.getElementById('severityBanner');
  const severityTitle = document.getElementById('severityTitle');
  const riskCategory = document.getElementById('riskCategory');
  const confidenceScore = document.getElementById('confidenceScore');
  const probList = document.getElementById('probList');
  const findingsBox = document.getElementById('findingsBox');
  const recommendationBox = document.getElementById('recommendationBox');
  const btnPrintReport = document.getElementById('btnPrintReport');
  const scanTime = document.getElementById('scanTime');

  // Application State
  const state = {
    currentGrade: 0,
    currentFilter: 'original',
    showLesions: true,
    gradCamOpacity: 0.6,
    customImage: null
  };

  // Class Names for ICDR Grades
  const gradeLabels = [
    "0: No DR",
    "1: Mild NPDR",
    "2: Moderate NPDR",
    "3: Severe NPDR",
    "4: Proliferative DR"
  ];

  /**
   * Main Render Pipeline Trigger
   */
  function render() {
    // 1. Draw Base Fundus Image (Custom Uploaded or Preset Renders)
    if (state.customImage) {
      const ctx = sourceCanvas.getContext('2d');
      ctx.drawImage(state.customImage, 0, 0, sourceCanvas.width, sourceCanvas.height);
    } else {
      CVEngine.generateSampleFundus(sourceCanvas, state.currentGrade);
    }

    // 2. Apply Computer Vision Filters
    if (state.currentFilter === 'green') {
      CVEngine.applyGreenChannel(sourceCanvas, sourceCanvas);
      leftTag.textContent = 'CV: Green Channel (High Vessel Contrast)';
    } else if (state.currentFilter === 'clahe') {
      CVEngine.applyCLAHE(sourceCanvas, sourceCanvas);
      leftTag.textContent = 'CV: CLAHE (Contrast Limited Adaptive Hist)';
    } else if (state.currentFilter === 'vessels') {
      CVEngine.applyVesselMask(sourceCanvas, sourceCanvas);
      leftTag.textContent = 'CV: Blood Vessel Edge Network';
    } else {
      leftTag.textContent = 'Original Retinal Fundus Photo';
    }

    // 3. Draw Lesion Markers if toggled
    if (state.showLesions && !state.customImage) {
      CVEngine.drawLesionBoundingBoxes(sourceCanvas, state.currentGrade);
    }

    // 4. Generate Grad-CAM Heatmap
    AIInferenceEngine.generateGradCAM(
      sourceCanvas,
      heatmapCanvas,
      state.currentGrade,
      state.gradCamOpacity
    );

    // 5. Update Diagnostic Panel
    updateDiagnosticPanel();
  }

  /**
   * Updates Diagnostic UI Results & Probabilities
   */
  function updateDiagnosticPanel() {
    const result = AIInferenceEngine.predict(state.currentGrade);
    const meta = result.meta;

    // Update Severity Card
    severityTitle.textContent = meta.title;
    riskCategory.textContent = meta.risk;
    confidenceScore.textContent = `${result.confidence.toFixed(1)}%`;
    severityBanner.style.setProperty('--banner-accent', meta.color);

    // Render Multi-Class Probability Bars
    probList.innerHTML = '';
    result.probabilities.forEach((prob, idx) => {
      const pct = (prob * 100).toFixed(1);
      const isPredicted = idx === state.currentGrade;
      const barColor = isPredicted ? meta.color : 'var(--primary-cyan)';

      const itemHtml = `
        <div class="prob-item">
          <div class="prob-meta">
            <span style="${isPredicted ? 'font-weight:700; color:var(--text-main);' : ''}">${gradeLabels[idx]}</span>
            <span>${pct}%</span>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width: ${pct}%; background: ${barColor};"></div>
          </div>
        </div>
      `;
      probList.insertAdjacentHTML('beforeend', itemHtml);
    });

    // Update Findings & Recommendations
    findingsBox.innerHTML = `<ul>${meta.findings.map(f => `<li>${f}</li>`).join('')}</ul>`;
    recommendationBox.textContent = meta.recommendations;
    scanTime.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // --- Event Handlers ---

  // Preset Grade Case Buttons
  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      presetBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      state.currentGrade = parseInt(btn.dataset.grade, 10);
      state.customImage = null; // Reset custom image when preset clicked
      render();
    });
  });

  // CV Filter Chips
  filterChips.forEach(chip => {
    chip.addEventListener('click', () => {
      filterChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');

      state.currentFilter = chip.dataset.filter;
      render();
    });
  });

  // Lesions Checkbox
  chkLesions.addEventListener('change', (e) => {
    state.showLesions = e.target.checked;
    render();
  });

  // Grad-CAM Opacity Slider
  gradCamSlider.addEventListener('input', (e) => {
    state.gradCamOpacity = parseFloat(e.target.value);
    gradCamVal.textContent = `${Math.round(state.gradCamOpacity * 100)}%`;
    render();
  });

  // View Switcher Tabs
  btnDualView.addEventListener('click', () => {
    btnDualView.classList.add('active');
    btnArchitectureView.classList.remove('active');
    visualizerViewport.style.display = 'grid';
    architectureViewport.classList.remove('active');
  });

  btnArchitectureView.addEventListener('click', () => {
    btnArchitectureView.classList.add('active');
    btnDualView.classList.remove('active');
    visualizerViewport.style.display = 'none';
    architectureViewport.classList.add('active');
  });

  // File Upload Handlers
  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleImageUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleImageUpload(e.target.files[0]);
    }
  });

  function handleImageUpload(file) {
    const reader = new FileReader();
    reader.onload = (evt) => {
      const img = new Image();
      img.onload = () => {
        state.customImage = img;
        // Estimate random grade for uploaded image demonstration
        state.currentGrade = Math.floor(Math.random() * 4) + 1;
        presetBtns.forEach(b => b.classList.remove('active'));
        render();
      };
      img.src = evt.target.result;
    };
    reader.readAsDataURL(file);
  }

  // PDF Report Export Trigger
  btnPrintReport.addEventListener('click', () => {
    window.print();
  });

  // Initial Run on Load
  render();
});
