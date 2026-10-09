/**
 * AI Deep Learning Inference Engine & Grad-CAM Explainable AI (XAI) Heatmap Generator
 */

const AIInferenceEngine = {
  /**
   * Diagnostic Metadata per ICDR Severity Grade
   */
  gradeMetadata: {
    0: {
      title: "Grade 0: No Diabetic Retinopathy",
      risk: "Low Risk",
      color: "#10b981",
      findings: [
        "Normal retinal vascular architecture.",
        "Optic disc boundaries crisp and well-defined.",
        "No microaneurysms, hemorrhages, or exudates observed.",
        "Foveal avascular zone (FAZ) intact."
      ],
      recommendations: "Routine annual diabetic eye screening. Maintain optimal glycemic (HbA1c < 7.0%) and blood pressure control."
    },
    1: {
      title: "Grade 1: Mild NPDR",
      risk: "Mild Risk",
      color: "#eab308",
      findings: [
        "Presence of isolated microaneurysms (< 5 per quadrant).",
        "No hard exudates or macular edema detected.",
        "Retinal blood vessels remain intact with normal caliber.",
        "No venous beading or intraretinal microvascular abnormalities (IRMA)."
      ],
      recommendations: "Follow-up comprehensive eye exam in 6-12 months. Monitor blood glucose, lipid profile, and blood pressure closely."
    },
    2: {
      title: "Grade 2: Moderate NPDR",
      risk: "Moderate Risk",
      color: "#f97316",
      findings: [
        "Multiple microaneurysms and dot-blot intraretinal hemorrhages.",
        "Hard lipid exudates present in temporal macula region.",
        "Early localized vessel tortuosity observed.",
        "No evidence of neovascularization."
      ],
      recommendations: "Referral to Ophthalmologist / Retina Specialist within 3-6 months. Optical Coherence Tomography (OCT) recommended to rule out Diabetic Macular Edema (DME)."
    },
    3: {
      title: "Grade 3: Severe NPDR",
      risk: "High Risk",
      color: "#ef4444",
      findings: [
        ">20 intraretinal hemorrhages in 4 quadrants (4-2-1 rule met).",
        "Definite venous beading present in 2+ quadrants.",
        "Prominent cotton wool spots (soft exudates / nerve fiber layer infarctions).",
        "Prominent Intraretinal Microvascular Abnormalities (IRMA)."
      ],
      recommendations: "Urgent referral to Retina Specialist within 2-4 weeks. Panretinal Photocoagulation (PRP) or Anti-VEGF intravitreal therapy evaluation required."
    },
    4: {
      title: "Grade 4: Proliferative DR (PDR)",
      risk: "Critical Risk (Sight Threatening)",
      color: "#dc2626",
      findings: [
        "Neovascularization of the Optic Disc (NVD) / Elsewhere (NVE).",
        "Vitreous / Preretinal hemorrhage present.",
        "Tractional retinal detachment risk due to fibrovascular proliferation.",
        "High risk of severe irreversible visual loss."
      ],
      recommendations: "IMMEDIATE (within 24-48 hours) Retina Specialist intervention. Panretinal Photocoagulation (PRP) laser therapy and/or Anti-VEGF injection."
    }
  },

  /**
   * Simulates Deep Learning Multi-Class Softmax Probabilities for target grade
   */
  predict(grade = 0) {
    const probs = [0.02, 0.03, 0.03, 0.02, 0.02];
    const confidence = 0.88 + Math.random() * 0.09; // 88% - 97% confidence
    
    probs[grade] = confidence;
    const remaining = (1.0 - confidence) / 4;
    
    for (let i = 0; i < 5; i++) {
      if (i !== grade) probs[i] = remaining + (Math.random() * 0.02 - 0.01);
    }
    
    // Normalize sum to 1.0
    const sum = probs.reduce((a, b) => a + b, 0);
    const normalizedProbs = probs.map(p => Math.max(0.005, p / sum));

    return {
      predictedGrade: grade,
      confidence: confidence * 100,
      probabilities: normalizedProbs,
      meta: this.gradeMetadata[grade]
    };
  },

  /**
   * Generates Grad-CAM Attention Heatmap overlay on target canvas
   */
  generateGradCAM(sourceCanvas, targetCanvas, grade = 0, opacity = 0.6) {
    const sCtx = sourceCanvas.getContext('2d');
    const tCtx = targetCanvas.getContext('2d');
    const w = sourceCanvas.width;
    const h = sourceCanvas.height;

    // Draw base source image
    tCtx.drawImage(sourceCanvas, 0, 0, w, h);

    if (grade === 0) {
      // Normal background low attention map
      return;
    }

    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) * 0.4;

    // Create temporary Heatmap Overlay
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = w;
    tempCanvas.height = h;
    const hCtx = tempCanvas.getContext('2d');

    // Define attention hot spots depending on grade pathology
    const spots = [];
    if (grade >= 1) spots.push({ x: cx - 50, y: cy + 30, r: 40, weight: 0.7 });
    if (grade >= 2) {
      spots.push({ x: cx - 110, y: cy + 40, r: 65, weight: 0.9 });
      spots.push({ x: cx - 70, y: cy + 70, r: 50, weight: 0.85 });
    }
    if (grade >= 3) {
      spots.push({ x: cx + 50, y: cy - 90, r: 70, weight: 0.95 });
      spots.push({ x: cx - 130, y: cy - 40, r: 80, weight: 0.9 });
    }
    if (grade >= 4) {
      spots.push({ x: cx + 110, y: cy - 20, r: 90, weight: 1.0 });
    }

    // Render radial heat spots (Jet color map)
    spots.forEach(spot => {
      const grad = hCtx.createRadialGradient(spot.x, spot.y, 0, spot.x, spot.y, spot.r);
      grad.addColorStop(0, `rgba(255, 0, 0, ${spot.weight})`);       // High attention (Red)
      grad.addColorStop(0.4, `rgba(255, 255, 0, ${spot.weight * 0.7})`); // Mid (Yellow)
      grad.addColorStop(0.7, `rgba(0, 255, 0, ${spot.weight * 0.4})`);   // Low-Mid (Green)
      grad.addColorStop(1, 'transparent');                             // Outer (Transparent)

      hCtx.beginPath();
      hCtx.arc(spot.x, spot.y, spot.r, 0, Math.PI * 2);
      hCtx.fillStyle = grad;
      hCtx.fill();
    });

    // Blend Heatmap over original with user specified opacity
    tCtx.save();
    tCtx.globalAlpha = opacity;
    tCtx.drawImage(tempCanvas, 0, 0);
    tCtx.restore();
  }
};
