/**
 * Diabetic Retinopathy Computer Vision (CV) Engine
 * Pure Client-side Canvas Image Processing Algorithms & Retinal Fundus Generator
 */

const CVEngine = {
  /**
   * Generates a synthetic clinical fundus image on canvas for preset DR grades (0 to 4).
   */
  generateSampleFundus(canvas, grade = 0) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.45;

    // 1. Clear background (Dark border)
    ctx.fillStyle = '#030712';
    ctx.fillRect(0, 0, width, height);

    // 2. Base Retinal Fundus Sphere Gradient
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.clip();

    const bgGrad = ctx.createRadialGradient(cx - 30, cy - 20, 20, cx, cy, radius);
    bgGrad.addColorStop(0, '#e05828');   # Warm orange-red core
    bgGrad.addColorStop(0.5, '#c23e18'); # Mid retinal red
    bgGrad.addColorStop(0.85, '#781c08');# Deep periphery red
    bgGrad.addColorStop(1.0, '#380902'); # Dark border edge
    ctx.fillStyle = bgGrad;
    ctx.fill();

    // 3. Optic Disc (Bright golden oval on nasal side)
    const discX = cx + radius * 0.45;
    const discY = cy - radius * 0.05;
    const discR = radius * 0.18;

    const discGrad = ctx.createRadialGradient(discX, discY, 5, discX, discY, discR);
    discGrad.addColorStop(0, '#fffae0');
    discGrad.addColorStop(0.6, '#ffd685');
    discGrad.addColorStop(1.0, '#e09830');
    ctx.beginPath();
    ctx.ellipse(discX, discY, discR, discR * 1.1, 0, 0, Math.PI * 2);
    ctx.fillStyle = discGrad;
    ctx.fill();

    // 4. Macula & Fovea (Darker avascular area on temporal side)
    const maculaX = cx - radius * 0.35;
    const maculaY = cy + radius * 0.05;
    const maculaR = radius * 0.22;

    const macGrad = ctx.createRadialGradient(maculaX, maculaY, 2, maculaX, maculaY, maculaR);
    macGrad.addColorStop(0, '#420a02');
    macGrad.addColorStop(0.7, '#85200c');
    macGrad.addColorStop(1.0, 'transparent');
    ctx.beginPath();
    ctx.arc(maculaX, maculaY, maculaR, 0, Math.PI * 2);
    ctx.fillStyle = macGrad;
    ctx.fill();

    // 5. Retinal Blood Vessel Tree (Branching from Optic Cup)
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#520803';
    ctx.lineCap = 'round';

    const drawVesselBranch = (startX, startY, angle, length, depth, width) => {
      if (depth <= 0) return;
      const endX = startX + Math.cos(angle) * length;
      const endY = startY + Math.sin(angle) * length;

      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.quadraticCurveTo(
        startX + Math.cos(angle + 0.2) * (length * 0.5),
        startY + Math.sin(angle + 0.2) * (length * 0.5),
        endX, endY
      );
      ctx.lineWidth = width;
      ctx.stroke();

      // Branching angles
      drawVesselBranch(endX, endY, angle - 0.45, length * 0.72, depth - 1, width * 0.7);
      drawVesselBranch(endX, endY, angle + 0.40, length * 0.68, depth - 1, width * 0.7);
    };

    // Major vascular arcades (Superior temporal, inferior temporal, superior nasal, inferior nasal)
    drawVesselBranch(discX, discY, Math.PI * 1.15, radius * 0.4, 4, 6);
    drawVesselBranch(discX, discY, Math.PI * 0.82, radius * 0.4, 4, 6);
    drawVesselBranch(discX, discY, Math.PI * 1.45, radius * 0.35, 3, 4.5);
    drawVesselBranch(discX, discY, Math.PI * 0.55, radius * 0.35, 3, 4.5);

    // 6. Inject Clinical Lesions according to ICDR Severity Grade
    if (grade >= 1) { // Mild NPDR: Microaneurysms
      this._drawMicroaneurysms(ctx, cx, cy, radius, 8 + grade * 6);
    }
    if (grade >= 2) { // Moderate NPDR: Hard Exudates & Hemorrhages
      this._drawHardExudates(ctx, maculaX, maculaY, radius * 0.4, 12);
      this._drawHemorrhages(ctx, cx, cy, radius, 10);
    }
    if (grade >= 3) { // Severe NPDR: Soft Exudates (Cotton Wool Spots) & Venous Beading
      this._drawCottonWoolSpots(ctx, cx, cy, radius, 5);
      this._drawHemorrhages(ctx, cx, cy, radius, 22);
    }
    if (grade >= 4) { // Proliferative DR: Neovascularization (Fragile new vessels) & Vitreous Bleeding
      this._drawNeovascularization(ctx, discX, discY, radius * 0.3);
      this._drawPreretinalHemorrhage(ctx, cx, cy, radius * 0.25);
    }

    ctx.restore();
  },

  _drawMicroaneurysms(ctx, cx, cy, radius, count) {
    ctx.fillStyle = '#660000';
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = (Math.random() * 0.6 + 0.1) * radius;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      const spotR = Math.random() * 2.5 + 1.5;

      ctx.beginPath();
      ctx.arc(x, y, spotR, 0, Math.PI * 2);
      ctx.fill();
    }
  },

  _drawHardExudates(ctx, maculaX, maculaY, spread, count) {
    ctx.fillStyle = '#fffaaa';
    for (let i = 0; i < count; i++) {
      const x = maculaX + (Math.random() - 0.5) * spread * 2;
      const y = maculaY + (Math.random() - 0.5) * spread * 2;
      const exR = Math.random() * 3 + 2;

      ctx.beginPath();
      ctx.arc(x, y, exR, 0, Math.PI * 2);
      ctx.fill();
    }
  },

  _drawHemorrhages(ctx, cx, cy, radius, count) {
    ctx.fillStyle = '#4a0000';
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = (Math.random() * 0.65 + 0.1) * radius;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      const rx = Math.random() * 6 + 3;
      const ry = Math.random() * 4 + 2;

      ctx.beginPath();
      ctx.ellipse(x, y, rx, ry, Math.random(), 0, Math.PI * 2);
      ctx.fill();
    }
  },

  _drawCottonWoolSpots(ctx, cx, cy, radius, count) {
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = (Math.random() * 0.5 + 0.2) * radius;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      const cR = Math.random() * 10 + 8;

      const grad = ctx.createRadialGradient(x, y, 1, x, y, cR);
      grad.addColorStop(0, 'rgba(255, 255, 240, 0.85)');
      grad.addColorStop(0.6, 'rgba(240, 240, 220, 0.4)');
      grad.addColorStop(1, 'transparent');

      ctx.beginPath();
      ctx.arc(x, y, cR, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    }
  },

  _drawNeovascularization(ctx, discX, discY, spread) {
    ctx.strokeStyle = '#ff3333';
    ctx.lineWidth = 1.2;
    for (let i = 0; i < 16; i++) {
      const angle = Math.random() * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(discX, discY);
      ctx.lineTo(discX + Math.cos(angle) * spread, discY + Math.sin(angle) * spread);
      ctx.stroke();
    }
  },

  _drawPreretinalHemorrhage(ctx, cx, cy, spread) {
    const x = cx - spread * 0.5;
    const y = cy - spread * 0.5;
    const grad = ctx.createRadialGradient(x, y, 2, x, y, spread);
    grad.addColorStop(0, '#590000');
    grad.addColorStop(0.8, '#3b0000');
    grad.addColorStop(1, 'transparent');

    ctx.beginPath();
    ctx.arc(x, y, spread, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
  },

  /**
   * Filter 1: Green Channel Extraction
   */
  applyGreenChannel(sourceCanvas, targetCanvas) {
    const sCtx = sourceCanvas.getContext('2d');
    const tCtx = targetCanvas.getContext('2d');
    const w = sourceCanvas.width;
    const h = sourceCanvas.height;

    const imgData = sCtx.getImageData(0, 0, w, h);
    const data = imgData.data;

    for (let i = 0; i < data.length; i += 4) {
      const g = data[i + 1]; // Green channel
      data[i] = g;     // R
      data[i + 1] = g; // G
      data[i + 2] = g; // B
    }

    tCtx.putImageData(imgData, 0, 0);
  },

  /**
   * Filter 2: CLAHE Contrast Enhancement Simulation
   */
  applyCLAHE(sourceCanvas, targetCanvas) {
    const sCtx = sourceCanvas.getContext('2d');
    const tCtx = targetCanvas.getContext('2d');
    const w = sourceCanvas.width;
    const h = sourceCanvas.height;

    const imgData = sCtx.getImageData(0, 0, w, h);
    const data = imgData.data;

    // Simple adaptive contrast enhancement
    for (let i = 0; i < data.length; i += 4) {
      const g = data[i + 1];
      // S-curve contrast boost
      let enhanced = (g - 128) * 1.6 + 128;
      enhanced = Math.min(255, Math.max(0, enhanced));
      
      data[i] = enhanced * 0.9;
      data[i + 1] = enhanced;
      data[i + 2] = enhanced * 0.8;
    }

    tCtx.putImageData(imgData, 0, 0);
  },

  /**
   * Filter 3: Vessel Mask & Edge Enhancement
   */
  applyVesselMask(sourceCanvas, targetCanvas) {
    const sCtx = sourceCanvas.getContext('2d');
    const tCtx = targetCanvas.getContext('2d');
    const w = sourceCanvas.width;
    const h = sourceCanvas.height;

    const imgData = sCtx.getImageData(0, 0, w, h);
    const data = imgData.data;
    const outputData = tCtx.createImageData(w, h);
    const out = outputData.data;

    // Sobel edge detector on Green Channel
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = (y * w + x) * 4;

        const g00 = data[((y - 1) * w + (x - 1)) * 4 + 1];
        const g01 = data[((y - 1) * w + x) * 4 + 1];
        const g02 = data[((y - 1) * w + (x + 1)) * 4 + 1];
        const g10 = data[(y * w + (x - 1)) * 4 + 1];
        const g12 = data[(y * w + (x + 1)) * 4 + 1];
        const g20 = data[((y + 1) * w + (x - 1)) * 4 + 1];
        const g21 = data[((y + 1) * w + x) * 4 + 1];
        const g22 = data[((y + 1) * w + (x + 1)) * 4 + 1];

        const gx = (g02 + 2 * g12 + g22) - (g00 + 2 * g10 + g20);
        const gy = (g20 + 2 * g21 + g22) - (g00 + 2 * g01 + g02);

        let mag = Math.sqrt(gx * gx + gy * gy);
        mag = mag > 35 ? Math.min(255, mag * 2.2) : 0;

        out[idx] = 0;
        out[idx + 1] = mag; // Green vessel highlight
        out[idx + 2] = mag * 0.8;
        out[idx + 3] = 255;
      }
    }

    tCtx.putImageData(outputData, 0, 0);
  },

  /**
   * Filter 4: Lesion Overlay Markers
   */
  drawLesionBoundingBoxes(targetCanvas, grade = 0) {
    if (grade === 0) return;
    const ctx = targetCanvas.getContext('2d');
    const w = targetCanvas.width;
    const h = targetCanvas.height;
    const cx = w / 2;
    const cy = h / 2;

    ctx.save();
    ctx.lineWidth = 2;
    ctx.font = '11px sans-serif';

    const drawBox = (x, y, size, label, color) => {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.strokeRect(x - size / 2, y - size / 2, size, size);
      ctx.fillText(label, x - size / 2, y - size / 2 - 4);
    };

    if (grade >= 1) {
      drawBox(cx - 60, cy + 40, 24, 'Microaneurysm', '#eab308');
      drawBox(cx + 80, cy - 50, 24, 'Microaneurysm', '#eab308');
    }
    if (grade >= 2) {
      drawBox(cx - 120, cy + 20, 32, 'Hard Exudate', '#f97316');
      drawBox(cx - 90, cy + 80, 30, 'Hemorrhage', '#ef4444');
    }
    if (grade >= 3) {
      drawBox(cx + 40, cy - 100, 36, 'Cotton Wool Spot', '#38bdf8');
      drawBox(cx - 140, cy - 60, 34, 'Venous Beading', '#ef4444');
    }
    if (grade >= 4) {
      drawBox(cx + 110, cy - 20, 42, 'Neovascularization', '#dc2626');
    }

    ctx.restore();
  }
};
