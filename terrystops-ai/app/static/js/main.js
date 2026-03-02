/**
 * TerryStops-AI  –  main.js
 * Gauge animation, form wizard, AJAX helpers, chart utilities.
 */

/* ── Gauge ────────────────────────────────────────────── */

function lerpColor(a, b, t) {
  const ah = parseInt(a.replace('#', ''), 16);
  const bh = parseInt(b.replace('#', ''), 16);
  const ar = (ah >> 16) & 0xff, ag = (ah >> 8) & 0xff, ab = ah & 0xff;
  const br = (bh >> 16) & 0xff, bg = (bh >> 8) & 0xff, bb = bh & 0xff;
  const rr = Math.round(ar + (br - ar) * t);
  const rg = Math.round(ag + (bg - ag) * t);
  const rb = Math.round(ab + (bb - ab) * t);
  return '#' + ((1 << 24) + (rr << 16) + (rg << 8) + rb).toString(16).slice(1);
}

function gaugeColor(pct) {
  if (pct < 0.5) return lerpColor('#00d4aa', '#ffc107', pct * 2);
  return lerpColor('#ffc107', '#6c63ff', (pct - 0.5) * 2);
}

function drawGauge(canvasId, value, maxValue) {
  maxValue = maxValue || 100;
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;
  const radius = Math.min(cx, cy) - 16;
  const startAngle = 0.75 * Math.PI;
  const endAngle   = 2.25 * Math.PI;
  const target = Math.min(value / maxValue, 1);
  let current = 0;

  function frame() {
    current += (target - current) * 0.06;
    if (Math.abs(current - target) < 0.002) current = target;

    ctx.clearRect(0, 0, W, H);

    // background arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.lineWidth = 14;
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineCap = 'round';
    ctx.stroke();

    // value arc
    const sweep = startAngle + current * (endAngle - startAngle);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, sweep);
    ctx.lineWidth = 14;
    ctx.strokeStyle = gaugeColor(current);
    ctx.lineCap = 'round';
    ctx.stroke();

    // text
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 32px Segoe UI, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(Math.round(current * maxValue) + '%', cx, cy - 4);

    ctx.font = '12px Segoe UI, sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText('Arrest Probability', cx, cy + 22);

    if (current !== target) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

/* ── Form wizard ──────────────────────────────────────── */

function initWizard() {
  const steps = document.querySelectorAll('.wizard-step');
  const dots  = document.querySelectorAll('.step-dot');
  if (!steps.length) return;

  let idx = 0;
  show(0);

  document.querySelectorAll('.btn-next').forEach(function (btn) {
    btn.addEventListener('click', function () { go(idx + 1); });
  });
  document.querySelectorAll('.btn-prev').forEach(function (btn) {
    btn.addEventListener('click', function () { go(idx - 1); });
  });

  function go(n) {
    if (n < 0 || n >= steps.length) return;
    idx = n;
    show(idx);
  }

  function show(n) {
    steps.forEach(function (s, i) {
      s.classList.toggle('active', i === n);
    });
    dots.forEach(function (d, i) {
      d.classList.toggle('active', i <= n);
    });
  }
}

/* ── AJAX prediction ──────────────────────────────────── */

function submitPrediction(formEl, resultEl) {
  var formData = new FormData(formEl);
  var submitBtn = formEl.querySelector('button[type="submit"]');
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Processing…';
  }

  fetch(formEl.action || '/predict', {
    method: 'POST',
    body: formData
  })
  .then(function (r) { return r.text(); })
  .then(function (html) {
    var doc = new DOMParser().parseFromString(html, 'text/html');
    var newResult = doc.getElementById('predictionResult');
    if (newResult && resultEl) {
      resultEl.innerHTML = newResult.innerHTML;
      resultEl.style.display = 'block';
      /* trigger gauge if data attribute present */
      var prob = resultEl.getAttribute('data-probability');
      if (prob) drawGauge('gaugeCanvas', parseFloat(prob), 100);
    }
  })
  .catch(function () {
    if (resultEl) {
      resultEl.innerHTML = '<div class="alert alert-danger">Request failed. Please try again.</div>';
      resultEl.style.display = 'block';
    }
  })
  .finally(function () {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = '<i class="fas fa-search me-1"></i>Predict';
    }
  });
}

/* ── Chart helpers ────────────────────────────────────── */

var plotlyDarkLayout = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font: { color: '#e8e8f0', family: 'Inter, Segoe UI, sans-serif' },
  margin: { t: 40, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: 'rgba(108, 99, 255, 0.06)', zerolinecolor: 'rgba(255,255,255,0.08)' },
  yaxis: { gridcolor: 'rgba(108, 99, 255, 0.06)', zerolinecolor: 'rgba(255,255,255,0.08)' }
};

function loadChart(url, elementId, buildFn) {
  var el = document.getElementById(elementId);
  if (!el) return;
  el.innerHTML = '<div class="skeleton skeleton-chart"></div>';
  fetch(url)
    .then(function (r) { return r.json(); })
    .then(function (data) {
      el.innerHTML = '';
      buildFn(data, el);
    })
    .catch(function () {
      el.innerHTML = '<p class="text-center text-muted mt-4">Unable to load chart data.</p>';
    });
}

/* ── Sidebar toggle (mobile) ──────────────────────────── */

function toggleSidebar() {
  document.querySelector('.sidebar').classList.toggle('show');
}

/* ── Init ─────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', function () {
  initWizard();

  /* close sidebar on outside click (mobile) */
  document.addEventListener('click', function (e) {
    var sb = document.querySelector('.sidebar');
    var toggler = document.querySelector('.sidebar-toggler');
    if (sb && sb.classList.contains('show') && !sb.contains(e.target) && (!toggler || !toggler.contains(e.target))) {
      sb.classList.remove('show');
    }
  });
});
