const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');

navToggle?.addEventListener('click', () => {
  const open = navLinks.classList.toggle('open');
  navToggle.setAttribute('aria-expanded', String(open));
});

navLinks?.addEventListener('click', (event) => {
  if (event.target.matches('a')) {
    navLinks.classList.remove('open');
    navToggle?.setAttribute('aria-expanded', 'false');
  }
});

const copyButton = document.querySelector('#copy-citation');
copyButton?.addEventListener('click', async () => {
  const citation = document.querySelector('#citation')?.textContent ?? '';
  await navigator.clipboard.writeText(citation);
  copyButton.textContent = 'Copied';
  window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
});

if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const teaser = document.querySelector('#hero-teaser');
  if (teaser?.dataset.staticSrc) teaser.src = teaser.dataset.staticSrc;
}

const resultData = {
  overall: [
    { model: 'SeedDance 2.0', srp: .439, rel: .785, place: .244, core: .156 },
    { model: 'Wan 2.7', srp: .412, rel: .853, place: .272, core: .163 },
    { model: 'Kling 3.0', srp: .409, rel: .529, place: .402, core: .312 },
    { model: 'CosmosPolicy · BenchCam', srp: .408, rel: .823, place: .234, core: .125 },
    { model: 'Hailuo 2.3', srp: .387, rel: .763, place: .251, core: .156 },
    { model: 'Veo 3.1', srp: .345, rel: .820, place: .228, core: .120 }
  ],
  standard: [
    { model: 'SeedDance 2.0', srp: .445, rel: .821, place: .247, core: .138 },
    { model: 'CosmosPolicy · BenchCam', srp: .420, rel: .781, place: .245, core: .138 },
    { model: 'Wan 2.7', srp: .410, rel: .869, place: .246, core: .138 },
    { model: 'Kling 3.0', srp: .396, rel: .506, place: .410, core: .325 },
    { model: 'Hailuo 2.3', srp: .365, rel: .754, place: .238, core: .138 },
    { model: 'Veo 3.1', srp: .331, rel: .809, place: .233, core: .125 }
  ],
  enhanced: [
    { model: 'SeedDance 2.0', srp: .433, rel: .749, place: .242, core: .175 },
    { model: 'Kling 3.0', srp: .423, rel: .552, place: .393, core: .300 },
    { model: 'Wan 2.7', srp: .414, rel: .836, place: .299, core: .188 },
    { model: 'Hailuo 2.3', srp: .410, rel: .773, place: .263, core: .175 },
    { model: 'CosmosPolicy · BenchCam', srp: .397, rel: .865, place: .222, core: .113 },
    { model: 'Veo 3.1', srp: .359, rel: .830, place: .223, core: .115 }
  ]
};

const leaderboard = document.querySelector('#leaderboard');
const viewLabel = document.querySelector('#view-label');

function metricBar(label, score) {
  return `<div class="subgoal"><span>${label}</span><div class="metric-track"><i class="metric-fill" style="--score:${score}"></i></div></div>`;
}

function renderLeaderboard(view) {
  const rows = resultData[view] ?? resultData.overall;
  leaderboard.innerHTML = rows.map((row, index) => `
    <article class="leaderboard-row">
      <div class="model-cell"><span class="rank">${String(index + 1).padStart(2, '0')}</span><strong>${row.model}</strong></div>
      <div class="primary-metric">
        <div class="metric-track"><i class="metric-fill" style="--score:${row.srp}"></i></div>
        <span class="metric-value">${row.srp.toFixed(3)}</span>
      </div>
      <div class="subgoals">
        ${metricBar('Rel', row.rel)}
        ${metricBar('Plc', row.place)}
        ${metricBar('Core', row.core)}
      </div>
    </article>
  `).join('');
  viewLabel.textContent = view.charAt(0).toUpperCase() + view.slice(1);
}

document.querySelectorAll('[data-result-view]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-result-view]').forEach((item) => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-selected', String(active));
    });
    renderLeaderboard(button.dataset.resultView);
  });
});

renderLeaderboard('overall');
