let benchmarkData = [];

async function loadBenchmark() {
  try {
    const response = await fetch("benchmark.json");

    const data = await response.json();

    benchmarkData = data.scores || [];

    renderStats();
    renderTable(benchmarkData);
    renderCharts();

    initializeFilters();
  } catch (err) {
    console.error(err);

    alert("Failed to load benchmark.json");
  }
}

// function renderStats() {
//   const models = new Set(benchmarkData.map((d) => d.evaluated_model));

//   const avg =
//     benchmarkData.reduce((a, b) => a + b.total_score, 0) / benchmarkData.length;

//   const correct = benchmarkData.filter((d) => d.verdict === "CORRECT").length;

//   const correctRate = ((correct / benchmarkData.length) * 100).toFixed(1);

//   document.getElementById("totalModels").innerText = models.size;

//   document.getElementById("totalSamples").innerText = benchmarkData.length;

//   document.getElementById("avgScore").innerText = avg.toFixed(2);

//   document.getElementById("correctRate").innerText = correctRate + "%";
// }

function renderStats() {
  const models = new Set(benchmarkData.map((d) => d.evaluated_model));

  const avg =
    benchmarkData.reduce((a, b) => a + b.total_score, 0) / benchmarkData.length;

  let weightedCorrect = 0;

  console.log(benchmarkData);

  benchmarkData.forEach((d) => {
    if (d.verdict === "CORRECT") {
      weightedCorrect += 1.0;
    } else if (d.verdict === "PARTIALLY_CORRECT") {
      weightedCorrect += 1.0;
    }
  });

  console.log("Weighted Correct", weightedCorrect);
  console.log("Length", benchmarkData.length);

  const weightedAccuracy = (
    (weightedCorrect / benchmarkData.length) *
    100
  ).toFixed(1);

  document.getElementById("totalModels").innerText = models.size;

  document.getElementById("totalSamples").innerText =
    benchmarkData.length.toLocaleString();

  document.getElementById("avgScore").innerText = avg.toFixed(2);

  //   document.getElementById("correctRate").innerText = weightedAccuracy + "%";
}

function renderTable(data) {
  const tbody = document.getElementById("tableBody");

  tbody.innerHTML = "";

  const grouped = {};

  data.forEach((row) => {
    const model = row.evaluated_model;

    if (!grouped[model]) {
      grouped[model] = {
        total: 0,

        score: 0,

        weightedCorrect: 0,

        correct: 0,

        partial: 0,

        incorrect: 0,
      };
    }

    grouped[model].total += 1;

    grouped[model].score += row.total_score;

    if (row.verdict === "CORRECT") {
      grouped[model].correct += 1;

      grouped[model].weightedCorrect += 1.0;
    } else if (row.verdict === "PARTIALLY_CORRECT") {
      grouped[model].partial += 1;

      grouped[model].weightedCorrect += 0.5;
    } else {
      grouped[model].incorrect += 1;
    }
  });

  const leaderboard = Object.entries(grouped)

    .map(([model, stats]) => {
      const weightedAccuracy = (
        (stats.weightedCorrect / stats.total) *
        100
      ).toFixed(1);

      return {
        model,

        avgScore: (stats.score / stats.total).toFixed(2),

        weightedAccuracy,

        correctRate: ((stats.correct / stats.total) * 100).toFixed(1),

        partialRate: ((stats.partial / stats.total) * 100).toFixed(1),

        incorrectRate: ((stats.incorrect / stats.total) * 100).toFixed(1),
      };
    })

    .sort(
      (a, b) => parseFloat(b.weightedAccuracy) - parseFloat(a.weightedAccuracy),
    );

  leaderboard.forEach((row, index) => {
    tbody.innerHTML += `
      <tr>

        <td>
          #${index + 1}
        </td>

        <td>
          ${row.model}
        </td>

        <td>
          ${row.avgScore}
        </td>

        <td class="correct">
          ${row.weightedAccuracy}%
        </td>

        <td>
          ${row.correctRate}%
        </td>

        <td>
          ${row.partialRate}%
        </td>

        <td class="incorrect">
          ${row.incorrectRate}%
        </td>

      </tr>
    `;
  });
}

function renderCharts() {
  const verdictCounts = {};

  benchmarkData.forEach((d) => {
    verdictCounts[d.verdict] = (verdictCounts[d.verdict] || 0) + 1;
  });

  new Chart(document.getElementById("verdictChart"), {
    type: "doughnut",

    data: {
      labels: Object.keys(verdictCounts),

      datasets: [
        {
          data: Object.values(verdictCounts),

          backgroundColor: ["#22c55e", "#eab308", "#ef4444"],
        },
      ],
    },
  });

  const grouped = {};

  benchmarkData.forEach((d) => {
    if (!grouped[d.evaluated_model]) {
      grouped[d.evaluated_model] = [];
    }

    grouped[d.evaluated_model].push(d.total_score);
  });

  const labels = Object.keys(grouped);

  const averages = labels.map((model) => {
    const arr = grouped[model];

    return (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2);
  });

  new Chart(document.getElementById("scoreChart"), {
    type: "bar",

    data: {
      labels,

      datasets: [
        {
          label: "Average Score",
          data: averages,
          backgroundColor: "#2563eb",
        },
      ],
    },

    options: {
      responsive: true,

      plugins: {
        legend: {
          display: false,
        },
      },
    },
  });
}

function initializeFilters() {
  //   document
  //     .getElementById("searchInput")
  //     .addEventListener("input", applyFilters);
  //   document
  //     .getElementById("verdictFilter")
  //     .addEventListener("change", applyFilters);
}

function applyFilters() {
  const search = document.getElementById("searchInput").value.toLowerCase();

  const verdict = document.getElementById("verdictFilter").value;

  const filtered = benchmarkData.filter((row) => {
    const matchSearch = row.evaluated_model.toLowerCase().includes(search);

    const matchVerdict = verdict === "ALL" || row.verdict === verdict;

    return matchSearch && matchVerdict;
  });

  renderTable(filtered);
}

loadBenchmark();
