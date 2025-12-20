(() => {
  const WIDTH = 1180;
  const HEIGHT = 720;
  const PAD = 22;

  const STATE_ABBR = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
  };

  const mapStatusEl = document.getElementById("mapStatus");
  const selectionTitleEl = document.getElementById("selectionTitle");
  const selectionMetaEl = document.getElementById("selectionMeta");
  const loadingEl = document.getElementById("loading");
  const errorEl = document.getElementById("error");
  const summaryEl = document.getElementById("summary");
  const conventionalEl = document.getElementById("conventional");
  const trunkedEl = document.getElementById("trunked");
  const talkgroupsEl = document.getElementById("talkgroups");

  const svg = d3
    .select("#mapSvg")
    .attr("viewBox", `0 0 ${WIDTH} ${HEIGHT}`)
    .attr("preserveAspectRatio", "xMidYMid meet");

  const gFit = svg.append("g");
  const path = d3.geoPath(d3.geoIdentity());

  const stateNames = new Map();
  const countyPathById = new Map();
  let selectedCountyId = null;

  function pad2(id) {
    return String(id).padStart(2, "0");
  }

  function pad5(id) {
    return String(id).padStart(5, "0");
  }

  function showMapStatus(msg) {
    mapStatusEl.textContent = msg || "";
    mapStatusEl.style.display = msg ? "grid" : "none";
  }

  function setHover(id) {
    countyPathById.forEach((el, cid) => {
      el.classed("is-hover", id && cid === id);
    });
  }

  function setSelected(id) {
    selectedCountyId = id;
    countyPathById.forEach((el, cid) => {
      el.classed("is-selected", cid === id);
    });
  }

  function formatCountyLabel(feature) {
    const fips = pad5(feature.id);
    const stateFips = fips.slice(0, 2);
    const stAbbr = STATE_ABBR[stateFips] || stateNames.get(stateFips) || stateFips;
    const countyName = feature.properties?.name || "County";
    return `${countyName}, ${stAbbr}`;
  }

  function fitToFeatures(features) {
    const bounds = path.bounds({ type: "FeatureCollection", features });
    const dx = bounds[1][0] - bounds[0][0];
    const dy = bounds[1][1] - bounds[0][1];
    if (!isFinite(dx) || !isFinite(dy) || dx <= 0 || dy <= 0) return;
    const scale = Math.min((WIDTH - PAD * 2) / dx, (HEIGHT - PAD * 2) / dy);
    const cx = (bounds[0][0] + bounds[1][0]) / 2;
    const cy = (bounds[0][1] + bounds[1][1]) / 2;
    const tx = WIDTH / 2 - scale * cx;
    const ty = HEIGHT / 2 - scale * cy;
    gFit.attr("transform", `translate(${tx},${ty}) scale(${scale})`);
  }

  function escapeHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function setLoading(isLoading) {
    loadingEl.classList.toggle("hidden", !isLoading);
  }

  function setError(msg) {
    errorEl.textContent = msg || "";
    errorEl.classList.toggle("hidden", !msg);
  }

  function renderSummary(data) {
    const convCount = data?.conventional?.length || 0;
    const trunkSystems = data?.trunked?.length || 0;
    const tgs = data?.talkgroups?.length || 0;
    summaryEl.innerHTML = `
      <div class="stat-row">
        <div class="stat"><div class="label">Conventional</div><div class="value">${convCount}</div></div>
        <div class="stat"><div class="label">Trunked Systems</div><div class="value">${trunkSystems}</div></div>
        <div class="stat"><div class="label">Talkgroups</div><div class="value">${tgs}</div></div>
      </div>
    `;
  }

  function formatFreq(r) {
    if (r.frequency_mhz) return `${Number(r.frequency_mhz).toFixed(3)} MHz`;
    if (r.frequency_hz) return `${(Number(r.frequency_hz) / 1e6).toFixed(3)} MHz`;
    return "-";
  }

  function renderConventional(rows) {
    if (!rows || rows.length === 0) {
      conventionalEl.innerHTML = `
        <div class="section">
          <h3>Conventional Channels</h3>
          <p class="muted">No conventional frequencies were found for this county.</p>
        </div>
      `;
      return;
    }

    const take = rows.slice(0, 80);
    const more = rows.length - take.length;
    const body = take
      .map(
        (r) => `
        <tr>
          <td>${escapeHtml(r.department || "")}</td>
          <td>${escapeHtml(r.channel || "")}</td>
          <td>${formatFreq(r)}</td>
          <td>${escapeHtml(r.modulation || "")}</td>
          <td>${escapeHtml(r.tone || r.nac || r.color_code || r.ran || "-")}</td>
        </tr>`
      )
      .join("");

    conventionalEl.innerHTML = `
      <div class="section">
        <h3>Conventional Channels</h3>
        <table class="freq-list">
          <thead>
            <tr><th>Department</th><th>Channel</th><th>Frequency</th><th>Mode</th><th>Tone/NAC</th></tr>
          </thead>
          <tbody>${body}</tbody>
        </table>
        ${more > 0 ? `<div class="muted" style="margin-top:6px;">+${more} more not shown</div>` : ""}
      </div>
    `;
  }

  function renderTrunked(systems) {
    if (!systems || systems.length === 0) {
      trunkedEl.innerHTML = `
        <div class="section">
          <h3>Trunked Systems</h3>
          <p class="muted">No trunked systems reported for this county.</p>
        </div>
      `;
      return;
    }

    const trimmed = systems.slice(0, 6);
    const more = systems.length - trimmed.length;
    const html = trimmed
      .map((sys) => {
        const sites = (sys.sites || []).slice(0, 3);
        const siteHtml =
          sites.length > 0
            ? sites
                .map((site) => {
                  const freqs = (site.frequencies || []).slice(0, 6);
                  const freqBadges =
                    freqs.length > 0
                      ? freqs
                          .map((f) => `<span class="badge">${formatFreq(f)}${f.channel_name ? ` · ${escapeHtml(f.channel_name)}` : ""}</span>`)
                          .join(" ")
                      : '<span class="muted">No site frequencies</span>';
                  return `
                    <div class="card">
                      <div class="meta-row" style="justify-content: space-between;">
                        <div><strong>${escapeHtml(site.site_name || "Site "+site.site_id)}</strong></div>
                        ${site.distance_miles ? `<span class="badge teal">${site.distance_miles.toFixed(1)} mi</span>` : ""}
                      </div>
                      <div class="meta-row" style="margin-top:6px;">${freqBadges}</div>
                    </div>
                  `;
                })
                .join("")
            : '<div class="card"><div class="muted">No sites for this system.</div></div>';

        return `
          <div class="card">
            <div class="meta-row" style="justify-content: space-between;">
              <div>
                <h4>${escapeHtml(sys.system_name || `System ${sys.trunk_id}`)}</h4>
                <div class="meta-row">${escapeHtml(sys.system_type || "")}</div>
              </div>
              <span class="badge orange">${sys.trunk_id}</span>
            </div>
            <div class="list" style="margin-top:8px;">${siteHtml}</div>
          </div>
        `;
      })
      .join("");

    trunkedEl.innerHTML = `
      <div class="section">
        <h3>Trunked Systems</h3>
        <div class="list">${html}</div>
        ${more > 0 ? `<div class="muted" style="margin-top:6px;">+${more} more systems not shown</div>` : ""}
      </div>
    `;
  }

  function renderTalkgroups(rows) {
    if (!rows || rows.length === 0) {
      talkgroupsEl.innerHTML = `
        <div class="section">
          <h3>Talkgroups</h3>
          <p class="muted">No talkgroups reported for this county.</p>
        </div>
      `;
      return;
    }

    const take = rows.slice(0, 80);
    const more = rows.length - take.length;
    const body = take
      .map(
        (tg) => `
        <tr>
          <td>${escapeHtml(tg.system_name || tg.trunk_id)}</td>
          <td>${escapeHtml(tg.category || "")}</td>
          <td>${escapeHtml(tg.alpha_tag || tg.talkgroup)}</td>
          <td>${escapeHtml(tg.talkgroup || "")}</td>
          <td>${escapeHtml(tg.service || "")}</td>
        </tr>`
      )
      .join("");

    talkgroupsEl.innerHTML = `
      <div class="section">
        <h3>Talkgroups</h3>
        <table class="tg-list">
          <thead><tr><th>System</th><th>Category</th><th>Alpha Tag</th><th>ID</th><th>Service</th></tr></thead>
          <tbody>${body}</tbody>
        </table>
        ${more > 0 ? `<div class="muted" style="margin-top:6px;">+${more} more not shown</div>` : ""}
      </div>
    `;
  }

  async function fetchCounty(stateAbbr, countyName) {
    setLoading(true);
    setError("");
    summaryEl.innerHTML = "";
    conventionalEl.innerHTML = "";
    trunkedEl.innerHTML = "";
    talkgroupsEl.innerHTML = "";

    try {
      const url = `/hpdb/query?state=${encodeURIComponent(stateAbbr)}&county=${encodeURIComponent(countyName)}&include_trunk_site_frequencies=true`;
      const res = await fetch(url);
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`API error (${res.status}): ${txt}`);
      }
      const data = await res.json();
      renderSummary(data);
      renderConventional(data.conventional || []);
      renderTrunked(data.trunked || []);
      renderTalkgroups(data.talkgroups || []);
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to fetch radio data.");
    } finally {
      setLoading(false);
    }
  }

  function handleCountyClick(feature) {
    const fips = pad5(feature.id);
    const stateFips = fips.slice(0, 2);
    const stAbbr = STATE_ABBR[stateFips] || stateNames.get(stateFips) || stateFips;
    const stName = stateNames.get(stateFips) || stAbbr;
    const countyName = (feature.properties?.name || "").trim();

    setSelected(fips);
    selectionTitleEl.textContent = `${countyName} County, ${stAbbr}`;
    selectionMetaEl.textContent = `FIPS ${fips} • ${stName}`;
    fetchCounty(stAbbr, countyName);
  }

  async function initMap() {
    try {
      showMapStatus("Loading map data…");
      const [stateTopo, countyTopo] = await Promise.all([
        d3.json("/static/data/states-albers-10m.json"),
        d3.json("/static/data/counties-10m.json"),
      ]);

      const nation = topojson.feature(stateTopo, stateTopo.objects.nation);
      const states = topojson.feature(stateTopo, stateTopo.objects.states).features;
      const counties = topojson.feature(countyTopo, countyTopo.objects.counties).features;

      states.forEach((s) => stateNames.set(pad2(s.id), s.properties?.name || ""));

      fitToFeatures(states);

      gFit
        .append("path")
        .datum(nation)
        .attr("class", "nation-outline")
        .attr("d", path);

      gFit
        .append("g")
        .selectAll("path")
        .data(states)
        .join("path")
        .attr("class", "state")
        .attr("d", (d) => path(d));

      const countyPaths = gFit
        .append("g")
        .selectAll("path")
        .data(counties)
        .join("path")
        .attr("class", "county")
        .attr("d", (d) => path(d))
        .on("mouseenter", (_, d) => setHover(pad5(d.id)))
        .on("mouseleave", () => setHover(null))
        .on("click", (_, d) => handleCountyClick(d));

      countyPaths.append("title").text((d) => formatCountyLabel(d));
      countyPaths.each(function (d) {
        countyPathById.set(pad5(d.id), d3.select(this));
      });

      showMapStatus("");
    } catch (err) {
      console.error(err);
      showMapStatus("Could not load map geometry. Check static assets.");
    }
  }

  function init() {
    selectionTitleEl.textContent = "Pick a county";
    selectionMetaEl.textContent = "Click any county outline to load HPDB data.";
    initMap();
  }

  init();
})();
