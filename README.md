
 🌾 फसल-मित्र | Fasal-Mitra

From Space to Soil: Satellite-Based Crop Intelligence & Agricultural Risk Screening

हर खेत की कहानी, डेटा की ज़ुबानी। 
Har Khet Ki Kahaani, Data Ki Zubaani.

Machine Learning · Google Earth Engine · Remote Sensing · Precision Agriculture · Geospatial Analytics

[**🚀 Open Live Earth Engine App**](https://pheno-crop-analytics.projects.earthengine.app/view/fasal-mitra) · [**📚 Explore Methodology**](#-machine-learning-methodology) · [**🛰️ Datasets**](#-data-sources) · [**📈 Validation**](#-evaluation-and-scientific-validity) · [**🛠️ Setup**](#-installation-and-running-the-project)

Study area:- Varanasi district, Uttar Pradesh, India  
Project developer: Pushkkal Gupta  
Project type:-machine-learning demonstrator

## 📑 Contents

1. [Project at a glance](#-project-at-a-glance)
2. [The problem: why this project exists](#-the-problem-why-this-project-exists)
3. [What Fasal-Mitra does](#-what-fasal-mitra-does)
4. [Who benefits, and how](#-who-benefits-and-how)
5. [Where and when the system operates](#-where-and-when-the-system-operates)
6. [How it works](#-how-it-works-end-to-end)
7. [Data sources](#-data-sources)
8. [Machine-learning methodology](#-machine-learning-methodology)
9. [Platform features and implementation status](#-platform-features-and-implementation-status)
10. [Technical architecture and stack](#-technical-architecture-and-stack)
11. [Novelty and innovation](#-novelty-and-innovation)
12. [Feasibility and constraints](#-feasibility-and-constraints)
13. [Scalability and production roadmap](#-scalability-and-production-roadmap)
14. [Evaluation and scientific validity](#-evaluation-and-scientific-validity)
15. [Installation and running the project](#-installation-and-running-the-project)
16. [Deployment and publishing](#-deployment-and-publishing)
17. [Demonstration guide](#-three-minute-competition-demo)
18. [Limitations, ethics, and security](#-limitations-ethics-and-security)
19. [Future work](#-future-work)
20. [References and project information](#-references-and-project-information)

---

## 🎯 Project at a glance

| Question | Answer |
|---|---|
| **What?** | A satellite-based, geospatial machine-learning and agricultural monitoring platform that classifies crop categories and helps examine changes in vegetation, moisture proxies, and potential environmental risks. |
| **Why?** | Field conditions can vary over time and across large agricultural areas. Traditional field visits alone are time-consuming, while satellite data is difficult for non-specialists to interpret. |
| **Where?** | The **administrative polygon of Varanasi district**, Uttar Pradesh, India, using the FAO GAUL 2015 level-2 boundary. |
| **When?** | Users choose a satellite-analysis period. Imagery and trends are dated by actual acquisitions, not presented as a continuous live sensor feed. Seasonal comparisons use matched calendar windows. |
| **How?** | Google Earth Engine processes optical/radar imagery and weather records; spectral/temporal features feed a supervised **Random Forest** classifier; separate calculated indices and rules power monitoring and risk-screening tools. |
| **For whom?** | Agricultural researchers, extension workers, survey teams, planners, and farmers who need more accessible **screening and investigation** information. |
| **Evidence today?** | The source includes 260 manually tagged sample locations across eight classes; real deployment metrics and independent ground-truth validation are **not provided**. |
| **Primary output?** | Interactive district/field maps, crop-category estimates, quality and validation views, temporal charts, potential-risk screens, and exportable evidence. |

### The central idea

**“Har khet ko visit karna har din possible nahi. Satellite observations can help us identify *where to look first*.”**

Fasal-Mitra turns complex Earth-observation datasets into interpretable maps and decision-support evidence. It is designed to **augment**, not replace, field visits and professional agronomic assessment.

---

## 🚜 The problem: why this project exists

Agricultural monitoring faces several connected operational challenges:

| Challenge | Why it matters | Fasal-Mitra's approach |
|---|---|---|
| **Limited visibility** | It is difficult to inspect large numbers of fields frequently. | District-wide and field-level satellite views. |
| **Crop identification uncertainty** | Aggregate records may not show what appears to be growing at each location. | Supervised, eight-class crop/land-use classification. |
| **Changing vegetation conditions** | A one-time image does not describe growth or deterioration. | NDVI/NDMI and seasonal time-series comparisons. |
| **Weather-related stress** | Possible waterlogging or rainfall deficits may warrant a timely field visit. | Multi-source screening using SAR/optical imagery, rainfall anomalies, and vegetation trends. |
| **Hard-to-interpret remote sensing** | Data catalogs, bands, and algorithms are not intuitive for general users. | Interactive legends, explanations, field cards, guided demo, and multilingual introduction. |
| **Insufficient evidence traceability** | A prediction without its time, source, quality, or validation context can mislead. | Observation provenance, model diagnostics, uncertainty cues, field-evidence exports. |

**Problem statement:** *How can multi-source satellite observations and machine learning support interpretable, scalable, and scientifically responsible crop monitoring for Varanasi district?*

### Project objectives

- Classify a limited set of crop and land-use categories from satellite-derived features.
- Monitor changes in vegetation indicators and surface/canopy moisture proxies.
- Help identify **areas for investigation**, not assert unverified agronomic diagnoses.
- Present data acquisition dates, observation coverage, model quality, and uncertainty.
- Enable field-level selection, comparisons, and exportable evidence.
- Demonstrate an accessible, multilingual entry point for non-specialist visitors.
- Establish an extensible foundation for future field verification and operational deployment.

---

## 🌱 What Fasal-Mitra does

### Eight supported classification categories

| Category | हिन्दी | Original manually tagged points |
|---|---|---:|
| Rice | धान | 40 |
| Wheat | गेहूँ | 40 |
| Sugarcane | गन्ना | 40 |
| Maize | मक्का | 20 |
| Gram | चना | 20 |
| Mustard | सरसों | 20 |
| Vegetable | सब्जियाँ | 40 |
| Fallow | परती / खाली खेत | 40 |
| **Total** | | **260** |

**Important:** `Fallow` is a land-use state rather than a crop species. These are **manually labelled sample locations**, not a complete ground-truth survey of Varanasi. The source contains a proximity-conflict screen for eight closely spaced cross-class points, while district filtering and valid imagery can further change the usable sample count. Exact training/test counts must be reported from the run being evaluated.

### Main outputs

1. **Crop classification maps:** estimated crop/land-use class at eligible locations/pixels after the supervised model is trained.
2. **Vegetation and moisture indicators:** NDVI, NDMI, NDWI, NDRE, EVI, SAVI, MNDWI, BSI, etc., interpreted cautiously.
3. **Historical comparisons:** monthly/seasonal charts and matched-period NDVI change layers.
4. **Agricultural risk screening:** possible surface-water signals, regional rainfall anomalies, and vegetation-decline flags.
5. **Field intelligence:** user-selected field geometry, observations, map inspection, comparison, and an Evidence Passport export.
6. **Evidence and quality tools:** scene coverage, imagery freshness, uncertainty indicators, confusion-matrix/evaluation workflows, and spatial cross-validation.
7. **Project education:** separate photo-led website with English, हिंदी, and Hinglish introduction to the scientific workflow.

---

## 👥 Who benefits, and how

| User | Example workflow | Potential value (not a measured outcome) |
|---|---|---|
| **Farmer or field representative** | View a field's available imagery and vegetation trend. | Helps prioritize questions to ask or fields to inspect. |
| **Agricultural extension officer** | Filter apparent changes and inspect supporting satellite dates. | More focused field-visit planning. |
| **Research student** | Run a classifier, compare features, inspect holdout metrics. | Transparent geospatial ML experimentation. |
| **District-level planner** | Look at crop categories and possible weather-related stress zones. | Exploratory spatial context, subject to validation. |
| **Field-survey team** | Review uncertain predictions and collect verified labels. | Better future training data and auditability. |

No reduction in farmer losses, yield increase, drought-detection precision, or monetary savings is claimed without field trials.

---

## 📍 Where and when the system operates

### Geographic scope

- **Country:** India
- **State:** Uttar Pradesh
- **District:** Varanasi
- **AOI:** Official FAO GAUL 2015 level-2 **district polygon**, **not** a manually drawn bounding rectangle.
- **Coordinate systems:** Earth Engine works with geospatial imagery and geometry; project processing uses UTM Zone 44N (`EPSG:32644`) where configured, with geographic export options (`EPSG:4326`).

The default source date selection is configurable. In the repository's v26 Earth Engine script, the configured analysis end date is **2026-09-30**, with date windows selectable by the user. The interface must communicate the most recent usable **satellite acquisition date** rather than implying that the satellite scene is live video.

### Seasonality

Different crops grow in different Kharif/Rabi/Zaid periods. Seasonal context helps interpret spectral signatures, but a generic crop calendar **cannot independently determine an individual field's sowing, growth, or harvest date**. Multi-season classification is most meaningful when training and validation labels match the chosen observation season.

---

## 🔄 How it works: end-to-end

mermaid flowchart 
  A[Varanasi district boundary] --> B[Earth Engine data acquisition]
  S[Sentinel-2 optical] --> B
  R[Sentinel-1 SAR] --> B
  W[CHIRPS and ERA5-Land] --> B
  E[SRTM and reference land cover] --> B
  B --> C[Cloud masking and temporal composites]
  C --> D[Spectral radar terrain and time-series features]
  L[Manually tagged sample locations] --> F[Label screening and train-test partition]
  D --> F
  F --> G[Random Forest crop classifier]
  G --> H[Classification and tree-vote uncertainty maps]
  G --> I[Holdout and spatial validation]
  D --> J[Vegetation indices and rule-based screening]
  W --> J
  H --> K[Interactive map and field intelligence]
  I --> K
  J --> K
  K --> M[Review and Evidence Passport export]
  M -. optional manual import .-> P[Separate FastAPI and PostGIS foundation]

### Step-by-step explanation

1. **Choose a location and date range.** The application uses the Varanasi district polygon as the scientific AOI.
2. **Acquire imagery.** Query Sentinel-2 surface reflectance, Sentinel-1 radar, and supporting environmental datasets in Earth Engine.
3. **Prepare data.** Mask clouds/invalid optical observations, combine imagery by time window, compute optical and radar-derived features, and retain missing-observation indicators.
4. **Prepare labels.** Verify that manual points are in the selected district and have usable feature values; screen conflicts and avoid treating many nearby pixels as independent field observations.
5. **Train the model.** Fit supervised Random Forest on the training partition; reserve held-out points for evaluation.
6. **Evaluate predictions.** Compute real class-wise and overall results; optionally perform spatially blocked cross-validation.
7. **Explore results.** View the predicted class, observation quality, spectral indices, and risk screening layers.
8. **Inspect and export.** Select a field, review its dates/history, save inspection notes, or export a GeoJSON/CSV/GeoTIFF record as supported by the running environment.

---

## 🛰️ Data sources

| Dataset | Earth Engine dataset ID | Role | Important caveat |
|---|---|---|---|
| **Sentinel-2 SR Harmonized** | `COPERNICUS/S2_SR_HARMONIZED` | Multispectral reflectance, indices, time-series features | Cloud masking, valid-observation coverage, and acquisition dates matter; 10/20 m bands differ. |
| **Sentinel-1 GRD** | `COPERNICUS/S1_GRD` | VV/VH radar signals and possible surface-water screening | Radar-dark surfaces are not necessarily flooded fields. |
| **CHIRPS Daily** | `UCSB-CHG/CHIRPS/DAILY` | Rainfall accumulation and seasonal anomalies | Coarse regional product, approximately 0.05° grid; not 10 m field rainfall. |
| **ERA5-Land Hourly** | `ECMWF/ERA5_LAND/HOURLY` | Temperature and weather context | Coarse reanalysis, not a farm weather-station measurement. |
| **SRTM Elevation** | `USGS/SRTMGL1_003` | Topography and slope | Surface elevation, **not** crop height. |
| **FAO GAUL 2015** | `FAO/GAUL/2015/level2` | Administrative district boundary | Administrative boundaries are dataset- and vintage-dependent. |
| **ESA WorldCover** | `ESA/WorldCover/v200` | Reference land-cover/cropland context | Static 2021 land-cover reference, **not** eight-class crop ground truth. |

**Reference documents:** [Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets) · [Earth Engine Guides](https://developers.google.com/earth-engine/guides) · [Sentinel-2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) · [Sentinel-1](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) · [CHIRPS](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY).

**Data provenance principle:** Satellite-derived observations are not *the same thing* as field-measured soil moisture, nutrient concentration, yield, or crop disease. Explanations should use the correct observational terminology.

---

## 🧠 Machine-learning methodology

### Why Random Forest?

Random Forest is a reasonable baseline for multisource tabular features when annotated data is limited. It can model nonlinear relationships without requiring a large labelled image dataset, provides reproducible parameters, and supports global feature-importance analysis. The choice **does not prove** it will outperform every alternative in Varanasi.

### Model type and configured settings

| Parameter | Latest v26 launch configuration |
|---|---|
| Learning task | Supervised **multiclass classification** |
| Algorithm | `ee.Classifier.smileRandomForest()` |
| Trees | **80** |
| Random seed | **42** |
| Approximate train / holdout source split | **70% / 30%**, class-stratified |
| Minimum leaf population | **2** |
| Bag fraction | **0.7** |
| Feature families | Optical bands/indices, SAR signals, temporal summaries, missingness and terrain |
| Spatial research validation | Optional **5-fold, 5-km spatial blocks**, with spatial exclusion settings in the evidence module |

**Note:** General module defaults may differ from these explicit launcher overrides. Use the values from the exact script actually run in a competition or paper.

### Satellite feature engineering

Optical vegetation-index examples:

\[
\mathrm{NDVI}=\frac{NIR-Red}{NIR+Red},\qquad
\mathrm{NDMI}=\frac{NIR-SWIR}{NIR+SWIR}
\]

The feature stack can also include NDRE, NDWI, EVI, SAVI, MNDWI, BSI; Sentinel-1 VV/VH and ratios/differences; observation-count information; temporal vegetation indicators; and topographic context.

**Why temporal features?** Rice, wheat, vegetables, and other vegetation can sometimes exhibit similar reflectance at one moment. Their growth patterns across different windows can be more informative, provided the season and data coverage are appropriate.

### Training data and split

- **260** manually labelled original points, eight classes, with screening for near-duplicate cross-class conflicts.
- Points must fall within the selected district and intersect valid feature-stack pixels.
- A reproducible per-class point-level split is used for the standard **holdout**.
- A geographically separated **blocked validation** workflow is available for stronger evidence against spatial leakage.
- The project does **not** include verified, independent field-survey ground truth sufficient to claim real-world deployment accuracy.

### What the classifier actually predicts

It estimates **one of eight classes** from the trained feature patterns. An RF tree-vote value is **not automatically a calibrated probability**. A 70% vote share should not be described as a validated 70% real-world chance.

### ML vs deterministic analysis

| System component | Type | Correct interpretation |
|---|---|---|
| Crop classification | **Trained ML**: Random Forest | Predicted class requiring evaluation. |
| RF tree-vote/low-vote map | **Model-derived uncertainty proxy** | Voting agreement, not calibrated probability. |
| NDVI / NDMI / NDWI | **Computed remote-sensing indices** | Observational proxies, not independent ML predictions. |
| Vegetation / moisture health score | **Heuristic calculation** | Relative screening indicator, not clinical plant diagnosis. |
| Flood/waterlogging screen | **Threshold/radar/optical analysis** | Potential water signal, not independently verified flooding. |
| Rainfall anomaly / drought screen | **Weather statistics + heuristic** | Candidate stress condition, not certified drought detection. |
| Farmer guidance | **Rule-based content** | Informational decision support, not an autonomous agronomist. |
| Yield forecasting / crop-stage training | **Optional code paths** | Require appropriate labelled data; disabled/not validated in default launch. |

---

## 🧩 Platform features and implementation status

The original analytics code is organized around **15 module families**: preprocessing, classification, crop health, decision support, dashboard UI, reporting, phenology, field analytics, model quality, weather anomaly, water balance, crop rotation, yield, alerts, and farmer advisory. Not every module is a validated prediction service; several are analyses, screens, or optional research interfaces.

| Area | Feature | Current state |
|---|---|---|
| **Introduction** | Photo-led, multilingual landing website; project story, gallery and launch link | **Implemented in v27 HTML** |
| **Geography** | Official Varanasi GAUL district AOI, crop/reference layers, legends | **Implemented in v26 GEE** |
| **Remote sensing** | Sentinel imagery, multiple indices, SAR/weather layers | **Implemented in v26 GEE**; Earth Engine processing on demand |
| **ML** | RF training, classification, vote-based uncertainty screening | **Implemented**; run and verify results |
| **Evaluation** | Holdout metrics, confusion matrix, Kappa/F1, spatial evidence module | **Implemented workflows**; no published verified scores included |
| **Field tools** | Draw/select field, observation quality, monthly history, period comparison | **Implemented in GEE UI** |
| **Risk screening** | Possible surface water, 5-year-reference rainfall anomaly, vegetation decline | **Implemented on demand**; screening only |
| **Human verification** | Pending inspection and uncertain-sample review/export | **Implemented as session/export workflows** |
| **Field Evidence Passport** | Field evidence collection with GeoJSON export | **Implemented**; not automatic PDF and not persistent by itself |
| **Website accessibility** | English, हिंदी, Hinglish; keyboard-accessible feature tabs; reduced-motion design | **Implemented in v27 HTML** |
| **Startup performance** | Defer optional preflight, summaries and advanced map controls | **Implemented in v26 GEE**; no measured latency SLA |
| **Persistent records** | FastAPI + PostGIS fields, observations, review and model records | **Separate backend foundation only**; requires deployment/integration |
| **Automated notifications / production accounts** | Scheduled real alerts, end-user hosted identity service | **Not end-to-end deployed** |

### Public interface overview

- **Landing website:** static HTML/CSS/JavaScript with compressed local photos and language switching.
- **Earth Engine App:** separate hosted interactive GIS and scientific workspace.
- **Launch Platform:** the website links to the **published GEE App**, not to an unpublished Earth Engine Code Editor session.

> The website's charts or decorative graphics are **illustrative UI**, unless explicitly labelled with real computed Earth Engine data.

---

## 🏗️ Technical architecture and stack
mermaid flowchart
  U[Public visitor] --> W[Static v27 website]
  W -->|Launch Analytics| A[Published Earth Engine App]
  A --> G[Google Earth Engine JS API]
  G --> O[Sentinel optical and SAR imagery]
  G --> C[Weather topography and boundary datasets]
  G --> RF[Random Forest classification]
  G --> V[Validation indices risk screening]
  RF --> UI[Maps charts and field evidence]
  V --> UI
  UI --> X[Manual GeoJSON or CSV export]
  X -. import after configuration .-> B[Separate FastAPI backend]
  B --> DB[(PostgreSQL + PostGIS)]
  B --> ID[External OIDC identity provider]
  B --> J[Optional scheduled EE inventory worker]


### Technology stack

| Layer | Technology | Why it is used |
|---|---|---|
| Static introduction frontend | HTML5, CSS3, vanilla JavaScript | Portable, inexpensive, multilingual, responsive website without a required Node.js server. |
| Landing-page assets | Local WebP photos, SVG/favicon | Visual storytelling, cacheable assets. |
| Geospatial application | **Google Earth Engine JavaScript API + `ui`** | Server-side satellite computation plus built-in maps/panels/charts. |
| Remote sensing | Sentinel-1, Sentinel-2, CHIRPS, ERA5-Land, SRTM, WorldCover | Optical/radar/weather/terrain context. |
| Classification | Earth Engine **Random Forest** | Supervised multiclass learning from labelled spectral/temporal samples. |
| Vector operations | Earth Engine geometries/feature collections; GeoJSON | District and field geometry, sample locations, exports. |
| Optional backend API | **Python + FastAPI** | Authenticated record operations, model-run metadata, processing-job APIs. |
| Optional persistent DB | **PostgreSQL + PostGIS** | Spatial fields, observations, audits, review records. |
| Optional authentication | OIDC/JWKS RS256 JWT validation | Authorized access in the separate backend foundation. |
| Website hosting | GitHub Pages, Netlify, Vercel | Host the static introduction site. |
| Analytics hosting | Google Earth Engine Apps | Share an Earth Engine UI app via public link, subject to permissions/quotas. |
| Source/version control | Git + GitHub | Collaboration, traceability and reproducibility. |

**Why two frontends?** Earth Engine Apps use their own JavaScript UI; they do **not** host arbitrary HTML/CSS or the animated photo-based landing page. Keeping the static intro separate allows richer design, while the GEE App runs geospatial analysis.

## 💡 Novelty and innovation 

1. **District-shaped geospatial intelligence:** analysis constrained to the administrative district polygon rather than an arbitrary box.
2. **Multi-source evidence:** optical imagery, radar observations, rainfall/weather context, terrain, and land-cover reference.
3. **Crop identification + monitoring:** classification combined with vegetation/moisture trends and seasonal changes, rather than a static class map alone.
4. **Trust-aware outputs:** source freshness, validation views, uncertain predictions, and explicit warnings about uncalibrated votes.
5. **Investigation-first risks:** flood/drought-type outputs are responsibly named *screens* pending field verification.
6. **Field Evidence Passport:** a connected record of observation context, comparisons, and exportable findings.
7. **Human-in-the-loop workflow:** uncertainty/inspection queues invite better labels and future retraining.
8. **Accessible storytelling:** photo-led Hindi/Hinglish/English introduction reduces the entry barrier for first-time users.
9. **Separation of research and operations:** scientific Earth Engine script plus a distinct API foundation for future persistent systems.

### Comparison with narrower workflows

| Capability | Raw satellite viewer | Simple vegetation dashboard | Fasal-Mitra research demonstrator |
|---|---:|---:|---:|
| Multisource imagery | Sometimes | Sometimes | Yes |
| Eight-class supervised RF workflow | No | Usually no | Yes |
| Field vegetation trends | Requires analysis | Common | Yes |
| Validation and uncertainty tools | Usually external | Varies | Included |
| Risk-screening layers | Requires analysis | Varies | Included |
| Field review/evidence export | Varies | Varies | Included, with limitations |
| Production-verified agronomic decisions | No | Not inherently | **No; additional validation needed** |

This is a **conceptual feature comparison**, not a measured superiority benchmark against named competitors.

---

## ✅ Feasibility and constraints

### Technical feasibility

- Earth Engine provides scalable geospatial operations without requiring a local satellite-data warehouse for the prototype.
- Random Forest is feasible on the current labelled-point dataset as an exploratory classifier.
- On-demand computations and a fast-start dashboard avoid *some* unnecessary startup requests.
- A static introductory website can be deployed independently at minimal infrastructure complexity.
- The backend starter demonstrates a plausible persistence route without pretending it is already integrated.

### Practical feasibility

| Dimension | Current assessment | Remaining requirement |
|---|---|---|
| **Prototype development** | Feasible with a Google account, Earth Engine project, and browser | Test all functions in a real EE environment. |
| **Compute performance** | Acceptable for demos with on-demand data, subject to limits | Benchmark cold starts, layer tiles and model runs; precompute popular layers where justified. |
| **Data availability** | Public Earth observation products are accessible in Earth Engine | Resolve cloud gaps, label seasonality, dataset update latency, and scale mismatch. |
| **Scientific reliability** | Validation code exists | Independently verify crop labels, run spatial/seasonal holdouts and publish real metrics. |
| **Economics** | Static website is low-cost | Review EE access terms/quotas, backend hosting, storage, map/export workloads and operational support. |
| **Adoption** | Multilingual explanation and guided UI lower entry friction | User testing with farmers/extension staff and offline/low-bandwidth consideration. |
| **Operational security** | Backend starter has OIDC and role-based checks | Deploy real IdP, HTTPS, rate limits, migrations, backup, monitoring and privacy controls. |

### Known risks and mitigations

- **Cloudy imagery:** optical masking, Sentinel-1 complementary radar, and visible quality/coverage indicators.
- **Spatial label leakage:** blocked cross-validation and independent field verification.
- **Season mismatch:** dated features and season-specific experiments.
- **Sparse labels:** active learning and more verified field surveys before geographic expansion.
- **Public app slowness:** lazy operations, precomputed validated assets, less expensive client-side UI construction.
- **Over-precise weather maps:** present rainfall/weather at their actual coarse resolution and avoid 10 m claims.
- **Broken public export buttons:** adapt Code Editor Tasks workflows for an app-compatible export/download interface.

---

## 📈 Scalability and production roadmap

**Scaling a district prototype to multiple districts is not just changing a dropdown.** New geography can change crops, crop calendars, training distributions, cloud patterns, data availability, and model performance.

### Scale by dimension

| Dimension | Prototype | Required for broader deployment |
|---|---|---|
| **Coverage** | One district polygon: Varanasi | Multi-district polygons, additional verified labels, per-region validation. |
| **Users** | GEE App audience subject to quotas | Session/auth design, capacity testing, cached results, controlled concurrency. |
| **Raster processing** | On-demand EE computation | Scheduled batch jobs, precomputed assets, cache policy, task monitoring. |
| **Data storage** | UI session and manual exports | Deployed PostGIS, backups, retention policies, object storage. |
| **ML lifecycle** | Manual training and results inspection | Model registry, dataset versioning, monitoring, drift checks, controlled retraining. |
| **Trust** | Manually annotated training points | Independent geotagged ground-truth collection, calibration, seasonal/spatial evaluation. |
| **Reliability** | Research demo | Logging, error budgets, alerts, retries, service-level monitoring. |
| **Security** | Browser UI plus separate API example | Real user tenancy, least privilege, protected secrets, HTTPS, access review. |
| **Farmer experience** | Multilingual intro and decision-support pages | User studies, low-bandwidth/mobile support, regional language refinement, accessibility. |

### Proposed phases

**Phase 1: Competition-ready demonstrator**
- Stabilize public app, simplify demo journey, verify map layers and GEE permissions.
- Publish a reproducible model evaluation with confusion matrix and spatial folds.
- Add a clear limitations and data-provenance statement.

**Phase 2: Validated district pilot**
- Collect verified crop/field labels, register acquisition dates and ground-truth sources.
- Evaluate by season and geography; tune threshold-based screens against actual events.
- Conduct usability testing with field teams; create an app-compatible report workflow.

**Phase 3: Managed backend integration**
- Deploy FastAPI/PostGIS, migrate with proper schema tooling, configure an identity provider.
- Automate limited, monitored Earth Engine inventory/processing jobs.
- Connect verified field observations to the frontend through authenticated APIs.

**Phase 4: Multi-district operational service**
- District-specific validation, retraining thresholds, precomputed map tiles/assets and permissions.
- Capacity/load tests, cost monitoring, audit retention, incident response and compliance review.

### Scalability success measures (to be measured)

Public app cold-start time, 95th-percentile map response time, model evaluation latency, active users under quota, cost per analyzed area, new-label review time, spatial-CV accuracy, and task failure rate. **No numeric SLA or maximum concurrent-user capacity has been established.**

---

## 📊 Evaluation and scientific validity

**Do not invent performance percentages.** This repository intentionally does not claim a measured accuracy until outputs have been generated and checked in a known run.

### Required evaluation deliverables

| Metric / evidence | What it answers | Current status |
|---|---|---|
| Holdout Accuracy | Fraction of correct held-out predictions | **Run required** |
| Confusion matrix | Which crop classes get confused | **Run required** |
| Precision / Recall / F1 per class | Performance for each of eight classes | **Run required** |
| Macro F1 | Balanced summary across classes | **Run required** |
| Cohen's Kappa | Agreement adjusted for chance | **Run required** |
| Spatial 5-fold blocked scores | Generalization to separated locations | **Run required** |
| Training/testing class counts | Whether the split is representative | **Run required** |
| Majority-class baseline | Whether ML improves on a trivial classifier | **Run required** |
| Satellite coverage and source dates | Whether imagery actually supports the evaluated fields | **Run required** |
| Independently verified field labels | Whether sample labels reflect the real world | **Not supplied** |
| Risk-screen sensitivity/precision | Whether drought/flood flags match documented field events | **Not established** |

### Evaluation guidance

1. Run the exact committed Earth Engine script for a specified date window.
2. Record the **usable sample count**, class distribution, split protocol, feature set and random seed.
3. Train Random Forest and export or record the actual holdout confusion matrix, class-wise scores, and model metadata.
4. Run the separate 5-km blocked evaluation, document the exclusion-buffer setting and per-fold sample counts.
5. Compare with a majority-class baseline; if possible compare another baseline classifier.
6. Report uncertainty and likely spatial autocorrelation limitations. A standard point holdout is **not** the same as independent spatial validation.
7. For future studies, repeat across seasons and verify labels through georeferenced field observations.

Saved CSV placeholders in `results/` are explicitly marked 


## 🛠️ Installation and running the project

 B. Run the Earth Engine application

1. Register and sign in to [Google Earth Engine Code Editor](https://code.earthengine.google.com/).
2. Choose a Google Cloud project registered/authorized for Earth Engine, with required access and API enabled.
3. Create a new Code Editor script, copy **all contents** of `earth-engine/Fasal_Mitra_v26_FAST_GEE_DASHBOARD.js`, and save.
4. Click **Run**. The project introduction should load first.
5. Select **Launch Main Dashboard**, then request a satellite layer or classification as needed.
6. Train the classifier only after confirming there are sufficient valid samples in the chosen period.
7. Use the validation/field/risk workspaces to test model results and supporting evidence.
8. For Code Editor exports, create and start an export in **Tasks**. A public Earth Engine App does **not** expose the Code Editor Tasks pane as an end-user download interface.

**Troubleshooting:** An OAuth/authentication failure needs valid account/project access. Heavy composite or classification layers can remain slow even when the dashboard UI loads quickly. Do not edit training point locations to solve access errors.

### C. Optional API foundation (advanced)

The `backend/` directory contains an **independent, not deployed** FastAPI/PostGIS proof-of-integration foundation. Follow [`backend/README.md`](backend/README.md) for database/identity setup and test instructions. It does **not** automatically persist Earth Engine Code Editor UI state.

---

## 🌐 Deployment and publishing

### Website: GitHub Pages

### Analytics: Earth Engine Apps

The published app URL supplied for this project is:
https://pheno-crop-analytics.projects.earthengine.app/view/fasal-mitra



### API: separate deployment


---

## 🎤 Three-minute competition demo


> **“फसल-मित्र: खेत की कहानी अब सिर्फ आँखों से नहीं, satellite data aur machine learning se bhi samajhiye.”**


## 🛡️ Limitations, ethics, and security

### Scientific limitations

- **Label quality:** Original 260 points are manually tagged; no independent verified field survey accompanies them.
- **Training bias:** Sparse class examples and neighboring samples can inflate standard validation results.
- **Transferability:** A model trained around Varanasi should not be assumed accurate in other districts or seasons.
- **Data resolution:** CHIRPS/ERA5-Land have much coarser grids than Sentinel imagery; displayed resampling does not create local weather measurements.
- **Image availability:** Clouds, missing scenes, differing optical/radar acquisition dates, and temporal composites introduce uncertainty.
- **Risk uncertainty:** Satellite water or vegetation signals alone do not establish actual flood damage, crop disease, drought severity, or monetary loss.
- **Yield/phenological stages:** Optional training paths require real field labels and remain disabled/unvalidated in the default release.
- **Confidence:** Random Forest voting agreement is not a calibrated probability without further testing.

### Engineering limitations

- Some Earth Engine **Code Editor Tasks** export actions require redesign for public-app visitors.
- The backend API is a separate reference implementation, not automatically integrated with the hosted GEE App.
- No production throughput/latency measurements, availability SLA, load-test results, or external user audit have been provided.
- Earth Engine quotas, permissions, commercial/noncommercial eligibility and asset visibility must be checked before broad deployment.

### Privacy, licensing, and open-source hygiene
---

## 🚀 Future work

- Independent georeferenced field verification and an evidence-based active-learning feedback loop.
- Multi-season, multi-district model comparison with fair held-out benchmarks.
- Calibrated uncertainty and responsible per-prediction explanations.
- Validated flood/drought event datasets for risk-screen evaluation.
- Persistent authenticated field profiles and proper backend/frontend integration.
- App-compatible reporting and shareable, timestamped PDFs (rather than Code Editor-only export tasks).
- Precomputed district-layer assets to reduce repeated heavy satellite computation.
- Usability studies with farmers, regional-language accessibility, and mobile connectivity testing.
- Measured infrastructure performance, cost and quota management.

---

## 📚 References and project information

**Official sources and tooling**

- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [Earth Engine Apps](https://developers.google.com/earth-engine/guides/apps)
- [Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets)
- [Google Earth Engine Random Forest Classification](https://developers.google.com/earth-engine/guides/classification)
- [FAO GAUL 2015 Level 2](https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level2)
- [Sentinel-1 GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
- [Sentinel-2 SR Harmonized](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [CHIRPS Daily](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY)
- [FastAPI](https://fastapi.tiangolo.com/) · [PostGIS](https://postgis.net/)

### Project record

| Item | Details |
|---|---|
| **Project name** | फसल-मित्र (Fasal-Mitra) |
| **Tagline** | हर खेत की कहानी, डेटा की ज़ुबानी |
| **Developer** | Pushkkal Gupta |
| **Focus** | Geospatial machine learning and agricultural monitoring |
| **Study region** | Varanasi district, Uttar Pradesh, India |
| **Live Earth Engine App** | [Launch Fasal-Mitra](https://pheno-crop-analytics.projects.earthengine.app/view/fasal-mitra) |


### 🌱 फसल-मित्र: हर खेत की कहानी, डेटा की ज़ुबानी।

**Built to make satellite evidence more accessible, actionable, and scientifically responsible.**

</div>
