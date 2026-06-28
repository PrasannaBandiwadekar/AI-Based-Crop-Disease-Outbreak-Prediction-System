"""
build_knowledge_graph.py
------------------------
Batch pipeline for Crop Disease Outbreak Prediction System (v5).

This script writes all 7 structured JSON knowledge files to:
  research_comp/knowledge_graph/

Each entry was manually curated from the research papers in:
  research_comp/evidence_base/literature/extracted_text/

Run:
  python scratch/build_knowledge_graph.py
"""

import os
import json

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE, "research_comp", "knowledge_graph")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Knowledge Records ─────────────────────────────────────────────────────────
RECORDS = [

    # 1. Minnatullah et al. 2025 — comprehensive red rot review (AATCC)
    {
        "file_name": "13_2AATCC_01.txt",
        "data": {
            "paper_type": "management",
            "disease": "red_rot",
            "mechanism": {
                "infection_process": (
                    "Colletotrichum falcatum infects sugarcane stalks primarily "
                    "through infected seed setts and soil-borne inoculum, causing "
                    "reddish discoloration of internal tissues and invertase-driven "
                    "sucrose breakdown."
                ),
                "spread_modes": [
                    "Infected seed setts",
                    "Soil-borne inoculum",
                    "Nodal and internodal infection",
                ],
                "progression_pattern": (
                    "Sett/stalk infection → reddish pith with white spots → "
                    "degradation of cane juice quality (Brix, Pol, sucrose) → "
                    "reduced germination and settling mortality → complete crop drying."
                ),
            },
            "environmental_drivers": {
                "temperature": (
                    "Optimum pathogen growth/sporulation at 30 °C. "
                    "Peak field incidence at T2M_MAX ~33.1 °C, T2M_MIN ~26.4 °C."
                ),
                "humidity": (
                    "Morning RH 80–90 %, evening RH 50–76 % coincide with peak incidence. "
                    "Ideal growth RH 70–85 %."
                ),
                "rainfall": (
                    "Peak incidence (25.5 %) coincides with 115.8 mm rainfall in the "
                    "second fortnight of August. Disease most prevalent July–September."
                ),
                "variability": (
                    "Disease incidence rises progressively: 0 % (May–Jun) → "
                    "2–10 % (Jul–Aug) → 5–15 % (Sep–Oct) → 10–20 % (Nov–Dec)."
                ),
            },
            "temporal_behavior": {
                "latent_phase": True,
                "requires_accumulation": True,
                "delay_description": (
                    "Disease builds over the monsoon season. "
                    "Accumulation of high humidity + rainfall over multiple fortnights "
                    "drives the incidence peak in the second fortnight of August."
                ),
            },
            "feature_mapping": {
                "weather_features": [
                    "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR"
                ],
                "temporal_features": [
                    "lag_features (15-day delay in disease response)",
                    "rolling_windows (14-day, 28-day rainfall/humidity accumulation)",
                ],
                "agronomic_features": [
                    "variety_susceptibility",
                    "intercropping_system",
                    "micronutrient_levels",
                    "soil_pH",
                ],
            },
            "outbreak_logic": {
                "requires_sustained_conditions": True,
                "supports_early_warning": True,
                "threshold_type": "variable",
            },
            "advisory_actions": [
                "MHAT seed cane at 54 °C for 4 hours (RH 95–100 %).",
                "Sett treatment with Carbendazim 50 WP or Thiophanate methyl 0.25 %.",
                "Intercropping with Garlic or Coriander reduces incidence by ~20 %.",
                "Apply ZnSO4 (30 kg/ha) + Boron as foliar application.",
                "Biocontrol: Trichoderma harzianum or Pseudomonas spp. soil application.",
                "Plant resistant varieties; avoid monocropping susceptible varieties.",
            ],
            "confidence_level": "high",
        },
    },

    # 2. Saharan 1992 — PhD thesis, SMRA epidemiology models
    {
        "file_name": "230495.txt",
        "data": {
            "paper_type": "epidemiology",
            "disease": "red_rot",
            "mechanism": {
                "infection_process": (
                    "C. falcatum infects standing canes during the rainy season "
                    "via nodal inoculation. Soil-borne inoculum survives 60–90 days "
                    "and spreads via irrigation/rain water."
                ),
                "spread_modes": [
                    "Soil-borne (infected debris survives 60–90 days)",
                    "Water dissemination of conidia",
                    "Nodal infection during monsoon",
                ],
                "progression_pattern": (
                    "Nodal transgression → lesion spreading across cane width → "
                    "white spot formation → yellowing and drying of tops."
                ),
            },
            "environmental_drivers": {
                "temperature": (
                    "Optimum red rot development at mean T2M 29.4–31.0 °C. "
                    "T2M_MIN is the single most important predictor (negatively correlated "
                    "with disease incidence in the succeeding 15-day fortnight; R² 0.82–0.87)."
                ),
                "humidity": (
                    "Good disease development at RH 79–92 %. "
                    "Evening RH is a significant secondary predictor."
                ),
                "rainfall": (
                    "Water plays a key role in disseminating conidia and "
                    "predisposing the host to penetration."
                ),
                "variability": (
                    "SMRA models show T2M_MIN variation in one fortnight strongly "
                    "predicts disease incidence in the next fortnight (15-day lag)."
                ),
            },
            "temporal_behavior": {
                "latent_phase": True,
                "requires_accumulation": True,
                "delay_description": (
                    "T2M_MIN in one 15-day period negatively predicts disease in the "
                    "next 15-day period. Soil inoculum persists 60–90 days. "
                    "Top-drying is a cumulative delayed symptom."
                ),
            },
            "feature_mapping": {
                "weather_features": [
                    "T2M", "T2M_MIN", "T2M_MAX", "RH2M", "PRECTOTCORR"
                ],
                "temporal_features": [
                    "lag_features (15-day lag — critical for SMRA model replication)",
                    "rolling_windows (14-day mean of T2M_MIN, RH2M)",
                ],
                "agronomic_features": [
                    "variety_susceptibility",
                    "nitrogen_fertilizer_level",
                    "soil_amendments",
                ],
            },
            "outbreak_logic": {
                "requires_sustained_conditions": True,
                "supports_early_warning": True,
                "threshold_type": "variable",
            },
            "advisory_actions": [
                "Treat setts with Bavistin (0.25 % dip, 1–2 hours).",
                "Foliar spray Bavistin 0.1 % twice (Aug 15 and Nov 4).",
                "Incorporate soil amendments (zinc sulphate, ferrous sulphate).",
                "Remove and destroy infected crop debris 90+ days before replanting.",
            ],
            "confidence_level": "high",
        },
    },

    # 3. Viswanathan 2022 — multi-disease diagnostics & management review (RASSA)
    {
        "file_name": "RAASAfinal2022.txt",
        "data": {
            "paper_type": "management",
            "disease": "multiple",
            "mechanism": {
                "infection_process": (
                    "Primary infection via systemically infected setts or soil-borne "
                    "inoculum. Pathogen load accumulates in stalks over successive "
                    "crop cycles causing varietal degeneration."
                ),
                "spread_modes": [
                    "Infected planting materials (setts) — primary vector for red rot, smut, wilt, SCGS, YLD, RSD",
                    "Aerial dispersal — smut spores, Pokkah boeng conidia, foliar disease spores",
                    "Soil-borne — wilt (Fusarium sacchari), sett rot (Ceratocystis paradoxa)",
                    "Ratooning of infected clumps accumulates pathogen load",
                ],
                "progression_pattern": (
                    "Systemic colonization → slow varietal degeneration over ratoon cycles → "
                    "reduced cane yield and sugar recovery. "
                    "Viral symptoms (YLD) take 5–6 months to appear."
                ),
            },
            "environmental_drivers": {
                "temperature": (
                    "Hot dry weather favors smut. "
                    "Cool temperatures + high moisture favor foliar diseases (rust, eye spot)."
                ),
                "humidity": (
                    "High humidity during rainy months favors Pokkah boeng and foliar diseases. "
                    "Excess irrigation and un-stripped dry leaves build up microclimatic RH."
                ),
                "rainfall": (
                    "High rainfall reduces smut severity but drives Pokkah boeng and "
                    "foliar diseases. Waterlogging predisposes plants to wilt."
                ),
                "variability": (
                    "Drought followed by waterlogging predisposes to wilt. "
                    "Plant stress increases smut whip development."
                ),
            },
            "temporal_behavior": {
                "latent_phase": True,
                "requires_accumulation": True,
                "delay_description": (
                    "Pathogen load accumulates over successive ratoon crops. "
                    "YLD symptoms appear 5–6 months post-infection. "
                    "Pokkah boeng symptoms develop during grand growth (3–7 months crop age)."
                ),
            },
            "feature_mapping": {
                "weather_features": [
                    "T2M", "RH2M", "PRECTOTCORR"
                ],
                "temporal_features": [
                    "lag_features",
                    "rolling_windows",
                ],
                "agronomic_features": [
                    "variety_susceptibility",
                    "infected_sett",
                    "crop_age_ratoon",
                    "field_sanitation",
                ],
            },
            "outbreak_logic": {
                "requires_sustained_conditions": True,
                "supports_early_warning": True,
                "threshold_type": "variable",
            },
            "advisory_actions": [
                "Use disease-free seed canes from certified nurseries.",
                "MHAT or AST to eliminate sett-borne infections.",
                "Fungicide sett treatment + biocontrol (Pseudomonas) via mechanized vacuum infiltration.",
                "Plant resistant varieties; avoid >50 % monocropping of single variety.",
                "Virus-free nurseries via meristem-tip culture + molecular indexing (PCR/LAMP).",
                "Foliar fungicides (mancozeb 0.2 %, propiconazole) for rust, smut, eye spot.",
            ],
            "confidence_level": "high",
        },
    },

    # 4. Duttamajumder 2008 — Red rot of Sugarcane (IISR book)
    {
        "file_name": "Red_rot_of_Sugarcane.txt",
        "data": {
            "paper_type": "epidemiology",
            "disease": "red_rot",
            "mechanism": {
                "infection_process": (
                    "Primary infection via infected setts (seed-borne). "
                    "Pathogen can traverse from sett to stalk and crown without "
                    "conspicuous early symptoms. July inoculations produce more "
                    "potent secondary spread than August/September ones."
                ),
                "spread_modes": [
                    "Infected setts (primary inoculum)",
                    "Spindle infection (conidia spreading from stalk to crown)",
                    "Vascular bundle transit",
                    "Soil survival (3–4 months in natural soil)",
                ],
                "progression_pattern": (
                    "Death of young shoots → spindle infection → "
                    "yellowing/withering of crown leaves from margin → "
                    "internal reddening with white patches and alcoholic odor → "
                    "nodal rotting and plant death."
                ),
            },
            "environmental_drivers": {
                "temperature": (
                    "Optimum pathogen germination at 25 °C. "
                    "Disease flares up during grand growth phase (T2M 24–30 °C). "
                    "Lower temperatures push pathogen towards stool base."
                ),
                "humidity": (
                    "Monsoon humidity drives grand growth phase and pathogen flare-up."
                ),
                "rainfall": (
                    "Monsoon rain provides moisture for grand growth and "
                    "secondary spread of inoculum."
                ),
                "variability": (
                    "July timing of infection is critical — produces more severe "
                    "secondary spread than later months."
                ),
            },
            "temporal_behavior": {
                "latent_phase": True,
                "requires_accumulation": True,
                "delay_description": (
                    "Pathogen traverses sett → stalk without visible symptoms. "
                    "Mid-rib isolates remain reservoir before acquiring stalk infection capability. "
                    "Symptoms manifest mainly during monsoon (July–Sept) but depend on "
                    "14–21 day weather accumulation."
                ),
            },
            "feature_mapping": {
                "weather_features": [
                    "T2M", "T2M_MIN", "T2M_MAX", "RH2M", "PRECTOTCORR"
                ],
                "temporal_features": [
                    "lag_features (14–21 day pre-symptom accumulation window)",
                    "rolling_windows (14-day, 28-day)",
                ],
                "agronomic_features": [
                    "variety_susceptibility",
                    "infected_sett",
                    "crop_age",
                    "inoculation_timing",
                ],
            },
            "outbreak_logic": {
                "requires_sustained_conditions": True,
                "supports_early_warning": True,
                "threshold_type": "variable",
            },
            "advisory_actions": [
                "Three-tier seed nursery programme using disease-free setts.",
                "Physical heat treatment of setts.",
                "Field sanitation: remove and destroy infected clumps.",
                "Select planting material from disease-free fields only.",
            ],
            "confidence_level": "high",
        },
    },

    # 5. Ghini et al. 2011 — Climate change & tropical crop diseases (Plant Pathology)
    {
        "file_name": "ghini_2011_extracted.txt",
        "data": {
            "paper_type": "climate",
            "disease": "multiple",
            "mechanism": {
                "infection_process": (
                    "Climate change alters host–pathogen interactions by modifying "
                    "temperature, CO2, and humidity regimes. Elevated CO2 increases "
                    "canopy density, creating higher internal humidity favorable to pathogens."
                ),
                "spread_modes": [
                    "Climate-driven range expansion of pathogens into new regions",
                    "Elevated CO2 accelerating host canopy growth and pathogen substrate",
                    "Temperature-driven changes in infection window duration",
                ],
                "progression_pattern": (
                    "Warmer temperatures shorten latent periods and accelerate infection cycles. "
                    "Elevated CO2 (700–900 ppm) can reduce red rot latent period from "
                    "~36 days to ~20 days."
                ),
            },
            "environmental_drivers": {
                "temperature": (
                    "Temperature increases shift disease geographic ranges and "
                    "accelerate pathogen life cycles. Non-linear response — "
                    "moderate warming can benefit some pathogens while harming others."
                ),
                "humidity": (
                    "Elevated CO2 increases canopy biomass, creating higher internal "
                    "relative humidity within crop stands, favoring disease development."
                ),
                "rainfall": (
                    "Heavy and unseasonal rains (a climate-change projection) are "
                    "major drivers of disease spread and epidemic onset."
                ),
                "variability": (
                    "Frequency of extreme weather events (heavy rain, drought, heatwaves) "
                    "is the key climate-change variable for disease outbreak risk — "
                    "not just mean changes."
                ),
            },
            "temporal_behavior": {
                "latent_phase": True,
                "requires_accumulation": True,
                "delay_description": (
                    "Elevated CO2 shortens red rot latent period from ~36 to ~20 days. "
                    "This compression of the latent phase increases outbreak risk under "
                    "future climate conditions."
                ),
            },
            "feature_mapping": {
                "weather_features": [
                    "T2M", "RH2M", "PRECTOTCORR"
                ],
                "temporal_features": [
                    "lag_features (shortened latent periods under elevated CO2)",
                    "rolling_windows (14-day, 28-day extremes for outbreak risk)",
                ],
                "agronomic_features": [
                    "CO2_concentration",
                    "canopy_density",
                ],
            },
            "outbreak_logic": {
                "requires_sustained_conditions": True,
                "supports_early_warning": True,
                "threshold_type": "variable",
            },
            "advisory_actions": [
                "Monitor disease range expansion using climate projections.",
                "Develop early warning systems sensitive to extreme weather events.",
                "Include climate variability (not just means) in outbreak prediction models.",
                "Use FACE studies to calibrate pathogen response under elevated CO2.",
            ],
            "confidence_level": "medium",
        },
    },

    # 6. Luck et al. 2011 — Climate change & food crop diseases (Plant Pathology)
    {
        "file_name": "luck_2011_extracted.txt",
        "data": {
            "paper_type": "climate",
            "disease": "multiple",
            "mechanism": {
                "infection_process": (
                    "Climate change affects pathogen life cycles non-linearly. "
                    "Increased CO2 favors some pathogens (e.g., Fusarium) while "
                    "limiting others (e.g., Puccinia striiformis). "
                    "Warming winters reduce pathogen kill periods, expanding infection windows."
                ),
                "spread_modes": [
                    "Warmer winters allow pathogen survival and range expansion",
                    "Heavy/unseasonal rains drive inoculum splash dispersal",
                    "Elevated CO2 increases plant biomass as pathogen substrate",
                ],
                "progression_pattern": (
                    "Climate-driven changes shift epidemic onset timing and alter "
                    "the duration of conducive infection windows. "
                    "Non-linear disease responses mean small climate shifts can "
                    "trigger disproportionately large outbreak risk changes."
                ),
            },
            "environmental_drivers": {
                "temperature": (
                    "Non-linear response: +1–3 °C can benefit some pathogens while "
                    "suppressing others. Warmer winters extend growing seasons and "
                    "infection windows. Critical for 14–28 day rolling window models."
                ),
                "humidity": (
                    "Increased CO2-driven canopy density creates higher internal crop humidity, "
                    "favoring disease development even without external RH changes."
                ),
                "rainfall": (
                    "Heavy and unseasonal rains are key climate-change drivers. "
                    "Drought conditions can also stress plants, increasing susceptibility."
                ),
                "variability": (
                    "Extreme events (not just mean changes) are the primary disease drivers. "
                    "Multi-day accumulation of conducive conditions matters more than daily peaks."
                ),
            },
            "temporal_behavior": {
                "latent_phase": True,
                "requires_accumulation": True,
                "delay_description": (
                    "Epidemic onset shifts by 14–28 days under projected climate changes. "
                    "Multi-day accumulation of humidity + temperature within the conducive "
                    "range is critical for outbreak prediction."
                ),
            },
            "feature_mapping": {
                "weather_features": [
                    "T2M", "T2M_MIN", "T2M_MAX", "RH2M", "PRECTOTCORR"
                ],
                "temporal_features": [
                    "lag_features",
                    "rolling_windows (14-day and 28-day windows — validated for epidemic onset shift)",
                ],
                "agronomic_features": [
                    "variety_susceptibility",
                    "crop_calendar",
                ],
            },
            "outbreak_logic": {
                "requires_sustained_conditions": True,
                "supports_early_warning": True,
                "threshold_type": "variable",
            },
            "advisory_actions": [
                "Use 14–28 day rolling windows in outbreak prediction models.",
                "Include climate variability metrics (not just means) as model features.",
                "Develop multi-factor quantitative outbreak models.",
                "Incorporate altered planting schedules as an adaptive strategy.",
            ],
            "confidence_level": "high",
        },
    },

    # 7. Red Rot Integrated Management 2022 (JUST Agriculture journal)
    {
        "file_name": "red_rot_mgmt_2022_extracted.txt",
        "data": {
            "paper_type": "management",
            "disease": "red_rot",
            "mechanism": {
                "infection_process": (
                    "Primary infection via seed-borne and soil-borne inoculum. "
                    "Infected setts introduce the pathogen; secondary spread occurs "
                    "via rain-splash dispersal of conidia and spindle infection."
                ),
                "spread_modes": [
                    "Rain water and rain-splash dispersal of conidia",
                    "Secondary spread through spindle infection and nodal rotting",
                    "Water-borne movement along vascular bundles",
                ],
                "progression_pattern": (
                    "Early leaf drooping and colour loss → rotten setts and dull/shrunken rind → "
                    "reddened pith with white spots and alcoholic odour → "
                    "extensive internal rotting and plant death."
                ),
            },
            "environmental_drivers": {
                "temperature": (
                    "Optimal disease development at 29–31 °C; "
                    "disease flares during grand growth phase at 24–30 °C."
                ),
                "humidity": "85–100 % RH required for severe outbreaks.",
                "rainfall": (
                    "Heavy/unseasonal rains and cloud cover provide moisture for "
                    "rain-splash inoculum dispersal. Rainy days are the key trigger."
                ),
                "variability": (
                    "Variability in rainfall intensity and timing (heavy or unseasonal rain) "
                    "increases outbreak risk."
                ),
            },
            "temporal_behavior": {
                "latent_phase": True,
                "requires_accumulation": True,
                "delay_description": (
                    "Latent period of several days to weeks. "
                    "Spindle infection shows secondary spread ~7 days after inoculation; "
                    "mid-rib infection may take >1 month before visible symptoms."
                ),
            },
            "feature_mapping": {
                "weather_features": [
                    "T2M (29–31 °C optimal)", "T2M_MIN (24 °C)", "T2M_MAX (31 °C)",
                    "RH2M (85–100 %)", "PRECTOTCORR (rainy days)"
                ],
                "temporal_features": [
                    "lag_features (latent period days–weeks)",
                    "rolling_windows (3–14 day sustained favorable conditions)",
                ],
                "agronomic_features": [
                    "variety_susceptibility",
                    "sett_health",
                    "field_sanitation",
                ],
            },
            "outbreak_logic": {
                "requires_sustained_conditions": True,
                "supports_early_warning": True,
                "threshold_type": "variable",
            },
            "advisory_actions": [
                "Use resistant varieties (Co 98015, Co 98016, Co 285, Cos-109, Cos-443, Bo 3, Bo 32).",
                "Seed treatment: MHAT at 54 °C for 2 hours (more effective than hot-water at 50 °C).",
                "Field sanitation: use disease-free setts, well-drained fields, avoid ratooning infected crop.",
                "Quarantine infected plots; apply proper water management.",
            ],
            "confidence_level": "high",
        },
    },
]

# ── Write JSON files ──────────────────────────────────────────────────────────
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Total records to write: {len(RECORDS)}\n")

for rec in RECORDS:
    file_name = rec["file_name"]
    json_name = file_name.replace(".txt", ".json")
    out_path = os.path.join(OUTPUT_DIR, json_name)

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec["data"], f, indent=4, ensure_ascii=False)
        print(f"[OK]  {file_name:45s} -> knowledge_graph/{json_name}")
    except Exception as e:
        print(f"[ERR] ERROR writing {json_name}: {e}")

print("\n[DONE] All knowledge graph files written successfully.")
print(f"       Location: {OUTPUT_DIR}")
