# Congestion Estimation: Flow-Based (old) vs Density + Speed (current)

This document explains how congestion classification changed in commit
`5531ef3`, why the old method systematically misread traffic jams as
"free flow", and why the current method is correct.

## TL;DR

| | Old (flow-based) | Current (density + speed) |
|---|---|---|
| Primary signal | Vehicles **crossing a line** per minute | Vehicles **present in view** (density) |
| Secondary signal | — | Fraction of tracked vehicles that are **stopped** |
| Jam at a red light | **FREE_FLOW** (wrong) | **SEVERE** (correct) |
| Empty road | FREE_FLOW | FREE_FLOW |
| Per-lane breakdown | No | Yes (lane A/B via per-camera split line) |
| Depends on tracking quality | Yes (missed tracks = missed counts) | Only for the stopped refinement; density needs no tracking |
| Where implemented | `calculate_traffic_metrics()` (removed) | `lane_utils.classify_congestion()`, shared by the live detector and the background estimator |

---

## 1. The old method

Each detector counted vehicles whose DeepSORT track crossed a horizontal
counting line at mid-frame, over a 10-second window:

```python
vehicles_per_minute = crossings_in_10s * 6

if vehicles_per_minute < 10:   FREE_FLOW   (LOS A)
elif < 30:                     MODERATE    (LOS C)
elif < 60:                     CONGESTED   (LOS D)
else:                          SEVERE      (LOS F)
```

This is a **flow** (throughput) measurement: how many vehicles pass a point
per unit time.

## 2. Why flow cannot classify congestion

The fundamental relationship of traffic engineering is:

```
flow = density × speed        (q = k · v)
```

Flow is **low at both extremes** of the road's state:

| Road state | Density | Speed | Flow |
|---|---|---|---|
| Empty road | ~0 | high | **~0** |
| Free flow | low | high | moderate |
| Heavy but moving | high | moderate | **maximum** |
| **Gridlock / queue** | **maximum** | **~0** | **~0** |

A single flow number is therefore ambiguous: `vehicles_per_minute ≈ 0` means
either *nobody is on the road* or *everybody is stuck on it*. The old
thresholds mapped "low flow" to FREE_FLOW, so the worst possible traffic state
produced the best possible label.

### The observed failure

At TUGUMUDA during the morning peak (screenshots from 2026-07-06):

- The video showed **~16 vehicles queued bumper-to-bumper** at the light.
- Almost none crossed the counting line (they were stopped), so
  `vehicles_per_minute = 6` → the dashboard reported **FREE_FLOW / LOS A**.

This is not a tuning problem — no threshold on flow alone can fix it, because
the signal itself does not distinguish empty from jammed. This is also why
professional practice (e.g. the Highway Capacity Manual) defines level of
service for road segments from **density**, not flow.

## 3. The current method

Two signals, both computed over the same 10-second window
(`DetectionWorker.calculate_traffic_metrics`):

1. **Density** — the average number of vehicles *present* in the frame across
   processed frames (`density_samples`). Needs no tracking: YOLO detections
   per frame are enough, so it also works for the background estimator that
   samples each camera from single frames every ~30 s.
2. **Stopped fraction** — the share of DeepSORT tracks whose centroid moved
   slower than ~1% of frame height per second over its recent history
   (`_stopped_fraction`). This separates "many vehicles, flowing" from
   "many vehicles, stationary queue".

Classification (`lane_utils.classify_congestion`):

```
density  < 5   →  FREE_FLOW  (A)
density  < 12  →  MODERATE   (C)
density  < 20  →  CONGESTED  (D)
density ≥ 20   →  SEVERE     (F)

if density ≥ 5 and stopped_fraction ≥ 50%:  bump one level worse
```

Which yields the decision matrix:

| | moving | mostly stopped |
|---|---|---|
| **low density (<5)** | FREE_FLOW | FREE_FLOW (a few cars at a light is not congestion) |
| **medium (5–12)** | MODERATE | CONGESTED |
| **high (12–20)** | CONGESTED | **SEVERE** |
| **very high (≥20)** | SEVERE | SEVERE |

`vehicles_per_minute` is still measured and displayed — but relabelled
**Throughput**, because that is what it is. It no longer drives the
congestion label.

### Validation on the same intersection

Same camera, same morning peak, current model (captured live at 06:38):

- Vehicles Present: **16**, Stopped: **58%**, Throughput: 6.0/min
- Result: density 16 → CONGESTED, +1 for ≥50% stopped → **SEVERE / LOS F** ✓

The identical conditions that the old model called FREE_FLOW.

## 4. Per-lane extension

Because density is a *presence* measure, it splits naturally by image region.
Each detection is assigned to lane A or B by which side of a per-camera
**split line** its ground point (bottom-centre of the box) falls on
(`lane_utils.lane_of`). Both the live detector and the background estimator
publish `lanes: {A: {density, congestion_level, los}, B: {...}}`, and
`/api/traffic/roads` renders each road as two polylines offset ±6 m from the
OSM centreline, each coloured by its own lane's congestion. Flow-based
counting could not have supported this: a stopped lane generates no crossings
at all, and so would have been invisible.

- Split line default: vertical centre of the frame.
- Calibration: `GET/POST /api/cctvs/<id>/lane_line` with
  `{"split_line": [[x1,y1],[x2,y2]], "flip": false}` in normalised (0–1)
  image coordinates; persisted in `lane_config.json`.
- Per-lane thresholds are stricter (a lane is roughly half the scene):
  `<3` free, `<7` moderate, `<12` congested, else severe
  (`lane_utils.lane_congestion`).

## 5. Why the current method is better — summary

1. **It cannot confuse a jam with an empty road.** Density is monotone in
   "how full is the road"; flow is not. This was the actual observed bug.
2. **It matches traffic-engineering practice.** LOS for road segments is
   defined on density (veh/km/lane in the HCM); the thresholds here are the
   per-camera-view analogue.
3. **The speed term resolves density's one ambiguity.** At the same density,
   "flowing" and "stopped" are different states; the stopped fraction
   distinguishes them at zero extra calibration cost (DeepSORT already runs).
4. **It degrades gracefully.** With no usable tracks (DeepSORT unavailable,
   or the single-frame background estimator), the classifier falls back to
   density-only instead of failing or lying.
5. **One shared classifier for both pipelines.** The live detector and the
   30-second background estimator use the same function, so a camera no
   longer changes its congestion opinion when you open its popup.
6. **It enables per-lane reporting.** Presence splits by region; crossings of
   a stationary queue do not exist.
7. **Nothing is lost.** Throughput (veh/min) is still measured and shown —
   it is a useful *capacity* number — it just no longer masquerades as a
   *congestion* number.

## 6. Known limitations & tuning

- **Thresholds are FOV-dependent.** A camera seeing 300 m of road holds more
  vehicles at free flow than one seeing 50 m. The `<5/<12/<20` defaults fit
  the Semarang intersection cameras; tune per site if a camera is
  systematically over/under-reporting.
- **Stopped detection uses pixel speed, not real speed.** ~1% of frame height
  per second is a heuristic for "not moving"; far-field vehicles move fewer
  pixels, so distant slow traffic can read as stopped. Acceptable for a
  one-level bump; do not use it as a speed measurement.
- **Night recall.** YOLO at `conf=0.3` misses vehicles in low light, which
  deflates density (conservative direction: under-reports congestion).
- **Lane split needs calibration** on cameras where the two directions are
  not left/right of the frame centre, and the A/B ↔ geographic side mapping
  is a convention — use the `flip` flag when a camera's lanes render swapped
  on the map.
