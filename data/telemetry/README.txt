Falcon 9 first-stage webcast telemetry for four return-to-launch-site landings (CRS-11, CRS-12, SAOCOM 1A,
ZUMA), copied unchanged from shahar603/Telemetry-Data (https://github.com/shahar603/Telemetry-Data, the
"stage1 raw.json" file of each mission), which is released into the public domain under the Unlicense
(LICENSE in this folder). Fields: time (s), velocity (m/s), altitude (km), extracted from the SpaceX webcast
overlay at 30 frames per second. Altitude is shown to 0.1 km, so it is coarse near the ground.
Used by landingSim/analysis/telemetry.py and scripts/compareTelemetry.py; see docs/references.txt [11], [D1]-[D4].
