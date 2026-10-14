# Jal-SAR

Jal-SAR ("jal", जल, is water in Hindi; SAR is the radar) makes Sentinel-1
flood maps of the Alaknanda corridor (Rudraprayag to Rishikesh) and says, for each part of a
map, how far it can be trusted. It is built on the three models of this
project (Random Forest, U-Net, TerraMind-small) and on their accuracy by
terrain class (Table D).

Status:

| Part | File | State |
|---|---|---|
| Core: maps, agreement, trust layer, flood polygons, village and road exposure | `jalsar_core.py` | built, tested on synthetic data |
| Command line | `jalsar.py` (`events`, `roads`, `maps`, `bulletin`, `webmap`, `run`, `check-download`) | built, tested |
| Web map (one HTML file, every event) | `jalsar_webmap.py` | built, tested |
| Bulletin (HTML and PDF, maroon header) | `jalsar_bulletin.py`, `templates/bulletin.html.j2` | built, tested (PDF with Chromium in the cloud; Edge on the laptop) |
| Runner for new dates (Earth Engine download, strata, maps) | `jalsar_runner.py`, `jalsar.env.example` | built; download tested with a stand-in; needs the 21 Oct 2021 check on the laptop |
| Shared colours, wording, pictures | `jalsar_display.py` | built, tested |
| Synthetic events for the self-tests | `jalsar_testdata.py` | test support |

Launchers (double-click):

| File | Does |
|---|---|
| `RUN JAL-SAR MAPS.bat` | roads once, maps for every event on disk, bulletins, web map |
| `RUN JAL-SAR MAPS AGAIN.bat` | the same, replacing what is there (after a change to Jal-SAR's code) |
| `RUN JAL-SAR CHECK DOWNLOAD.bat` | downloads 21 Oct 2021 again and compares it with `05 Data` (run once before any new date) |
| `RUN JAL-SAR.bat` | a new date: asks for it, then the whole chain |

Outputs go to `06 Outputs\jalsar\` (one folder per event, `webmap\`, `scenes\`,
`checks\`). Nothing is written into `05 Data`. Strata for a new orbit go to
`FYP_code\data\strata_orbitNNN_<pass>\`.

Settings for the runner: `jalsar.env`, made from `jalsar.env.example` on
the first run and kept out of git, so editing it never blocks a `git pull`.

The web map has one side panel: choose the event, switch five layers (flood
map by trust, where the models agree, trust across the corridor, villages,
reference windows), each described in a line of what it is and why it is
there, and pick the background. The background is terrain shading made from
the Copernicus DEM, inside the file; a satellite picture (Esri World Imagery)
is optional. OpenStreetMap background tiles are not used: their servers refuse
a page opened from a file on disk (9 October). The page needs the internet
only for the map library (Leaflet) and the optional satellite picture.

Village figures: "share in flood" is the mapped flood over the whole village
area; villages are listed largest flood area first; the bulletin gives both
the villages touching the flood and those with at least 1 ha of it
(`VILLAGE_MIN_KM2` in `jalsar_core.py`). Population is for whole villages.

What Jal-SAR is not: a warning system, an operational accuracy claim, or a
depth or damage estimate. It maps the corridor only.

Village households and population are Census 2011 (Survey of India village
boundaries). Roads (c) OpenStreetMap contributors, ODbL.
