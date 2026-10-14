// ===========================================================================
// 01_export_s1.js
// Sentinel-1 GRD export for the Alaknanda corridor, plus DEM and land cover.
//
// HOW TO USE
//   1. Go to https://code.earthengine.google.com
//   2. Make sure the project shown top right is gbpuat-fyp-alaknanda
//   3. Paste this whole file into the code panel
//   4. Click Run
//   5. Look at the map to check the corridor sits on the river
//   6. Open the Tasks tab on the right. Click RUN on each task.
//   7. Files land in a Google Drive folder called FYP_exports.
//      Download them into 05 Data\sentinel1\ and 05 Data\dem\
//
// The four Sentinel-1 dates come from 01_scene_inventory.py, which found
// 100% corridor coverage on all four.
//
// NOTE ON RTC. COPERNICUS/S1_GRD is the standard GRD product: thermal noise
// removed, radiometrically calibrated to sigma0 in dB, terrain corrected
// against SRTM. It is NOT radiometrically terrain corrected. True S1RTC for
// this corridor is still an open question, and the Vollrath angular slope
// correction is the likely substitute. That ablation is a separate job.
// ===========================================================================

// --- corridor, same seven anchor points and 5 km buffer as the Python -------
var pts = [
  [78.9811, 30.2844],  // Rudraprayag
  [78.9000, 30.2400],
  [78.7833, 30.2222],  // Srinagar (Garhwal)
  [78.6800, 30.1900],
  [78.5981, 30.1469],  // Devprayag
  [78.4300, 30.1300],
  [78.2676, 30.0869]   // Rishikesh
];
var centreline = ee.Geometry.LineString(pts);
var corridor  = centreline.buffer(5000);

Map.centerObject(corridor, 9);
Map.addLayer(corridor, {color: 'yellow'}, 'corridor (5 km buffer)');
Map.addLayer(centreline, {color: 'red'}, 'centreline (approximate)');
print('corridor area, km2:', corridor.area(1).divide(1e6));

var CRS   = 'EPSG:32644';   // UTM zone 44N, correct for 78-84 E
var DRIVE = 'FYP_exports';

// --- one Sentinel-1 scene ---------------------------------------------------
function s1scene(dateStr, orbit, pass) {
  var d0 = ee.Date(dateStr);
  var col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(corridor)
    .filterDate(d0, d0.advance(1, 'day'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('relativeOrbitNumber_start', orbit))
    .filter(ee.Filter.eq('orbitProperties_pass', pass))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
  print(dateStr + '  orbit ' + orbit + ' ' + pass + '  scenes:', col.size());
  // 'angle' is the per-pixel ellipsoidal incidence angle. 03_terrain_strata.py
  // needs it to compute local incidence, layover and shadow properly. Without
  // it that script falls back to a constant 38.5 degrees and records the
  // substitution in strata_meta.json.
  return col.mosaic().select(['VV', 'VH', 'angle']).clip(corridor);
}

// --- the four scenes --------------------------------------------------------
var scenes = [
  {tag: '2023_post', date: '2023-08-16', orbit: 129, pass: 'ASCENDING'},
  {tag: '2023_pre',  date: '2022-11-01', orbit: 129, pass: 'ASCENDING'},
  {tag: '2021_post', date: '2021-10-21', orbit: 63,  pass: 'DESCENDING'},
  {tag: '2021_pre',  date: '2020-11-19', orbit: 63,  pass: 'DESCENDING'}
];

// Extra winter pre-images, same orbit as the 2023 event. Early November is
// still draining after the monsoon, so compare against these before you
// settle on a reference date.
var extraPre = [
  {tag: '2023_pre_alt_dec', date: '2022-12-19', orbit: 129, pass: 'ASCENDING'},
  {tag: '2023_pre_alt_jan', date: '2023-01-12', orbit: 129, pass: 'ASCENDING'},
  {tag: '2023_pre_alt_feb', date: '2023-02-05', orbit: 129, pass: 'ASCENDING'}
];

var all = scenes.concat(extraPre);

all.forEach(function (s) {
  var img = s1scene(s.date, s.orbit, s.pass);
  Map.addLayer(img.select('VV'), {min: -25, max: 0}, 'VV ' + s.tag, false);
  Export.image.toDrive({
    image: img.toFloat(),
    description: 'S1_' + s.tag + '_' + s.date.replace(/-/g, ''),
    folder: DRIVE,
    fileNamePrefix: 'S1_' + s.tag + '_' + s.date.replace(/-/g, '') +
                    '_orb' + s.orbit,
    region: corridor,
    scale: 10,
    crs: CRS,
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });
});

// --- terrain ---------------------------------------------------------------
// If COPERNICUS/DEM/GLO30_2024_1 throws "not found", fall back to
// COPERNICUS/DEM/GLO30 on the line below.
var glo30 = ee.ImageCollection('COPERNICUS/DEM/GLO30_2024_1')
              .select('DEM').mosaic().clip(corridor);

Export.image.toDrive({
  image: glo30.toFloat(),
  description: 'DEM_COP_GLO30',
  folder: DRIVE,
  fileNamePrefix: 'DEM_COP_GLO30_30m',
  region: corridor, scale: 30, crs: CRS, maxPixels: 1e10
});

var srtm = ee.Image('USGS/SRTMGL1_003').clip(corridor);
Export.image.toDrive({
  image: srtm.toFloat(),
  description: 'DEM_SRTM',
  folder: DRIVE,
  fileNamePrefix: 'DEM_SRTM_30m',
  region: corridor, scale: 30, crs: CRS, maxPixels: 1e10
});

// --- land cover -------------------------------------------------------------
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(corridor);
Export.image.toDrive({
  image: wc,
  description: 'LC_ESA_WorldCover',
  folder: DRIVE,
  fileNamePrefix: 'LC_ESA_WorldCover_10m',
  region: corridor, scale: 10, crs: CRS, maxPixels: 1e10
});

// --- permanent water, post-2015 only ---------------------------------------
// The Srinagar hydroelectric reservoir filled in 2015. Anything earlier will
// miss it and the model will read standing reservoir water as flooding.
var jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            .select('occurrence').clip(corridor);
Export.image.toDrive({
  image: jrc.toFloat(),
  description: 'WATER_JRC_occurrence',
  folder: DRIVE,
  fileNamePrefix: 'WATER_JRC_occurrence_30m',
  region: corridor, scale: 30, crs: CRS, maxPixels: 1e10
});

print('Ten export tasks queued. Open the Tasks tab and press RUN on each.');
