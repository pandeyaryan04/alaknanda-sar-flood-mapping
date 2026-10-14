// ===========================================================================
// 04_catchment_rainfall.js
// CHIRPS daily rainfall over the UPSTREAM CATCHMENT, not the corridor.
// Written 5 September 2026.
//
// WHY THIS EXISTS, AND WHY 03's RAINFALL EXPORT WAS THE WRONG QUESTION
// -------------------------------------------------------------------
// Script 03 exported corridor-mean daily rainfall. That was a mistake and it
// is worth being clear about, because the numbers it produced look like an
// answer and are not one.
//
// The Alaknanda at Rudraprayag drains 8602 km2, a figure taken from the
// India-WRIS station record on your own disk. The corridor is a 5 km strip
// around the lower 90 km of river, 808 km2. The rain that floods the corridor
// falls almost entirely UPSTREAM of it, in the high Garhwal catchment above
// Rudraprayag, which the corridor polygon does not touch.
//
// So corridor-mean rainfall describes what fell on the study area. It does not
// test whether a flood was meteorologically plausible, which is the question
// the rainfall data exists to answer. This script asks the right one.
//
// It also self-checks: it prints the catchment area it derived, and that should
// land near 8602 km2. If it does not, the traversal is wrong and nothing below
// it should be believed.
//
// HOW TO USE
//   1. https://code.earthengine.google.com, project gbpuat-fyp-alaknanda
//   2. Paste this whole file into a new script
//   3. Click Run
//   4. Check the printed catchment area against 8602 km2
//   5. Look at the catchment outline on the map. It should sit above
//      Rudraprayag and cover the Alaknanda headwaters, not the reach you study
//   6. Open the Tasks tab, press RUN on the three tasks
//   7. Download into 05 Data\rainfall\ and 05 Data\boundaries\
// ===========================================================================

var DRIVE = 'FYP_exports';
var CRS   = 'EPSG:32644';

// Rudraprayag gauge, from the station record in 05 Data\gauges\.
// Station 004-HGDDDN, zero gauge datum 614.00 m, catchment 8602 km2.
var GAUGE = ee.Geometry.Point([78.9842, 30.288061]);
var GAUGE_CATCHMENT_KM2 = 8602;

// HydroSHEDS level 7 basins. Level 7 is fine here: the Alaknanda above
// Rudraprayag breaks into a manageable number of them, and going finer makes
// the upstream traversal slower without changing the polygon much.
var hybas = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_7');

// Walk upstream. Start from the basin containing the gauge, then repeatedly
// add every basin whose NEXT_DOWN is already in the set. Twelve passes is more
// than the Alaknanda needs at this level; extra passes cost nothing because
// the set stops growing.
var seed = hybas.filterBounds(GAUGE).aggregate_array('HYBAS_ID');

var upstreamIds = ee.List(
  ee.List.sequence(1, 12).iterate(function (i, acc) {
    acc = ee.List(acc);
    var next = hybas.filter(ee.Filter.inList('NEXT_DOWN', acc))
                    .aggregate_array('HYBAS_ID');
    return acc.cat(next).distinct();
  }, seed)
);

var basins    = hybas.filter(ee.Filter.inList('HYBAS_ID', upstreamIds));
var catchment = basins.union(1).geometry();

var areaKm2 = catchment.area(100).divide(1e6);
print('basins found upstream of the gauge:', basins.size());
print('derived catchment area, km2:', areaKm2);
print('India-WRIS says the gauge catchment is, km2:', GAUGE_CATCHMENT_KM2);
print('ratio, derived over reported (want something near 1):',
      areaKm2.divide(GAUGE_CATCHMENT_KM2));

Map.centerObject(catchment, 8);
Map.addLayer(catchment, {color: 'cyan'}, 'upstream catchment');
Map.addLayer(GAUGE, {color: 'red'}, 'Rudraprayag gauge');

// --- daily rainfall over that catchment ------------------------------------
// Windows run well before each event. Antecedent rainfall is the part that
// makes a flood plausible: the day of the peak says nothing on its own about
// whether the catchment was already saturated.
var RAIN = [
  {tag: '2023', start: '2023-06-01', end: '2023-08-31'},
  {tag: '2021', start: '2021-08-01', end: '2021-10-31'}
];

var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');

RAIN.forEach(function (w) {
  var rows = chirps.filterDate(w.start, w.end).map(function (img) {
    var s = img.reduceRegion({
      reducer: ee.Reducer.mean().combine(ee.Reducer.max(), '', true),
      geometry: catchment, scale: 5000, maxPixels: 1e9, bestEffort: true
    });
    return ee.Feature(null, {
      date: img.date().format('YYYY-MM-dd'),
      rain_mean_mm: s.get('precipitation_mean'),
      rain_max_mm:  s.get('precipitation_max')
    });
  });

  Export.table.toDrive({
    collection: ee.FeatureCollection(rows),
    description: 'RAIN_CHIRPS_catchment_' + w.tag,
    folder: DRIVE,
    fileNamePrefix: 'RAIN_CHIRPS_catchment_daily_' + w.tag,
    fileFormat: 'CSV',
    selectors: ['date', 'rain_mean_mm', 'rain_max_mm']
  });
});

// The catchment mean smooths a cloudburst into nothing, which is exactly how a
// real event can look like a quiet week. rain_max_mm is the wettest 5 km cell
// in the catchment on that day, and it is the column that shows a localised
// storm. Report both or you will talk yourself out of a real event.

// --- the catchment polygon itself, worth keeping ---------------------------
Export.table.toDrive({
  collection: ee.FeatureCollection([ee.Feature(catchment, {
    name: 'Alaknanda catchment above Rudraprayag',
    source: 'WWF/HydroSHEDS/v1/Basins/hybas_7, upstream traversal on NEXT_DOWN'
  })]),
  description: 'BOUNDARY_catchment_above_Rudraprayag',
  folder: DRIVE,
  fileNamePrefix: 'BOUNDARY_catchment_above_Rudraprayag',
  fileFormat: 'GeoJSON'
});

print('Three tasks queued: two rainfall CSVs and the catchment polygon.');
print('Check the area ratio above BEFORE pressing RUN. If it is not near 1, ' +
      'the traversal picked up the wrong basins and the rainfall is for the ' +
      'wrong ground.');
