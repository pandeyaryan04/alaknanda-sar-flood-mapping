// ===========================================================================
// 05_s1_timing_inventory.js
// Every Sentinel-1 acquisition over the corridor near each rainfall peak,
// on any orbit and either pass.
// Written 5 September 2026.
//
// WHY THIS EXISTS
// ---------------
// The catchment rainfall came back today and it changed what we know about
// timing. Both current acquisitions sit THREE DAYS after their rainfall peak:
//
//   2021: 68.6 mm catchment mean on 18 October, SAR on 21 October
//   2023: 36.3 mm catchment mean on 13 August,  SAR on 16 August
//
// Three days after the rain in an 8638 km2 Himalayan catchment is well past
// the peak. Both flood maps are therefore mapping recession, and the 1.8 to
// 3.1 km2 they report is what was left, not what happened.
//
// The two scenes on disk were chosen from orbit 129 ascending and orbit 63
// descending only. Sentinel-1 covers this corridor from more than one relative
// orbit, and a different one may pass one or two days after a peak instead of
// three. One or two days instead of three is not a detail: on a flashy
// mountain catchment it can be the difference between mapping a flood and
// mapping a river.
//
// It also matters that the two largest rainfall days of the 2023 monsoon were
// 25 June (85.3 mm mean, 218 mm max) and 15 July (66.7 mm mean, 322 mm max),
// both far larger than anything in August. Nobody has checked whether a usable
// SAR scene exists after those.
//
// This script does not commit you to anything. It tells you what exists.
// Changing the event scene would mean rebuilding the strata for that scene,
// which is a run, not a rewrite, and a different look geometry supports the
// two-look-geometries claim rather than undermining it.
//
// HOW TO USE
//   1. https://code.earthengine.google.com, project gbpuat-fyp-alaknanda
//   2. New script, paste this whole file in
//   3. Click Run
//   4. Copy the four text blocks out of the Console and keep them
//
// Nothing exports. This is an inventory only.
// ===========================================================================

var pts = [
  [78.9811, 30.2844],  // Rudraprayag
  [78.9000, 30.2400],
  [78.7833, 30.2222],  // Srinagar (Garhwal)
  [78.6800, 30.1900],
  [78.5981, 30.1469],  // Devprayag
  [78.4300, 30.1300],
  [78.2676, 30.0869]   // Rishikesh
];
var corridor = ee.Geometry.LineString(pts).buffer(5000);

Map.centerObject(corridor, 9);
Map.addLayer(corridor, {color: 'yellow'}, 'corridor');

// The rainfall peaks, read off the catchment CHIRPS export.
// A window of -2 to +8 days: nothing before the rain is useful as a post
// image, and past about a week the flood is gone on a catchment this size.
var PEAKS = [
  {tag: '2021 peak, 18 Oct 2021, 68.6 mm mean / 149.8 mm max', date: '2021-10-18'},
  {tag: '2023 peak, 13 Aug 2023, 36.3 mm mean /  81.8 mm max', date: '2023-08-13'},
  {tag: '2023 BIGGEST, 15 Jul 2023, 66.7 mm mean / 322.0 mm max', date: '2023-07-15'},
  {tag: '2023 second, 25 Jun 2023, 85.3 mm mean / 218.3 mm max', date: '2023-06-25'}
];

var DAYS_BEFORE = 2;
var DAYS_AFTER  = 8;

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(corridor)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

function row(img, peakDate) {
  img = ee.Image(img);
  var d = ee.Date(img.get('system:time_start'));

  // Coverage of the corridor by THIS acquisition, which is what decides
  // whether a scene is usable at all. A scene reaching 60 per cent of the
  // corridor is half a scene and the missing half becomes nodata.
  var cov = img.select('VV').mask().unmask(0).clip(corridor)
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: corridor,
                   scale: 200, maxPixels: 1e9, bestEffort: true}).get('VV');

  return ee.String(d.format('YYYY-MM-dd HH:mm'))
    .cat('  T+').cat(d.difference(ee.Date(peakDate), 'day').format('%4.1f'))
    .cat('  orbit ').cat(ee.Number(img.get('relativeOrbitNumber_start')).format('%4.0f'))
    .cat('  ').cat(ee.String(img.get('orbitProperties_pass')).slice(0, 4))
    .cat('  coverage ').cat(ee.Number(cov).multiply(100).format('%6.1f'))
    .cat('  ').cat(ee.String(img.get('system:index')).slice(0, 20));
}

PEAKS.forEach(function (p) {
  var start = ee.Date(p.date).advance(-DAYS_BEFORE, 'day');
  var end   = ee.Date(p.date).advance(DAYS_AFTER + 1, 'day');
  var col   = s1.filterDate(start, end).sort('system:time_start');

  var rows = col.toList(60).map(function (img) { return row(img, p.date); });

  print(ee.String('=== ' + p.tag + ' ===\n')
    .cat('acquisitions from T-' + DAYS_BEFORE + ' to T+' + DAYS_AFTER + '\n')
    .cat('date        time    lag    orbit  pass  coverage  scene id\n')
    .cat(ee.List(rows).join('\n')));
});

// ---------------------------------------------------------------------------
// HOW TO READ IT
//
// Look for a row with coverage near 100 and the smallest positive lag.
// A single acquisition rarely covers the whole corridor: if the best-timed
// scene reaches only part of it, check whether a second acquisition a day or
// two later fills the rest, the same way two Sentinel-2 dates mosaicked to
// exactly 100 per cent.
//
// Orbit 129 ascending and orbit 63 descending are the two already on disk.
// A row on a different relative orbit is not a problem. It is a different look
// geometry over the same reach, which is the thing this project set out to
// study, and 03_terrain_strata.py rebuilds the strata for whatever pass you
// give it.
//
// What would change the project:
//   a scene at T+1 or T+2 with full coverage, on either event
//   a usable scene after 15 July 2023, which was the largest rainfall day of
//     that monsoon by maximum cell and is currently not looked at at all
// ---------------------------------------------------------------------------

print('Inventory only. Nothing exports from this script.');
