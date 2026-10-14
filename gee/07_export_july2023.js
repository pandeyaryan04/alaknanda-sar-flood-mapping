// ===========================================================================
// 07_export_july2023.js
// The 19 July 2023 scene and its winter baselines, orbit 63 descending.
// Written 5 September 2026.
//
// WHY THIS SCENE
// --------------
// Decided today: 19 July 2023 is added as an unvalidated third scene. It is
// not replacing 16 August, which keeps the better optical reference and stays
// the validated 2023 event. July is here for two specific jobs:
//
//   1. It is on orbit 63 DESCENDING, the same relative orbit and the same look
//      geometry as the 2021 event. Right now every difference between your two
//      events is confounded with geometry, because one is ascending and one is
//      descending. This scene removes that confound for one comparison.
//
//   2. It follows the wettest period of the 2023 monsoon. Antecedent 7-day
//      rainfall is 150.3 mm against 52.1 mm for the August scene, 30-day is
//      393.1 mm against 208.9 mm, and 15 July had a wettest cell of 322 mm.
//      If change detection is measuring flood rather than noise, this scene
//      should show more water than the August one. If it does not, that is a
//      finding about the method.
//
// No Sentinel-2 is exported. The decision was to add this scene unvalidated,
// so it gets no reference extent and no digitising. If that changes later, the
// optical scene to reach for is 20 July 2023 at 75.4 per cent corridor-clear,
// lag +1.
//
// HOW TO USE
//   1. https://code.earthengine.google.com, project gbpuat-fyp-alaknanda
//   2. New script, paste this whole file in
//   3. Click Run
//   4. READ THE INVENTORY FIRST. Check that the three winter dates in
//      WINTER_EXPORT below actually appear in the printed list. Sentinel-1B
//      failed in December 2021, so from 2022 onward this orbit is Sentinel-1A
//      alone on a strict 12-day repeat and a date can be missing.
//   5. If a date is missing, swap it for the nearest one that is listed
//   6. Open the Tasks tab, press RUN on all four tasks
//   7. Download into 05 Data\sentinel1\
// ===========================================================================

var pts = [
  [78.9811, 30.2844], [78.9000, 30.2400], [78.7833, 30.2222], [78.6800, 30.1900],
  [78.5981, 30.1469], [78.4300, 30.1300], [78.2676, 30.0869]
];
var corridor = ee.Geometry.LineString(pts).buffer(5000);

var CRS   = 'EPSG:32644';
var DRIVE = 'FYP_exports';

Map.centerObject(corridor, 9);
Map.addLayer(corridor, {color: 'yellow'}, 'corridor');

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(corridor)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('relativeOrbitNumber_start', 63))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

// --- inventory: orbit 63 winter 2022-23 ------------------------------------
var WIN_START = '2022-11-15';
var WIN_END   = '2023-03-15';

function covRow(dateStr) {
  var d0  = ee.Date(dateStr);
  var day = s1.filterDate(d0, d0.advance(1, 'day'));
  var cov = day.select('VV').mosaic().mask().unmask(0).clip(corridor)
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: corridor,
                   scale: 200, maxPixels: 1e9, bestEffort: true}).get('VV');
  return ee.String(dateStr)
    .cat('   coverage ').cat(ee.Number(cov).multiply(100).format('%6.1f'))
    .cat('   scenes ').cat(day.size().format('%2.0f'));
}

var winterDates = s1.filterDate(WIN_START, WIN_END)
  .aggregate_array('system:time_start')
  .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); })
  .distinct().sort();

print(ee.String('=== orbit 63 DESCENDING, ' + WIN_START + ' to ' + WIN_END +
                ' ===\n')
        .cat('Check the three dates in WINTER_EXPORT appear below.\n')
        .cat(ee.List(winterDates.map(covRow)).join('\n')));

// --- what to export --------------------------------------------------------
// The winter dates are the 12-day repeat stepped back from 19 July 2023.
// They mirror the December, January and February baselines already used for
// the other two events, so all three scenes get the same treatment.
var POST_EXPORT = [
  {tag: '2023jul_post', date: '2023-07-19'}
];

// CORRECTED 5 September, after the inventory ran. 2023-01-20 came from
// arithmetic, stepping back from 19 July in 12-day multiples. The inventory
// shows the acquisitions on this orbit run 12-15, 12-27, 01-08, then jump
// straight to 02-01. There was no 20 January acquisition over this corridor.
// That is why the map layer threw "Band pattern 'VV' was applied to an Image
// with no bands": the filter matched nothing and mosaic() of an empty
// collection has no bands. 2023-01-08 is the real January scene.
var WINTER_EXPORT = [
  {tag: '2023jul_pre_dec', date: '2022-12-27'},
  {tag: '2023jul_pre_jan', date: '2023-01-08'},
  {tag: '2023jul_pre_feb', date: '2023-02-13'}
];

POST_EXPORT.concat(WINTER_EXPORT).forEach(function (s) {
  var d0 = ee.Date(s.date);
  // 'angle' is the per-pixel incidence angle. 03_terrain_strata.py needs it
  // and silently falls back to a constant 38.5 degrees without it.
  var img = s1.filterDate(d0, d0.advance(1, 'day'))
              .mosaic().select(['VV', 'VH', 'angle']).clip(corridor);
  var stamp = s.date.replace(/-/g, '');
  Export.image.toDrive({
    image: img.toFloat(),
    description: 'S1_' + s.tag + '_' + stamp,
    folder: DRIVE,
    fileNamePrefix: 'S1_' + s.tag + '_' + stamp + '_orb63',
    region: corridor, scale: 10, crs: CRS,
    maxPixels: 1e10, fileFormat: 'GeoTIFF'
  });
  Map.addLayer(img.select('VV'), {min: -25, max: 0}, 'VV ' + s.tag, false);
});

print('Four tasks queued, about 420 MB. Read the inventory before running ' +
      'them.');
