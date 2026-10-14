// ===========================================================================
// 09_export_s2_july2023.js
// Sentinel-2 for the 19 July 2023 scene: the 20 and 23 July pair.
// Written 9 September 2026.
//
// WHY THIS EXISTS
// ---------------
// 07_export_july2023.js exports the Sentinel-1 side of the July 2023 scene and
// says, in its own header, that no Sentinel-2 is exported because the scene was
// added unvalidated. That decision has been revisited: the July scene only
// becomes a validated third event if there is optical imagery good enough to
// digitise a reference extent from, and the only way to know is to test it.
//
// The test is a coverage test, not a preference. Neither date is clear on its
// own. 20 July is about 75 per cent corridor-clear and 23 July about 32 per
// cent. They reach a usable mosaic only if their clouds sit in different
// places. The script prints the answer before it exports anything.
//
//   If the merged pair reaches the 99.5 per cent threshold, July 2023 goes in
//   as a validated event.
//   If it does not, July 2023 stays unvalidated and that outcome is reported.
//
// Either answer is a result. Do not move the dates around until the number
// improves: choosing the window until the answer is the one you wanted is
// choosing the answer.
//
// BAND CONTRACT
// -------------
// It must match every Sentinel-2 scene already on disk, exactly, or t02_mosaic
// will read the wrong band as the wrong thing and say nothing.
//
//   *_10m.tif          six bands, Int16, in the order B2, B3, B4, B8, B11, SCL
//   *_indices_10m.tif  two bands, float32, MNDWI then NDWI
//                      MNDWI = normalizedDifference(B3, B11)
//                      NDWI  = normalizedDifference(B3, B8)
//   scale 10, CRS EPSG:32644, Drive folder FYP_exports
//
// HOW TO USE
//   1. https://code.earthengine.google.com, project gbpuat-fyp-alaknanda
//   2. New script, paste this whole file in
//   3. Click Run
//   4. READ THE CONSOLE FIRST. Three blocks print: the inventory of July
//      dates, the per-date clear percentage, and the coverage of the pair.
//      Check all three before you press RUN on anything.
//   5. Only if the pair block says the merge is worth having, open the Tasks
//      tab and press RUN on all four tasks
//   6. Download into 05 Data\sentinel2\ keeping the file names exactly as
//      Earth Engine writes them
// ===========================================================================

// --- corridor, identical to 01_export_s1.js and 03_export_remaining.js ------
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

var CRS   = 'EPSG:32644';
var DRIVE = 'FYP_exports';

Map.centerObject(corridor, 9);
Map.addLayer(corridor, {color: 'yellow'}, 'corridor (5 km buffer)');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');

// --- inventory -------------------------------------------------------------
// SCL: 0 nodata, 3 cloud shadow, 8 cloud medium, 9 cloud high, 10 cirrus,
// 11 snow and ice. Snow is counted separately even in July, because the upper
// corridor carries it on the high sides and it is not cloud.
function s2Row(dateStr) {
  var d0  = ee.Date(dateStr);
  var day = s2.filterBounds(corridor).filterDate(d0, d0.advance(1, 'day'));
  var scl = day.select('SCL').mosaic().unmask(0).clip(corridor);

  var covered = scl.gt(0).rename('covered');
  var cloud   = scl.eq(3).or(scl.gte(8).and(scl.lte(10))).rename('cloud');
  var snow    = scl.eq(11).rename('snow');
  var clear   = covered.and(cloud.not()).and(snow.not()).rename('clear');

  var m = covered.addBands(cloud).addBands(snow).addBands(clear).toFloat()
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: corridor,
                   scale: 100, maxPixels: 1e9, bestEffort: true});

  function pct(k) { return ee.Number(m.get(k)).multiply(100).format('%6.1f'); }

  return ee.String(dateStr)
    .cat('   cov ').cat(pct('covered'))
    .cat('   CLEAR ').cat(pct('clear'))
    .cat('   cloud ').cat(pct('cloud'))
    .cat('   snow ').cat(pct('snow'));
}

var WIN_START = '2023-07-10';
var WIN_END   = '2023-08-01';

var julyDates = s2.filterBounds(corridor).filterDate(WIN_START, WIN_END)
  .aggregate_array('system:time_start')
  .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); })
  .distinct().sort();

print(ee.String('=== 1. Sentinel-2 over the corridor, ' + WIN_START +
                ' to ' + WIN_END + ' ===\n')
        .cat('Not pre-filtered on tile cloud: in the monsoon that filter\n')
        .cat('throws away the scenes you are trying to measure.\n')
        .cat('date              cov      CLEAR     cloud      snow\n')
        .cat(ee.List(julyDates.map(s2Row)).join('\n')));

// --- the pair test ---------------------------------------------------------
// The two dates come from the event-window inventory that has already been
// run. 20 July is the +1 lag scene after 19 July and the clearer of the two.
var PAIR = ['2023-07-20', '2023-07-23'];

function clearMask(dateStr) {
  var d0  = ee.Date(dateStr);
  var scl = s2.filterBounds(corridor).filterDate(d0, d0.advance(1, 'day'))
              .select('SCL').mosaic().unmask(0).clip(corridor);
  var covered = scl.gt(0);
  var cloud   = scl.eq(3).or(scl.gte(8).and(scl.lte(10)));
  var snow    = scl.eq(11);
  return covered.and(cloud.not()).and(snow.not());
}

function pctOf(img) {
  return ee.Number(img.toFloat().reduceRegion({
    reducer: ee.Reducer.mean(), geometry: corridor,
    scale: 100, maxPixels: 1e9, bestEffort: true
  }).values().get(0)).multiply(100).format('%6.2f');
}

var a = clearMask(PAIR[0]);
var b = clearMask(PAIR[1]);

print(ee.String('=== 2. THE DECISION. Clear corridor, ' + PAIR[0] +
                ' and ' + PAIR[1] + ' ===\n')
  .cat(PAIR[0] + ' alone       ').cat(pctOf(a)).cat(' %\n')
  .cat(PAIR[1] + ' alone       ').cat(pctOf(b)).cat(' %\n')
  .cat('both mosaicked          ').cat(pctOf(a.or(b))).cat(' %\n')
  .cat('\nThe threshold is 99.5 per cent. Above it, July 2023 becomes a\n')
  .cat('validated event with a digitised reference. Below it, July 2023\n')
  .cat('stays an unvalidated third scene and the shortfall is reported.\n')
  .cat('Check this block before running any task.'));

Map.addLayer(a.selfMask(), {palette: ['00a0ff']}, 'clear ' + PAIR[0], false);
Map.addLayer(b.selfMask(), {palette: ['ff8000']}, 'clear ' + PAIR[1], false);
Map.addLayer(a.or(b).not().selfMask(), {palette: ['ff0000']},
             'gap, neither date clear', true);

// --- exports ---------------------------------------------------------------
// Four tasks: two scenes, each with a reflectance file and an indices file.
// The tags follow the naming already used for the dry-season pairs.
var S2_EXPORT = [
  {tag: '2023jul_event_a', date: PAIR[0]},
  {tag: '2023jul_event_b', date: PAIR[1]}
];

S2_EXPORT.forEach(function (s) {
  var d0  = ee.Date(s.date);
  var img = s2.filterBounds(corridor).filterDate(d0, d0.advance(1, 'day'))
              .mosaic().clip(corridor);
  var stamp = s.date.replace(/-/g, '');

  Export.image.toDrive({
    image: img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'SCL']).toInt16(),
    description: 'S2_' + s.tag + '_' + stamp,
    folder: DRIVE,
    fileNamePrefix: 'S2_' + s.tag + '_' + stamp + '_10m',
    region: corridor, scale: 10, crs: CRS,
    maxPixels: 1e10, fileFormat: 'GeoTIFF'
  });

  var mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI');
  var ndwi  = img.normalizedDifference(['B3', 'B8']).rename('NDWI');

  Export.image.toDrive({
    image: mndwi.addBands(ndwi).toFloat(),
    description: 'S2_' + s.tag + '_' + stamp + '_indices',
    folder: DRIVE,
    fileNamePrefix: 'S2_' + s.tag + '_' + stamp + '_indices_10m',
    region: corridor, scale: 10, crs: CRS,
    maxPixels: 1e10, fileFormat: 'GeoTIFF'
  });
});

print('=== 3. Four tasks queued, about 700 MB. Do not press RUN until block 2 '
    + 'has been read and checked. ===');
