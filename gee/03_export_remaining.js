// ===========================================================================
// 03_export_remaining.js
// The rest of the Earth Engine data for the Alaknanda corridor project.
// Written 5 September 2026.
//
// Three sections, each one an inventory you read first and an export you fill
// in second. Nothing exports until you put dates in the three EXPORT lists at
// the bottom of each section.
//
//   A.  Sentinel-1, winter 2021, orbit 63 descending
//       A dry-season pre-image for the 2021 event. The one you have is
//       19 November 2020, still draining after the monsoon, so change
//       detection against it compares wet against wetter.
//
//   B.  Sentinel-2, dry season, one scene per event year
//       For digitising the permanent channel, so the reference extent can be
//       flood rather than total water. Taking the permanent channel from the
//       30 m JRC layer instead would leave a false-flood rim three cells wide
//       along all 90 km of a 10 m reference.
//
//   C.  CHIRPS daily rainfall, corridor mean, as a CSV
//       Event context. A table of daily millimetres is more use in a report
//       than a stack of 5 km rasters, so this exports the table.
//
// HOW TO USE
//   1. https://code.earthengine.google.com, project gbpuat-fyp-alaknanda
//   2. Paste this whole file in
//   3. Click Run
//   4. Read the three text blocks in the Console
//   5. Fill in S1_WINTER_EXPORT and S2_DRY_EXPORT
//   6. Run again, then press RUN on each task in the Tasks tab
//   7. Download from Drive / FYP_exports into
//        05 Data\sentinel1\    05 Data\sentinel2\    05 Data\rainfall\
// ===========================================================================

// --- corridor, identical to 01_export_s1.js --------------------------------
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


// ===========================================================================
// SECTION A. Sentinel-1 winter pre-image, 2021 event, orbit 63 descending
// ===========================================================================

var S1_WINTER_START = '2020-12-01';
var S1_WINTER_END   = '2021-03-31';

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(corridor)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('relativeOrbitNumber_start', 63))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

function s1Row(dateStr) {
  var d0  = ee.Date(dateStr);
  var day = s1.filterDate(d0, d0.advance(1, 'day'));

  // Coverage is what matters here. A scene that reaches 60 per cent of the
  // corridor is not a pre-image, it is half a pre-image, and the missing half
  // would silently become nodata in the change map.
  var cov = day.select('VV').mosaic().mask().unmask(0).clip(corridor)
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: corridor,
                   scale: 100, maxPixels: 1e9, bestEffort: true})
    .get('VV');

  return ee.String(dateStr)
    .cat('   coverage ').cat(ee.Number(cov).multiply(100).format('%6.1f'))
    .cat('   scenes ').cat(day.size().format('%2.0f'));
}

var s1Dates = s1.filterDate(S1_WINTER_START, S1_WINTER_END)
  .aggregate_array('system:time_start')
  .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); })
  .distinct().sort();

print(ee.String('=== A. Sentinel-1 orbit 63 DESCENDING, ' + S1_WINTER_START +
                ' to ' + S1_WINTER_END + ' ===\n')
        .cat('Pick the one with coverage near 100 and a date in Jan or Feb.\n')
        .cat(ee.List(s1Dates.map(s1Row)).join('\n')));

// FILLED IN 5 September, after the inventory. All eleven winter dates came
// back at 100.0 per cent coverage, so coverage does not choose between them
// and season does.
//
// These three mirror the 2023 pre-images already on disk, date for date:
//   2022-12-19 -> 2020-12-25     2023-01-12 -> 2021-01-18
//   2023-02-05 -> 2021-02-11
//
// Three rather than one, because the 2 September note says to compare the
// winter alternatives before settling on a baseline, and that comparison is
// only meaningful if both events offer the same choice. Picking one date for
// 2021 and then comparing it against three for 2023 would build the asymmetry
// into the result. If disk or patience is short, 2021-01-18 alone is the
// minimum, being the direct analogue of the January scene you already have.
var S1_WINTER_EXPORT = [
  {tag: '2021_pre_alt_dec', date: '2020-12-25'},
  {tag: '2021_pre_alt_jan', date: '2021-01-18'},
  {tag: '2021_pre_alt_feb', date: '2021-02-11'}
];

S1_WINTER_EXPORT.forEach(function (s) {
  var d0 = ee.Date(s.date);
  // 'angle' is the per-pixel incidence angle. 03_terrain_strata.py needs it,
  // and falls back to a constant 38.5 degrees without it.
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


// ===========================================================================
// SECTION B. Sentinel-2 dry season, one per event year
// ===========================================================================
//
// The windows below match the Sentinel-1 pre-image windows already on disk,
// deliberately. If the permanent channel is drawn from the same season as the
// SAR baseline, then the change map and the reference are describing the same
// "normal" river, and the subtraction means something.

// AMENDED 5 September, after the event-window inventory came back.
// 2021 turned out to have a cloud-free scene on 16 October, five days before
// the flood and in the same season. That is a better baseline for the 2021
// event than any winter scene, and it is already being exported by script 02.
// The 2021 winter window is kept here anyway, because the two baselines answer
// different questions and the difference between them is itself a result:
// seasonal baseline gives event-specific inundation, dry-season baseline gives
// water above the dry channel, and the SAR change map uses a dry-season
// pre-image. 2023 has no clear pre-flood scene at all, so for 2023 the winter
// baseline is the only option and this is not a choice.

var S2_DRY = [
  {tag: '2023_dry', start: '2022-11-01', end: '2023-02-28'},
  {tag: '2021_dry', start: '2020-11-01', end: '2021-02-28'}
];

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');

// SCL: 0 nodata, 3 cloud shadow, 8 cloud medium, 9 cloud high, 10 cirrus,
// 11 snow and ice. Snow is counted separately here because these are winter
// windows and the upper corridor can carry snow on the valley sides, which
// would ruin a digitising session without ever showing up as cloud.
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

S2_DRY.forEach(function (w) {
  var dates = s2.filterBounds(corridor).filterDate(w.start, w.end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .aggregate_array('system:time_start')
    .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); })
    .distinct().sort();

  print(ee.String('=== B. Sentinel-2 dry season, ' + w.tag + ', ' +
                  w.start + ' to ' + w.end + ' ===\n')
          .cat('Pre-filtered to tile cloud under 60 per cent. ' +
               'Pick the highest CLEAR with low snow.\n')
          .cat('date              cov      CLEAR     cloud      snow\n')
          .cat(ee.List(dates.map(s2Row)).join('\n')));
});

// FILLED IN 5 September. Two dates per year, not one.
//
// The event-window test showed that one Sentinel-2 date reaches 91 to 93 per
// cent of the corridor and two dates two apart reach exactly 100. The dry
// scenes have to be whole for the same reason the event scenes do, so each
// year gets a pair two days apart.
//
// 2023: 3 and 5 February 2023, CLEAR 94.3 and 91.4, no cloud, no snow.
//   5 February is the SAME DAY as S1_2023_pre_alt_feb_20230205_orb129, which
//   is already on your disk. Optical and radar baseline from one day is as
//   tight as this gets, and it is luck rather than planning.
//
// 2021: 8 and 10 February 2021, CLEAR 90.7 and 91.3, no cloud, no snow.
//   Two and four days before 2021-02-11, the Sentinel-1 winter scene selected
//   above, so the same pairing holds for 2021.
//
// February over January for both, because February is the driest point of the
// year here and the channel is at its narrowest, which is what a permanent
// channel polygon should be drawn from.
var S2_DRY_EXPORT = [
  {tag: '2023_dry_a', date: '2023-02-03'},
  {tag: '2023_dry_b', date: '2023-02-05'},
  {tag: '2021_dry_a', date: '2021-02-08'},
  {tag: '2021_dry_b', date: '2021-02-10'}
];

// Same coverage test as the event scenes, on the two dry pairs. Both should
// print close to 100. If either does not, the pair is two views of the same
// relative orbit rather than two different ones, and the second date has to
// move by a day or two.
[['2023 dry', '2023-02-03', '2023-02-05'],
 ['2021 dry', '2021-02-08', '2021-02-10']].forEach(function (g) {
  function cov(dateStr) {
    var d0 = ee.Date(dateStr);
    return s2.filterBounds(corridor).filterDate(d0, d0.advance(1, 'day'))
             .select('SCL').mosaic().unmask(0).clip(corridor).gt(0);
  }
  function pctOf(img) {
    return ee.Number(img.toFloat().reduceRegion({
      reducer: ee.Reducer.mean(), geometry: corridor,
      scale: 100, maxPixels: 1e9, bestEffort: true
    }).values().get(0)).multiply(100).format('%6.2f');
  }
  print(ee.String('=== coverage, ' + g[0] + ' pair ===\n')
    .cat(g[1] + ' alone      ').cat(pctOf(cov(g[1]))).cat('\n')
    .cat(g[2] + ' alone      ').cat(pctOf(cov(g[2]))).cat('\n')
    .cat('both mosaicked         ').cat(pctOf(cov(g[1]).or(cov(g[2])))));
});

S2_DRY_EXPORT.forEach(function (s) {
  var d0 = ee.Date(s.date);
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


// ===========================================================================
// SECTION C. CHIRPS daily rainfall, corridor mean, as a CSV
// ===========================================================================
//
// Windows run well before each event, not just around it. Antecedent rainfall
// is the part that makes a flood plausible: the day of the peak tells you
// nothing on its own about whether the catchment was already saturated.
//
// This one needs no dates filled in. Two tasks queue on the first run.

var RAIN = [
  {tag: '2023', start: '2023-07-01', end: '2023-08-31'},
  {tag: '2021', start: '2021-09-01', end: '2021-10-31'}
];

var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');

RAIN.forEach(function (w) {
  var rows = chirps.filterDate(w.start, w.end).map(function (img) {
    var mm = img.reduceRegion({
      reducer: ee.Reducer.mean(), geometry: corridor,
      scale: 5000, maxPixels: 1e9, bestEffort: true
    }).get('precipitation');
    return ee.Feature(null, {
      date: img.date().format('YYYY-MM-dd'),
      rain_mm: mm
    });
  });

  Export.table.toDrive({
    collection: ee.FeatureCollection(rows),
    description: 'RAIN_CHIRPS_' + w.tag,
    folder: DRIVE,
    fileNamePrefix: 'RAIN_CHIRPS_corridor_daily_' + w.tag,
    fileFormat: 'CSV',
    selectors: ['date', 'rain_mm']
  });
});

// This line used to print unconditionally and said the export lists were still
// empty, which was wrong once they were filled. It counts now.
print('Tasks queued: ' + (S1_WINTER_EXPORT.length) + ' Sentinel-1, ' +
      (S2_DRY_EXPORT.length * 2) + ' Sentinel-2, ' + (RAIN.length) +
      ' rainfall CSV. Open the Tasks tab and press RUN on each.');

// The two rainfall CSVs queue on every run of this script, so if you have run
// it twice you will have queued them twice. Harmless, they are identical and
// about 2 KB each. Cancel the duplicates or let them land.
