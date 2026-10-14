// ===========================================================================
// 02_s2_reference_inventory.js
// Sentinel-2 availability over the Alaknanda corridor, for the reference
// flood extent.
//
// VERSION 3, 5 September 2026. The inventory has been run and read. This
// version keeps the inventory (re-run it any time) and adds two things:
//
//   SECTION B, a coverage test. Every single date in both tables came back at
//   91 to 93 per cent corridor coverage, never 100. That is not cloud, cloud
//   is counted separately. It is the Sentinel-2 footprint not reaching about
//   8 per cent of the corridor on any one date. This section finds out whether
//   mosaicking two dates closes the gap, and shows you where the gap is.
//
//   SECTION C, the export list, filled in with the five scenes chosen off the
//   tables. Reasoning is written above each one. Change them if you disagree.
//
// HOW TO USE
//   1. https://code.earthengine.google.com, project gbpuat-fyp-alaknanda
//   2. Paste this whole file in
//   3. Click Run
//   4. Read the Section B numbers in the Console
//   5. Open the Tasks tab and press RUN on each task
//   6. Download from Drive / FYP_exports into 05 Data\sentinel2\
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
var centreline = ee.Geometry.LineString(pts);
var corridor   = centreline.buffer(5000);

var CRS   = 'EPSG:32644';
var DRIVE = 'FYP_exports';

Map.centerObject(corridor, 9);
Map.addLayer(corridor, {color: 'yellow'}, 'corridor (5 km buffer)');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');

// SCL classes. 0 nodata, 3 cloud shadow, 8 cloud medium, 9 cloud high,
// 10 cirrus, 11 snow and ice.
function sclOf(dateStr) {
  var d0 = ee.Date(dateStr);
  return s2.filterBounds(corridor).filterDate(d0, d0.advance(1, 'day'))
           .select('SCL').mosaic().unmask(0).clip(corridor);
}
function fracOf(img, geom) {
  return ee.Number(img.toFloat().reduceRegion({
    reducer: ee.Reducer.mean(), geometry: geom || corridor,
    scale: 100, maxPixels: 1e9, bestEffort: true
  }).values().get(0)).multiply(100);
}


// ===========================================================================
// SECTION A. The inventory. Already run; kept so it can be re-run.
// Set RUN_INVENTORY to true to print the two tables again.
// ===========================================================================

var RUN_INVENTORY = false;

var WINDOW_DAYS = 15;
var EVENTS = [
  {tag: '2023', sar: '2023-08-16'},
  {tag: '2021', sar: '2021-10-21'}
];

if (RUN_INVENTORY) {
  var HEADER = 'date          lag      cov     CLEAR     cloud      snow  ' +
               '   tile   scenes';
  EVENTS.forEach(function (ev) {
    var start = ee.Date(ev.sar).advance(-WINDOW_DAYS, 'day');
    var end   = ee.Date(ev.sar).advance(WINDOW_DAYS + 1, 'day');
    var col   = s2.filterBounds(corridor).filterDate(start, end);
    var dates = col.aggregate_array('system:time_start')
      .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); })
      .distinct().sort();

    var rows = dates.map(function (d) {
      var ds  = ee.String(d);
      var d0  = ee.Date(ds);
      var day = s2.filterBounds(corridor).filterDate(d0, d0.advance(1, 'day'));
      var scl = day.select('SCL').mosaic().unmask(0).clip(corridor);
      var covered = scl.gt(0);
      var cloud   = scl.eq(3).or(scl.gte(8).and(scl.lte(10)));
      var snow    = scl.eq(11);
      var clear   = covered.and(cloud.not()).and(snow.not());
      return ds
        .cat('  lag ').cat(d0.difference(ee.Date(ev.sar), 'day').format('%5.0f'))
        .cat('   cov ').cat(fracOf(covered).format('%6.1f'))
        .cat('   CLEAR ').cat(fracOf(clear).format('%6.1f'))
        .cat('   cloud ').cat(fracOf(cloud).format('%6.1f'))
        .cat('   snow ').cat(fracOf(snow).format('%6.1f'))
        .cat('   tile ').cat(ee.Number(
              day.aggregate_mean('CLOUDY_PIXEL_PERCENTAGE')).format('%6.1f'))
        .cat('   scenes ').cat(day.size().format('%2.0f'));
    });

    print(ee.String('=== ' + ev.tag + ' event, Sentinel-1 on ' + ev.sar +
                    ', +/- ' + WINDOW_DAYS + ' days ===\n')
            .cat(HEADER).cat('\n').cat(ee.List(rows).join('\n')));
  });
}


// ===========================================================================
// SECTION B. The coverage gap
// ===========================================================================
//
// Every date came back at 91 to 93 per cent coverage. Same figure in both
// event windows, two years apart, which rules out weather and points at
// geometry: one Sentinel-2 relative orbit does not reach the whole corridor,
// and the missing strip is in the same place every time.
//
// This matters more than it looks. If 8 per cent of the corridor has no
// optical image on the reference date, then 8 per cent of the corridor has no
// reference, and any accuracy figure quoted for the whole corridor is really a
// figure for 92 per cent of it with the rest quietly dropped.
//
// Two things to find out, and both are cheap:
//   1. does mosaicking two nearby dates close the gap
//   2. where is the gap, so you can say which reach is affected
//
// If two dates close it, the fix is free and the reference is whole. If they
// do not, the honest move is to clip the corridor to the covered area and say
// so in the methods, giving the real study area rather than the nominal one.

var GAP_TEST = [
  {tag: '2021 event', dates: ['2021-10-21', '2021-10-23']},
  {tag: '2023 event', dates: ['2023-08-19', '2023-08-17']}
];

GAP_TEST.forEach(function (g) {
  var a = sclOf(g.dates[0]).gt(0);
  var b = sclOf(g.dates[1]).gt(0);
  var both = a.or(b);

  print(ee.String('=== B. coverage, ' + g.tag + ' ===\n')
    .cat(g.dates[0] + ' alone      ').cat(fracOf(a).format('%6.2f')).cat('\n')
    .cat(g.dates[1] + ' alone      ').cat(fracOf(b).format('%6.2f')).cat('\n')
    .cat('both mosaicked         ').cat(fracOf(both).format('%6.2f')));

  // Red is corridor with no Sentinel-2 on either date. Turn the layer on and
  // look at where it sits along the reach.
  Map.addLayer(both.not().selfMask(), {palette: ['red']},
               'no coverage, ' + g.tag, false);
});

// Also worth knowing: how much of the corridor the JRC permanent water layer
// says is always wet. If the uncovered strip happens to be mostly dry hillside
// it matters much less than if it sits on the channel.
Map.addLayer(ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
               .clip(corridor).gt(50).selfMask(),
             {palette: ['blue']}, 'JRC permanent water', false);


// ===========================================================================
// SECTION C. The exports
// ===========================================================================
//
// Five scenes, chosen off the tables. Reasoning per scene:
//
// 2023-08-19, lag +3, CLEAR 86.5
//   The only usable scene anywhere near the 2023 event. Three days after the
//   Sentinel-1 pass, so the flood has been receding for three days and this
//   reference will under-report extent. That is a stated limitation, not a
//   hidden one.
//
// 2023-08-17, lag +1, CLEAR 47.2
//   Half the corridor is cloudy, so it cannot be the reference. It is worth
//   having anyway, because where both scenes are clear you can measure how
//   much the water shrank between day +1 and day +3. That turns the lag from
//   an apology into a number.
//
// 2021-10-21, lag 0, CLEAR 81.9
//   Same day as the Sentinel-1 acquisition. Zero lag. This is the best
//   reference scene available for either event and it is the one to build the
//   2021 extent from.
//
// 2021-10-23, lag +2, CLEAR 91.6
//   Fills the 9.3 per cent of the corridor that was cloudy on the 21st, and
//   is the second date in the coverage test above.
//
// 2021-10-16, lag -5, CLEAR 91.6
//   Cloud-free, five days BEFORE the flood, same season. This is a pre-flood
//   optical image, which the 2023 event does not have and which changes what
//   is possible. See the note in the decision file.

var EXPORT_DATES = [
  {tag: '2023_ref',     date: '2023-08-19'},
  {tag: '2023_ref_alt', date: '2023-08-17'},
  {tag: '2021_ref',     date: '2021-10-21'},
  {tag: '2021_ref_alt', date: '2021-10-23'},
  {tag: '2021_pre',     date: '2021-10-16'}
];

EXPORT_DATES.forEach(function (s) {
  var d0  = ee.Date(s.date);
  var img = s2.filterBounds(corridor)
              .filterDate(d0, d0.advance(1, 'day'))
              .mosaic().clip(corridor);

  var stamp = s.date.replace(/-/g, '');

  // B2 B3 B4 true colour, B8 near infrared, B11 shortwave infrared for MNDWI,
  // SCL so you can see on screen which pixels the classifier called cloud
  // while you are drawing.
  var stack = img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'SCL']).toInt16();

  Export.image.toDrive({
    image: stack,
    description: 'S2_' + s.tag + '_' + stamp,
    folder: DRIVE,
    fileNamePrefix: 'S2_' + s.tag + '_' + stamp + '_10m',
    region: corridor, scale: 10, crs: CRS,
    maxPixels: 1e10, fileFormat: 'GeoTIFF'
  });

  // Indices precomputed so there is no raster calculator step in QGIS.
  // MNDWI, green against shortwave infrared, is the one to digitise against.
  // NDWI, green against near infrared, is a second opinion. The two disagree
  // over wet sand and over shadow, and where they disagree is exactly where
  // your own judgement is doing the work.
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

  Map.addLayer(img, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000},
               'true colour ' + s.tag, false);
  Map.addLayer(mndwi, {min: -0.5, max: 0.5}, 'MNDWI ' + s.tag, false);
});

print('Ten export tasks queued, two per scene. Roughly 800 MB in total. ' +
      'Open the Tasks tab and press RUN on each.');
