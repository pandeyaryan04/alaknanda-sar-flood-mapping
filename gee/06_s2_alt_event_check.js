// ===========================================================================
// 06_s2_alt_event_check.js
// Is there a usable Sentinel-2 scene after the July and June 2023 rainfall
// peaks? Inventory only, nothing exports.
// Written 5 September 2026.
//
// WHY
// ---
// The Sentinel-1 timing inventory came back and it changed the question.
// Only two relative orbits cover this corridor, 63 descending at 00:44 UTC and
// 129 ascending at 12:47 UTC, and both give 100 per cent coverage. So the
// choice of acquisition is not free: it is whatever those two orbits happen to
// offer near a rainfall peak.
//
// For the two events already chosen, they are the best available. Nothing sits
// closer to the peak than what is on disk. The three-day lag is imposed by the
// revisit, not by a decision, and that is a limitation to state rather than a
// mistake to fix.
//
// But 19 July 2023 looks like a better event than 16 August 2023 on every
// hydrological measure:
//
//                              19 Jul 2023      16 Aug 2023
//   antecedent 7-day rain        150.3 mm          52.1 mm
//   antecedent 30-day rain       393.1 mm         208.9 mm
//   largest day in that week      66.7 mm mean      36.3 mm mean
//                                322.0 mm max       81.8 mm max
//   rain on T-1 and T-2          11.5, 18.7 mm     0.0, 0.2 mm
//   orbit                        63 DESCENDING    129 ASCENDING
//
// The last row matters as much as the rest. 2021 is on orbit 63 descending. A
// 2023 scene on the same orbit would let the two events be compared without
// look geometry being confounded with everything else.
//
// The catch is cloud. 19 July is deeper into the monsoon than 19 August, and
// the reference extent has to be drawn from optical imagery. If there is no
// clear Sentinel-2 scene near 19 July, that event cannot be validated, and an
// event with no reference cannot carry an accuracy table.
//
// This script settles that in about two minutes.
//
// HOW TO USE
//   1. https://code.earthengine.google.com, project gbpuat-fyp-alaknanda
//   2. New script, paste this whole file in
//   3. Click Run
//   4. Copy the two text blocks out of the Console and keep them
// ===========================================================================

var pts = [
  [78.9811, 30.2844], [78.9000, 30.2400], [78.7833, 30.2222], [78.6800, 30.1900],
  [78.5981, 30.1469], [78.4300, 30.1300], [78.2676, 30.0869]
];
var corridor = ee.Geometry.LineString(pts).buffer(5000);
Map.centerObject(corridor, 9);
Map.addLayer(corridor, {color: 'yellow'}, 'corridor');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');

// The two candidate Sentinel-1 acquisitions, with the same +/- 15 day window
// used for the August and October checks so the numbers are comparable.
var CANDIDATES = [
  {tag: '19 Jul 2023, orbit 63 DESC, after the 322 mm day', sar: '2023-07-19'},
  {tag: '29 Jun 2023, orbit 129 ASC, after the 25 Jun storm', sar: '2023-06-29'}
];
var WINDOW_DAYS = 15;

function frac(img) {
  return ee.Number(img.toFloat().reduceRegion({
    reducer: ee.Reducer.mean(), geometry: corridor,
    scale: 100, maxPixels: 1e9, bestEffort: true
  }).values().get(0)).multiply(100);
}

var HEADER = 'date          lag      cov     CLEAR     cloud      snow  ' +
             '   tile   scenes';

CANDIDATES.forEach(function (ev) {
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
      .cat('   cov ').cat(frac(covered).format('%6.1f'))
      .cat('   CLEAR ').cat(frac(clear).format('%6.1f'))
      .cat('   cloud ').cat(frac(cloud).format('%6.1f'))
      .cat('   snow ').cat(frac(snow).format('%6.1f'))
      .cat('   tile ').cat(ee.Number(
            day.aggregate_mean('CLOUDY_PIXEL_PERCENTAGE')).format('%6.1f'))
      .cat('   scenes ').cat(day.size().format('%2.0f'));
  });

  print(ee.String('=== ' + ev.tag + ', Sentinel-1 on ' + ev.sar + ' ===\n')
          .cat(HEADER).cat('\n').cat(ee.List(rows).join('\n')));
});

// ---------------------------------------------------------------------------
// THE BAR TO BEAT
//
// 19 August 2023, the scene already exported for the August event, is
// 86.5 per cent clear at a lag of +3 days from its Sentinel-1 acquisition.
//
// For a July event to be worth switching to, something in that first block has
// to reach roughly 80 per cent clear within about four days of 19 July, and
// there has to be a second date two days away to mosaic with, because one date
// only ever covers 91 to 93 per cent of the corridor.
//
// If nothing does, the answer is that 16 August stays the validated 2023 event
// and July becomes at most an extra unvalidated scene. That is a finding, and
// "no cloud-free optical imagery existed within a week of the largest rainfall
// event of the monsoon" is a sentence worth having in the paper.
// ---------------------------------------------------------------------------

print('Inventory only. Nothing exports from this script.');
