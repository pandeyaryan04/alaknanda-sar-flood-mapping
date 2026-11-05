// ===========================================================================
// 12_check_s2_ingest_2021.js
// Prints when Earth Engine last wrote each Sentinel-2 image over the corridor
// on four 2021 dates (the asset's system:version). Written 7 October 2026.
// It exports nothing and queues no tasks.
//
// Why: 11_check_s2_versions_2021.js showed one image per tile per
// date, so Earth Engine did not add a second version beside the first. It
// may instead have replaced the original with Copernicus's reprocessed
// version (processing baseline 05.00). If the images for 10 February and
// 23 October 2021 were written after the September export, while 8 February
// and 21 October were not, the replacement is what changed the files.
//
// Caveat: system:version also changes when Earth Engine edits an image's
// properties only, so a recent date is strong evidence, not proof.
//
// HOW TO USE
//   1. https://code.earthengine.google.com, your own Earth Engine cloud project
//   2. NEW, File, name it 12_check_s2_ingest_2021, paste this whole file
//   3. Save, then Run
//   4. Copy everything in the Console into a text file and keep it with the results
// ===========================================================================

var box = ee.Geometry.Rectangle([78.24, 30.04, 79.00, 30.31]);  // corridor v2, with a margin
var DATES = ['2021-02-08', '2021-02-10', '2021-10-21', '2021-10-23'];
// 8 Feb and 21 Oct matched the old files; 10 Feb and 23 Oct did not

DATES.forEach(function (d) {
  var d0 = ee.Date(d);
  var c = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(box).filterDate(d0, d0.advance(1, 'day'));
  var lines = c.toList(50).map(function (img) {
    img = ee.Image(img);
    // system:version is in microseconds since 1970; ee.Date wants milliseconds
    var written = ee.Date(ee.Number(img.get('system:version')).divide(1000));
    return ee.String('tile ').cat(ee.String(img.get('MGRS_TILE')))
      .cat('   baseline ').cat(ee.String(img.get('PROCESSING_BASELINE')))
      .cat('   generated ').cat(ee.Date(img.get('GENERATION_TIME')).format('YYYY-MM-dd'))
      .cat('   written to Earth Engine ').cat(written.format('YYYY-MM-dd HH:mm', 'UTC'))
      .cat(' UTC');
  });
  print(ee.String(d + '   images: ').cat(c.size().format('%d')).cat('\n')
          .cat(ee.List(lines).join('\n')));
});
