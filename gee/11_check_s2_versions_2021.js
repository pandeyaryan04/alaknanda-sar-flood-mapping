// ===========================================================================
// 11_check_s2_versions_2021.js
// Lists every Sentinel-2 image Earth Engine holds over the corridor on four
// 2021 dates, with its processing version. Written 2 October 2026.
// It exports nothing and queues no tasks.
//
// Why: the 1 October re-export of 10 February and 23 October 2021 does not
// match the files exported in September, while 8 February and 21 October do.
// If a date now has the same tile twice with two processing baselines (for
// example 02.xx and 05.xx), Earth Engine added a reprocessed version and the
// mosaic picked it. That explains the change and settles what to write in
// the methods.
//
// HOW TO USE
//   1. https://code.earthengine.google.com, project gbpuat-fyp-alaknanda
//   2. NEW, File, name it 11_check_s2_versions_2021, paste this whole file
//   3. Save, then Run
//   4. Copy everything in the Console and keep it
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
    return ee.String(img.get('system:index'))
      .cat('   tile ').cat(ee.String(img.get('MGRS_TILE')))
      .cat('   baseline ').cat(ee.String(img.get('PROCESSING_BASELINE')))
      .cat('   generated ').cat(ee.Date(img.get('GENERATION_TIME')).format('YYYY-MM-dd'))
      .cat('   cloud % ').cat(ee.Number(img.get('CLOUDY_PIXEL_PERCENTAGE')).format('%.1f'));
  });
  print(ee.String(d + '   images: ').cat(c.size().format('%d')).cat('\n')
          .cat(ee.List(lines).join('\n')));
});
