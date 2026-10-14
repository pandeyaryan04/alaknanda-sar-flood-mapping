// ===========================================================================
// 08_export_context_layers.js
// Population and administrative boundaries, clipped to the corridor.
// Written 5 September 2026.
//
// WHY THIS INSTEAD OF THE PORTALS
// -------------------------------
// Both sources were checked on 5 September 2026.
//
// WorldPop publishes India as a single national GeoTIFF at 100 m:
//   https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/
//   ind_ppp_2020_UNadj.tif   1.5 GB
// That is a 1.5 GB download to use 808 km2 of it. The same data is in Earth
// Engine as WorldPop/GP/100m/pop, so it can be clipped before it is downloaded
// and arrives as a file measured in hundreds of KB.
//
// FAO GAUL level 2 gives district boundaries inside Earth Engine, again
// clipped before export. GADM is the better product for the report and its
// direct links are in the data file, but for anything computational this is
// the version that does not waste an evening.
//
// HOW TO USE
//   1. https://code.earthengine.google.com, project gbpuat-fyp-alaknanda
//   2. New script, paste this whole file in
//   3. Click Run
//   4. Read the printed district list and population total, they are the check
//   5. Tasks tab, RUN all four. Everything here is small
//   6. Download into 05 Data\population\ and 05 Data\boundaries\
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

// --- population ------------------------------------------------------------
// WorldPop Global Project, 100 m, people per pixel. 2020 is the most recent
// year in this collection. Sum it over the corridor as a sanity check: the
// reach holds Rudraprayag, Srinagar and Rishikesh, so a few hundred thousand
// is plausible and a few thousand or a few million is not.
// CORRECTED 5 September. The first version clipped this mosaic to the corridor
// and then clipped THAT to the catchment, so the catchment total was really the
// tiny overlap between the two and came back as 7,365 people for 8,638 km2.
// Keep the unclipped mosaic and clip per use.
var popRaw = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filter(ee.Filter.eq('country', 'IND'))
  .filter(ee.Filter.eq('year', 2020))
  .mosaic();

var pop = popRaw.clip(corridor);

var total = pop.reduceRegion({
  reducer: ee.Reducer.sum(), geometry: corridor,
  scale: 100, maxPixels: 1e9, bestEffort: true
}).get('population');

print('WorldPop 2020, people inside the corridor:', total);
Map.addLayer(pop, {min: 0, max: 50, palette: ['white', 'orange', 'red']},
             'population', false);

Export.image.toDrive({
  image: pop.toFloat(),
  description: 'POP_WorldPop_2020_corridor',
  folder: DRIVE,
  fileNamePrefix: 'POP_WorldPop_2020_corridor_100m',
  region: corridor, scale: 100, crs: CRS,
  maxPixels: 1e9, fileFormat: 'GeoTIFF'
});

// Same thing over the upstream catchment, for context in the study area
// section. The catchment is the 8638 km2 polygon derived in script 04.
var hybas = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_7');
var GAUGE = ee.Geometry.Point([78.9842, 30.288061]);
var seed = hybas.filterBounds(GAUGE).aggregate_array('HYBAS_ID');
var upIds = ee.List(ee.List.sequence(1, 12).iterate(function (i, acc) {
  acc = ee.List(acc);
  return acc.cat(hybas.filter(ee.Filter.inList('NEXT_DOWN', acc))
                      .aggregate_array('HYBAS_ID')).distinct();
}, seed));
var catchment = hybas.filter(ee.Filter.inList('HYBAS_ID', upIds))
                     .union(1).geometry();

print('WorldPop 2020, people in the upstream catchment:',
  popRaw.clip(catchment).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: catchment,
    scale: 100, maxPixels: 1e10, bestEffort: true}).get('population'));

// --- administrative boundaries ---------------------------------------------
// FAO GAUL 2015 level 2 is districts. Level 1 is states.
var adm2 = ee.FeatureCollection('FAO/GAUL/2015/level2')
             .filterBounds(corridor);
var adm1 = ee.FeatureCollection('FAO/GAUL/2015/level1')
             .filterBounds(corridor);

print('districts the corridor touches:', adm2.aggregate_array('ADM2_NAME'));
print('states the corridor touches:', adm1.aggregate_array('ADM1_NAME'));

Map.addLayer(adm2, {color: 'blue'}, 'districts', false);

Export.table.toDrive({
  collection: adm2,
  description: 'BOUNDARY_districts_corridor',
  folder: DRIVE,
  fileNamePrefix: 'BOUNDARY_districts_corridor',
  fileFormat: 'GeoJSON'
});

Export.table.toDrive({
  collection: ee.FeatureCollection('FAO/GAUL/2015/level2')
                .filterBounds(catchment),
  description: 'BOUNDARY_districts_catchment',
  folder: DRIVE,
  fileNamePrefix: 'BOUNDARY_districts_catchment',
  fileFormat: 'GeoJSON'
});

// --- land cover, for completeness ------------------------------------------
// You already have ESA WorldCover 10 m. This exports the class histogram
// inside the corridor as a table, which is what actually goes in a report:
// nobody wants the raster, they want the percentages.
// CORRECTED 5 September. The first version counted pixels with bestEffort on,
// which silently coarsens the scale, so multiplying the counts by 100 m2 gave
// 936 km2 for an 808 km2 corridor. Summing ee.Image.pixelArea() per class gives
// the true area whatever scale the reducer settles on.
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().clip(corridor);
var areas = ee.Image.pixelArea().divide(1e6).addBands(wc)
  .reduceRegion({
    reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class'}),
    geometry: corridor, scale: 10, maxPixels: 1e10, bestEffort: true
  });
print('ESA WorldCover area by class, km2, inside the corridor:', areas);
print('  these should sum to about 808 km2. If they do not, say so.');

var hist = wc.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(), geometry: corridor,
  scale: 10, maxPixels: 1e10, bestEffort: true
});
print('ESA WorldCover class pixel counts inside the corridor:', hist);
print('  class codes: 10 tree, 20 shrub, 30 grass, 40 crop, 50 built,',
      '60 bare, 70 snow/ice, 80 water, 90 wetland, 95 mangrove, 100 moss');

Export.table.toDrive({
  collection: ee.FeatureCollection([ee.Feature(null, areas)]),
  description: 'LANDCOVER_area_by_class_corridor',
  folder: DRIVE,
  fileNamePrefix: 'LANDCOVER_area_by_class_corridor',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: ee.FeatureCollection([ee.Feature(null, hist)]),
  description: 'LANDCOVER_class_histogram_corridor',
  folder: DRIVE,
  fileNamePrefix: 'LANDCOVER_class_histogram_corridor',
  fileFormat: 'CSV'
});

print('Five tasks queued. All small. Check the district list and the ' +
      'population total before running them.');
