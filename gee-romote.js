GEE获取影像javascript代码：
// 用户只需修改这两个变量 ↓
var targetYear = 2020;  // 目标年份（整数）
var targetMonth = 12;    // 目标月份（1-12整数）
  
// 生成时间标识（用于文件名）
var timeStamp = targetYear + '-' + targetMonth;
  
/************ 地理参数设置 ************/
// 坐标转换（36°40'48"N, 101°46'19"E）
var latitude = 36 + 40/60 + 48/3600;
var longitude = 101 + 46/60 + 19/3600;
  
// 创建研究区域
var point = ee.Geometry.Point([longitude, latitude]);
var buffer = point.buffer(2000); // 500米缓冲区
var rect = buffer.bounds();
  
// 地图显示设置
Map.centerObject(point, 12);
Map.addLayer(point, {color: 'red'}, 'Center Point');
Map.addLayer(rect, {color: 'blue', opacity: 0.2}, 'Study Area');
  
/************ 数据预处理 ************/
// Sentinel-2数据筛选
var s2COL = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(rect)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
           .filter(ee.Filter.calendarRange(targetYear, targetYear, 'year'))
           .filter(ee.Filter.calendarRange(targetMonth, targetMonth, 'month'));
  
// 去云函数
function maskS2sr(image) {
  var cloudProb = image.select('MSK_CLDPRB');
  var cloudMask = cloudProb.lt(30);
  var opticalBands = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
                          .multiply(0.0001)
                          .rename(['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']);
  return image.addBands(opticalBands, null, true).updateMask(cloudMask);
}
  
// 归一化函数
function norm_img(image) {
  var minmax = image.reduceRegion({
    reducer: ee.Reducer.minMax(),
    geometry: rect,
    scale: 10,
    maxPixels: 1e13
  }).values();
  return image.unitScale(minmax.get(1), minmax.get(0));
}
  
/************ 指标计算 ************/
function GetIMG() { // 移除了年份参数
  var s2 = s2COL
             .map(maskS2sr)
             .median()
             .clip(rect);
  
  var waterMask = s2.normalizedDifference(['Green', 'SWIR1']).lt(0.2);
    
  // NDVI计算
  var rawNDVI = s2.normalizedDifference(['NIR', 'Red']).rename('ndvi');
  var ndvi = norm_img(rawNDVI);
    
  // WET指数
  var rawWET = s2.expression(
    'B*0.3510 + G*0.3813 + R*0.3437 + NIR*0.7196 + SWIR1*0.2396 + SWIR2*0.1949', {  
      'B': s2.select('Blue'), 
      'G': s2.select('Green'),
      'R': s2.select('Red'),
      'NIR': s2.select('NIR'),
      'SWIR1': s2.select('SWIR1'),
      'SWIR2': s2.select('SWIR2')
  }).rename('wet');
  var wet = norm_img(rawWET);
    
  // NDBSI计算
  var ibi = s2.expression(
    '(2 * SWIR1 / (SWIR1 + NIR) - (NIR / (NIR + R) + G / (G + SWIR1))) / ' +
    '(2 * SWIR1 / (SWIR1 + NIR) + (NIR / (NIR + R) + G / (G + SWIR1)))', {  
      'SWIR1': s2.select('SWIR1'),
      'NIR': s2.select('NIR'),
      'R': s2.select('Red'),
      'G': s2.select('Green')
  }).rename('ibi');
    
  var si = s2.expression(
    '((SWIR1 + R) - (NIR + Blue)) / ((SWIR1 + R) + (NIR + Blue))', {
      'SWIR1': s2.select('SWIR1'),
      'NIR': s2.select('NIR'),
      'R': s2.select('Red'),
      'Blue': s2.select('Blue')
  }).rename('si');
    
  var rawNDBSI = ((ibi.add(si)).divide(2)).rename('ndbsi');
  var ndbsi = norm_img(rawNDBSI);
  
  return ee.Image([ndvi, wet, ndbsi]).updateMask(waterMask);
}
  
var indexImage = GetIMG(); // 直接获取当前时间影像
  
/************ PCA分析 ************/
function pca_model(image){
  var scale = 10;
  var bandNames = image.bandNames();
  var meanDict = image.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: rect,
        scale: scale,
        maxPixels: 1e13});
  var means = ee.Image.constant(meanDict.values(bandNames));
  var centered = image.subtract(means);
    
  var getNewBandNames = function(prefix) {
    return ee.List.sequence(1, bandNames.length()).map(function(b) {
      return ee.String(prefix).cat(ee.Number(b).int());
    })};
    
  var arrays = centered.toArray();
  var covar = arrays.reduceRegion({
    reducer: ee.Reducer.centeredCovariance(),
    geometry: rect,
    scale: scale,
    maxPixels: 1e13
  });
    
  var covarArray = ee.Array(covar.get('array'));
  var eigens = covarArray.eigen();
  var eigenValues = eigens.slice(1, 0, 1);
  var eigenVectors = eigens.slice(1, 1);
    
  var arrayImage = arrays.toArray(1);
  var principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage);
    
  var sdImage = ee.Image(eigenValues.sqrt())
    .arrayProject([0]).arrayFlatten([getNewBandNames('sd')]);
    
  return principalComponents
    .arrayProject([0])
    .arrayFlatten([getNewBandNames('pc')])
    .divide(sdImage);
}
  
var pca_result = pca_model(indexImage).select('pc1').rename('RSEI');
pca_result = norm_img(pca_result);
  
/************ 数据导出 ************/
// RSEI导出
Export.image.toDrive({
   image: pca_result,
   description: 'RSEI_' + timeStamp,
   scale: 10,
   region: rect,
   maxPixels: 1e13,
   crs:'EPSG:4326',
   fileFormat: 'GeoTIFF'
});
  
// 各指标单独导出
var exportBands = ['ndvi', 'wet', 'ndbsi'];
exportBands.forEach(function(band){
  Export.image.toDrive({
    image: indexImage.select(band),
    description: band.toUpperCase() + '_' + timeStamp,
    scale: 10,
    region: rect,
    maxPixels: 1e13,
    crs: 'EPSG:4326',
    fileFormat: 'GeoTIFF'
  });
});
  
/************ 可视化 ************/
var palette = ["d73027","f46d43","fdae61","fee08b","d9ef8b","a6d96a","66bd63","1a9850"];
Map.addLayer(pca_result, {min:0, max:1, palette: palette}, 'RSEI ' + timeStamp);
Map.addLayer(indexImage.select('ndvi'), {min:0, max:1, palette: palette}, 'NDVI');
Map.addLayer(indexImage.select('wet'), {min:0, max:1, palette: palette}, 'WET');
Map.addLayer(indexImage.select('ndbsi'), {min:0, max:1, palette: palette}, 'NDBSI');
