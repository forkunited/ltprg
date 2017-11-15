var DIMENSION_TYPE_COLOR = 0;
var DIMENSION_TYPE_SIDECOUNT = 1;
var DIMENSION_TYPE_PARENT = 2;

var OBJECT_HEIGHT = 400;
var OBJECT_WIDTH = 300;
var MIN_SHAPE_LENGTH = 40;
var MAX_SHAPE_LENGTH = 40;
var POSSIBLE_SHAPE_COUNTS = [1,2,3,4];
var POSSIBLE_SIDE_COUNTS = [3,4,5];

var makeRandom = function(trialNum, numRounds) {
  var obj = sampleObject(); // Sample a world state
  var numDimensions = getObjectDimensionCount(obj);
  var numDiffs = Math.floor((trialNum / numRounds)*numDimensions) + 1;
  var diffIndices = sampleSet(numDimensions, numDiffs);
  var otherObj = sampleSecondObject(obj, diffIndices);

  computeObjectGeometry(obj);
  computeObjectGeometry(otherObj);
  centerObject(obj);
  centerObject(otherObj);
  roundObject(obj);
  roundObject(otherObj);

  var speakerFirst = Math.floor(Math.random() * 2);
  var listenerFirst = Math.floor(Math.random() * 2);

  return { objs: [obj, otherObj],
           target : Math.floor(Math.random() * 2),
           speakerOrder : [speakerFirst, 1-speakerFirst],
           listenerOrder : [listenerFirst, 1-listenerFirst],
           diffs : numDiffs }
};

var fromDimensionValueString = function(str, separator) {
  var values = str.split(separator);
  var sTarget = parseInt(values[0]);
  var sOIndex0 = parseInt(values[1]);
  var sOIndex1 = parseInt(values[2]);
  //var lTarget = parseInt(values[3]); // Unnecessary
  var lOIndex0 = parseInt(values[4]);
  var lOIndex1 = parseInt(values[5]);
  var numDiffs = parseInt(values[6]);

  var target = 0;
  if (sTarget == 0)
    target = sOIndex0;
  else
    target = sOIndex1;

  var trial = { objs : [{}, {}],
                target : target,
                speakerOrder : [sOIndex0, sOIndex1],
                listenerOrder : [lOIndex0, lOIndex1],
                diffs : numDiffs };

  var sObj0 = { sideLength : parseFloat(values[14]), shapeCount : parseInt(values[7]), shapes : []};
  var maxShapes = Math.max.apply(null, POSSIBLE_SHAPE_COUNTS);
  var maxSides = Math.max.apply(null, POSSIBLE_SIDE_COUNTS)+1;
  var startIndex0 = 8;
  for (var i = 0; i < maxShapes; i++) {
    var shape = {};
    var startIndexS = startIndex0 + i*(9+maxSides*2);

    var numSides = parseInt(values[startIndexS+5]);
    if (numSides == 0)
      break;

    shape.occupiedSides = [];
    if (i != 0) {
      shape.parent = parseInt(values[startIndexS]);
      shape.parentSide =  parseInt(values[startIndexS+1]);
      shape.occupiedSides.push(0);
      sObj0.shapes[shape.parent].occupiedSides.push(shape.parentSide);
    }
    shape.color = [parseInt(values[startIndexS+2]),
                   parseInt(values[startIndexS+3]),
                   parseInt(values[startIndexS+4])];
    shape.numSides = numSides; // Skip 6 because length
    shape.center = { x : parseFloat(values[startIndexS+7]), y : parseFloat(values[startIndexS+8]) };
    shape.points = [];
    for (var j = 0; j < maxSides; j++) {
      var x = parseFloat(values[startIndexS+9 + j*2]);
      var y = parseFloat(values[startIndexS+9 + j*2+1]);
      if (x == 0.0 && y == 0.0)
        break;
      shape.points.push({ x : x, y : y});
    }

    if (sObj0.shapes.length == 0)
      shape.depth = 0;
    else
      shape.depth = sObj0.shapes[shape.parent].depth + 1;

    sObj0.shapes.push(shape);
  }

  var sObj1 = {  sideLength : parseFloat(values[startIndex0 + maxShapes*(9+maxSides*2) + 7]), shapeCount : parseInt(values[startIndex0 + maxShapes*(9+maxSides*2)]), shapes : [] };
  var startIndex1 = startIndex0 + maxShapes*(9+maxSides*2) + 1;
  for (var i = 0; i < maxShapes; i++) {
    var shape = {};
    var startIndexS = startIndex1 + i*(9+maxSides*2);

    var numSides = parseInt(values[startIndexS+5]);
    if (numSides == 0)
      break;

    shape.occupiedSides = [];
    if (i != 0) {
      shape.parent = parseInt(values[startIndexS]);
      shape.parentSide =  parseInt(values[startIndexS+1]);
      shape.occupiedSides.push(0);
      sObj1.shapes[shape.parent].occupiedSides.push(shape.parentSide);
    }
    shape.color = [parseInt(values[startIndexS+2]),
                   parseInt(values[startIndexS+3]),
                   parseInt(values[startIndexS+4])];
    shape.numSides = numSides; // Skip 6 because length
    shape.center = { x : parseFloat(values[startIndexS+7]), y : parseFloat(values[startIndexS+8]) };
    shape.points = [];
    for (var j = 0; j < maxSides; j++) {
      var x = parseFloat(values[startIndexS+9 + j*2]);
      var y = parseFloat(values[startIndexS+9 + j*2+1]);
      if (x == 0.0 && y == 0.0)
        break;
      shape.points.push({ x : x, y : y});
    }

    if (sObj1.shapes.length == 0)
      shape.depth = 0;
    else
      shape.depth = sObj1.shapes[shape.parent].depth + 1;

    sObj1.shapes.push(shape);
  }

  trial.objs[sOIndex0] = sObj0;
  trial.objs[sOIndex1] = sObj1;

  return trial;
};

var fromFlatObject = function(flat) {
  var names = getDimensionNames();
  var flatStr = "";
  for (var i = 0; i < names.length; i++) {
    flatStr += flat[names[i]] + ",";
  }

  flatStr = flatStr.substring(0, flatStr.length - 1);
  
  return fromDimensionValueString(flatStr, ",");
};

var getFlatObject = function(trial) {
  var obj = {};
  var names = getDimensionNames();
  for (var i = 0; i < names.length; i++) {
    obj[names[i]] = getDimensionValue(trial, names[i]);
  }
  return obj;
};

var getDimensionValuesString = function(trial, separator) {
  return getDimensionValues(trial).join(separator);
};

var getDimensionValues = function(trial) {
  var values = [];
  var names = getDimensionNames();
  for (var i = 0; i < names.length; i++) {
    values.push(getDimensionValue(trial, names[i]));
  }
  return values;
};

var getDimensionValue = function(trial, dimName) {
  var nameParts = getNameParts(dimName);
  if (dimName == "sTarget") {
    return trial.speakerOrder.indexOf(trial.target);
  } else if (dimName == "sOIndex0") {
    return trial.speakerOrder[0];
  } else if (dimName == "sOIndex1") {
    return trial.speakerOrder[1];
  } else if (dimName == "lTarget") {
    return trial.listenerOrder.indexOf(trial.target);
  } else if (dimName == "lOIndex0") {
    return trial.listenerOrder[0];
  } else if (dimName == "lOIndex1") {
    return trial.listenerOrder[1];
  } else if (dimName == "diffs") {
    return trial.diffs;
  } else if (dimName.indexOf("Shapes") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    return obj.shapeCount;
  } else if (dimName.indexOf("PrntSide") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    if (s >= obj.shapes.length || s == 0)
      return 0;
    return obj.shapes[s].parentSide;
  } else if (dimName.indexOf("Prnt" || s == 0) >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    if (s >= obj.shapes.length)
      return 0;
    return obj.shapes[s].parent;
  } else if (dimName.indexOf("ClrH") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    if (s >= obj.shapes.length)
      return 0;
    return obj.shapes[s].color[0];
  } else if (dimName.indexOf("ClrS") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    if (s >= obj.shapes.length)
      return 0;
    return obj.shapes[s].color[1];
  } else if (dimName.indexOf("ClrL") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    if (s >= obj.shapes.length)
      return 0;
    return obj.shapes[s].color[2];
  } else if (dimName.indexOf("Sides") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    if (s >= obj.shapes.length)
      return 0;
    return obj.shapes[s].numSides;
  } else if (dimName.indexOf("Length") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    if (s >= obj.shapes.length)
      return 0;
    return obj.sideLength;
  } else if (dimName.indexOf("CenterX") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    if (s >= obj.shapes.length)
      return 0;
    return obj.shapes[s].center.x;
  } else if (dimName.indexOf("CenterY") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    if (s >= obj.shapes.length)
      return 0;
    return obj.shapes[s].center.y;
  } else if (dimName.indexOf("Pnt") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    if (s >= obj.shapes.length)
      return 0;
    var p = nameParts["Pnt"];
    if (p >= obj.shapes[s].points.length)
      return 0;
    if (dimName.endsWith("X"))
      return obj.shapes[s].points[p].x;
    else
      return obj.shapes[s].points[p].y;
  }
}

var getNameParts = function(dimName) {
  var nameParts = dimName.split("_");
  var namePartsObj = {};
  for (var i = 0; i < nameParts.length; i++) {
    if (nameParts[i].indexOf("Obj") >= 0) {
      var value = parseInt(nameParts[i].substring(nameParts[i].indexOf("Obj") + 3));
      namePartsObj["Obj"] = value;
    } else if (nameParts[i].indexOf("Shp") >= 0) {
      var value = parseInt(nameParts[i].substring(nameParts[i].indexOf("Shp") + 3));
      namePartsObj["Shp"] = value;
    } else if (nameParts[i].indexOf("Pnt") >= 0) {
      var value = parseInt(nameParts[i].substring(nameParts[i].indexOf("Pnt") + 3));
      namePartsObj["Pnt"] = value;
    }
  }

  namePartsObj["Role"] = nameParts[0].substring(0, 1);

  return namePartsObj;
}

var getObjectByRole = function(trial, objectIndex, role) {
  if (role == "s") {
    return trial.objs[trial.speakerOrder[objectIndex]];
  } else if (role == "l") {
    return trial.objs[trial.listenerOrder[objectIndex]];
  }
};

var getDimensionNamesString = function(separator) {
  return getDimensionNames().join(separator);
};

var getDimensionNames = function() {
  var names = ["sTarget", "sOIndex0", "sOIndex1", "lTarget", "lOIndex0", "lOIndex1", "diffs"];
  names = names.concat(getTrialObjectDimensionNames("s", 0));
  names = names.concat(getTrialObjectDimensionNames("s", 1));
  names = names.concat(getTrialObjectDimensionNames("l", 0));
  names = names.concat(getTrialObjectDimensionNames("l", 1));
  return names;
};

var getTrialObjectDimensionNames = function(role, objectIndex) {
  var names = [];
  var prefix = role + "Obj" + objectIndex;

  names.push(prefix + "_Shapes");

  var maxShapes = Math.max.apply(null, POSSIBLE_SHAPE_COUNTS);
  var maxSides = Math.max.apply(null, POSSIBLE_SIDE_COUNTS)+1;
  for (var i = 0; i < maxShapes; i++) {

    names.push(prefix + "_Shp" + i + "_Prnt");
    names.push(prefix + "_Shp" + i + "_PrntSide");
    names.push(prefix + "_Shp" + i + "_ClrH");
    names.push(prefix + "_Shp" + i + "_ClrS");
    names.push(prefix + "_Shp" + i + "_ClrL");
    names.push(prefix + "_Shp" + i + "_Sides");
    names.push(prefix + "_Shp" + i + "_Length");
    names.push(prefix + "_Shp" + i + "_CenterX");
    names.push(prefix + "_Shp" + i + "_CenterY");
    // Vertex Points
    for (var j = 0; j < maxSides; j++) {
      names.push(prefix + "_Shp" + i + "_Pnt" + j + "_X");
      names.push(prefix + "_Shp" + i + "_Pnt" + j + "_Y");
    }
  }

  return names;
};

var getObjectDimensionCount = function(obj) {
  // shapeCount color dimensions
  // shapeCount side-count dimnensions
  // (shapeCount - 1) parent dimensions (first doesn't have parent)
  return 2*obj.shapeCount + (obj.shapeCount-1);
};

var sampleObject = function() {
  var obj = {};

  obj.sideLength = Math.random()*(MAX_SHAPE_LENGTH-MIN_SHAPE_LENGTH)+MIN_SHAPE_LENGTH;
  obj.shapeCount = POSSIBLE_SHAPE_COUNTS[Math.floor(Math.random() * POSSIBLE_SHAPE_COUNTS.length)];
  obj.shapes = [];

  for (var i = 0; i < obj.shapeCount; i++) {
    var shape = {};

    if (i > 0) {
      /* Find a parent with some unoccupied sides */
      shape.parent = Math.floor(Math.random()*i);
      while (obj.shapes[shape.parent].occupiedSides.length >= obj.shapes[shape.parent].numSides - 1) {
        shape.parent = Math.floor(Math.random()*i);
      }

      shape.depth = obj.shapes[shape.parent].depth + 1;
      shape.occupiedSides = [0]; // Connection to parent at side 0 is occupied

      // Don't add to same side twice
      shape.parentSide = Math.floor(Math.random()*obj.shapes[shape.parent].numSides);
      while (obj.shapes[shape.parent].occupiedSides.indexOf(shape.parentSide) >= 0) {
        shape.parentSide = Math.floor(Math.random()*obj.shapes[shape.parent].numSides);
      }

      obj.shapes[shape.parent].occupiedSides.push(shape.parentSide);
    } else {
      shape.depth = 0;
      shape.occupiedSides = [];
    }

    shape.color = utils.randomColor({ fixedL : true });
    shape.numSides = POSSIBLE_SIDE_COUNTS[Math.floor(Math.random()*POSSIBLE_SIDE_COUNTS.length)];
    obj.shapes.push(shape);
  }

  return obj;
};

var sampleSecondObject = function(obj, diffIndices) {
  // NOTE: Stupid way to deep copy, but it will work for our small
  // simple objects
  var objCopy = JSON.parse(JSON.stringify(obj));
  for (var i = 0; i < diffIndices.length; i++) {
    var diffIndex = diffIndices[i];
    setObjectDimension(objCopy, diffIndex, makeNewRandomDimensionValue(objCopy, diffIndex));
  }

  return objCopy;
};

var getDimensionType = function(obj, dimensionIndex) {
  return Math.floor(dimensionIndex / obj.shapeCount);
};

var getColorShapeIndexFromDimensionIndex = function(obj, dimensionIndex) {
  return (dimensionIndex % obj.shapeCount);
};

var getSideCountShapeIndexFromDimensionIndex = function(obj, dimensionIndex) {
  return (dimensionIndex % obj.shapeCount);
};

var getParentShapeIndexFromDimensionIndex = function(obj, dimensionIndex) {
  return (dimensionIndex % obj.shapeCount) + 1;
};

var makeNewRandomDimensionValue = function(obj, dimensionIndex) {
  var dimType = getDimensionType(obj, dimensionIndex);
  if (dimType == DIMENSION_TYPE_COLOR) {
    return utils.randomColor({ fixedL : true });
  } else if (dimType == DIMENSION_TYPE_SIDECOUNT) {
    var value = POSSIBLE_SIDE_COUNTS[Math.floor(Math.random()*POSSIBLE_SIDE_COUNTS.length)] + 1;
    var shapeIndex = getSideCountShapeIndexFromDimensionIndex(obj, dimensionIndex);
    while (value <= obj.shapes[shapeIndex].numSides) {
      value = POSSIBLE_SIDE_COUNTS[Math.floor(Math.random()*POSSIBLE_SIDE_COUNTS.length)] + 1;
    }
    return value;
  } else if (dimType == DIMENSION_TYPE_PARENT) {
    var shapeIndex = getParentShapeIndexFromDimensionIndex(obj, dimensionIndex);
    var shape = obj.shapes[shapeIndex];
    var parent = Math.floor(Math.random()*shapeIndex);
    var parentSide = Math.floor(Math.random()*obj.shapes[parent].numSides);
    while ((parent == shape.parent && parentSide == shape.parentSide)
      || obj.shapes[parent].occupiedSides.length == obj.shapes[parent].numSides
      || obj.shapes[parent].occupiedSides.indexOf(parentSide) >= 0) {
      parent = Math.floor(Math.random()*shapeIndex);
      parentSide = Math.floor(Math.random()*obj.shapes[parent].numSides);
    }

    return [parent, parentSide];
  }
};

var setObjectDimension = function(obj, dimensionIndex, dimensionValue) {
  var dimType = getDimensionType(obj, dimensionIndex);
  if (dimType == DIMENSION_TYPE_COLOR) {
    var shapeIndex = getColorShapeIndexFromDimensionIndex(obj, dimensionIndex);
    var shape = obj.shapes[shapeIndex];
    shape.color = dimensionValue;
  } else if (dimType == DIMENSION_TYPE_SIDECOUNT) {
    var shapeIndex = getSideCountShapeIndexFromDimensionIndex(obj, dimensionIndex);
    var shape = obj.shapes[shapeIndex];
    shape.numSides = dimensionValue;
  } else if (dimType == DIMENSION_TYPE_PARENT) {
    var shapeIndex = getParentShapeIndexFromDimensionIndex(obj, dimensionIndex);
    var shape = obj.shapes[shapeIndex];

    obj.shapes[shape.parent].occupiedSides.splice(obj.shapes[shape.parent].occupiedSides.indexOf(shape.parentSide), 1);

    shape.parent = dimensionValue[0];
    shape.parentSide = dimensionValue[1];
    obj.shapes[shape.parent].occupiedSides.push(shape.parentSide);

    for (var i = 1; i < obj.shapes.length; i++) {
      obj.shapes[i].depth = obj.shapes[obj.shapes[i].parent].depth + 1;
    }
  }
};

/*  Sample k unique integers from {0,...,n-1} */
var sampleSet = function(n, k) {
  var set = [];
  for (var i = 0; i < n; i++) {
     set.push(i);
  }

  for (var i = 0; i < n - k; i++) {
    var toRemove = Math.floor(Math.random() * set.length);
    set.splice(toRemove, 1);
  }
  return set;
};

var roundObject = function(obj) {
  for (var i = 0; i < obj.shapes.length; i++) {
    obj.shapes[i].center.x = Math.round(obj.shapes[i].center.x);
    obj.shapes[i].center.y = Math.round(obj.shapes[i].center.y);
    for (var j = 0; j < obj.shapes[i].points.length; j++) {
      obj.shapes[i].points[j].x = Math.round(obj.shapes[i].points[j].x);
      obj.shapes[i].points[j].y = Math.round(obj.shapes[i].points[j].y);
    }
  }
};

var centerObject = function(obj) {
  var point_min = { x : Number.POSITIVE_INFINITY, y : Number.POSITIVE_INFINITY };
  var point_max = { x : Number.NEGATIVE_INFINITY, y : Number.NEGATIVE_INFINITY };

  for (var i = 0; i < obj.shapes.length; i++) {
    for (var j = 0; j < obj.shapes[i].points.length; j++) {
      point_min.x = Math.min(point_min.x, obj.shapes[i].points[j].x);
      point_min.y = Math.min(point_min.y, obj.shapes[i].points[j].y);
      point_max.x = Math.max(point_max.x, obj.shapes[i].points[j].x);
      point_max.y = Math.max(point_max.y, obj.shapes[i].points[j].y);
    }
  }

  var center = { x : (point_max.x + point_min.x)/2.0, y : (point_max.y + point_min.y)/2.0 };
  var trans = { x : OBJECT_WIDTH / 2.0 - center.x, y : OBJECT_HEIGHT/2.0 - center.y};
  for (var i = 0; i < obj.shapes.length; i++) {
    obj.shapes[i].center.x += trans.x;
    obj.shapes[i].center.y += trans.y;
    for (var j = 0; j < obj.shapes[i].points.length; j++) {
      obj.shapes[i].points[j].x += trans.x;
      obj.shapes[i].points[j].y += trans.y;
    }
  }

};

var computeObjectGeometry = function(obj) {
  var rootCenterX = OBJECT_WIDTH / 2.0;
  var rootCenterY = OBJECT_HEIGHT / 2.0;
  for (var i = 0; i < obj.shapes.length; i++) {
    var shape = obj.shapes[i];

    if (i == 0) {
      var interiorAngle = (shape.numSides - 2.0)*Math.PI/shape.numSides;
      var point0 = { x: rootCenterX - obj.sideLength / 2.0, y: rootCenterY + (obj.sideLength/2.0)*Math.tan(interiorAngle/2.0) };
      var point1 = { x: point0.x + obj.sideLength, y: point0.y};
      computeShapeGeometry(shape, point0, point1);
    } else {
      var point0 = obj.shapes[shape.parent].points[shape.parentSide];
      var point1 = obj.shapes[shape.parent].points[(shape.parentSide + 1) % obj.shapes[shape.parent].numSides];
      computeShapeGeometry(shape, point0, point1);
    }
  }
};

var computeShapeGeometry = function(shape, point0, point1) {
  var theta = Math.atan2(point1.y-point0.y, point1.x-point0.x);
  var s = Math.sqrt(Math.pow(point1.y-point0.y,2.0)+Math.pow(point1.x-point0.x,2.0));
  var interiorAngle = (shape.numSides - 2.0)*Math.PI/shape.numSides;
  var dir = Math.pow(-1.0, shape.depth);

  var thetaCenter = theta + dir*(interiorAngle / 2.0);
  var l = s/(2.0*Math.cos(interiorAngle/2.0))
  shape.center = { x : point0.x + l*Math.cos(thetaCenter), y : point0.y + l*Math.sin(thetaCenter) };

  var points = [{x:point0.x, y:point0.y}, {x:point1.x,y:point1.y}];
  var curPoint = point1;
  for (var i = 0; i < shape.numSides - 2; i++) {
    theta += dir*(Math.PI-interiorAngle);
    nextPoint = { x : curPoint.x + s*Math.cos(theta), y : curPoint.y + s*Math.sin(theta) };
    points.push(nextPoint);
    curPoint = nextPoint;
  }

  shape.points = points;
};

if (typeof module !== 'undefined') {
  module.exports = {
    OBJECT_HEIGHT,
    OBJECT_WIDTH,
    MIN_SHAPE_LENGTH,
    MAX_SHAPE_LENGTH,
    POSSIBLE_SHAPE_COUNTS,
    POSSIBLE_SIDE_COUNTS,
    makeRandom,
    fromDimensionValueString,
    getDimensionNames,
    getDimensionNamesString,
    getDimensionValues,
    getDimensionValuesString,
    getDimensionValue
  };
}
