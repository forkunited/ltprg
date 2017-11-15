var DIMENSION_TYPE_COLOR = 0;
var DIMENSION_TYPE_SIDECOUNT = 1;
var DIMENSION_TYPE_PARENT = 2;

var OBJECT_HEIGHT = 400;
var OBJECT_WIDTH = 300;
var CELL_LENGTH = 40;
var GRID_DIMENSION = 3;

var makeRandom = function(trialNum, numRounds) {
  var obj = sampleObject(); // Sample a world state
  var numDimensions = getObjectDimensionCount(obj);
  var numDiffs = Math.floor((trialNum / numRounds)*numDimensions) + 1;
  var diffIndices = sampleSet(numDimensions, numDiffs);
  var otherObj = sampleSecondObject(obj, diffIndices);

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

  var sObj0 = { shapes : [], gridDimension : GRID_DIMENSION, cellLength : CELL_LENGTH };
  var startIndex0 = 7;
  for (var i = 0; i < GRID_DIMENSION*GRID_DIMENSION; i++) {
    var shape = {};
    var startIndexS = startIndex0 + i*3;

    shape.color = [parseInt(values[startIndexS]),
                   parseInt(values[startIndexS+1]),
                   parseInt(values[startIndexS+2])];

    sObj0.shapes.push(shape);
  }

  var sObj1 = { shapes : [], gridDimension : GRID_DIMENSION, cellLength : CELL_LENGTH };
  var startIndex1 = startIndex0 + GRID_DIMENSION*GRID_DIMENSION*3;
  for (var i = 0; i < GRID_DIMENSION*GRID_DIMENSION; i++) {
    var shape = {};
    var startIndexS = startIndex1 + i*3;

    shape.color = [parseInt(values[startIndexS]),
                   parseInt(values[startIndexS+1]),
                   parseInt(values[startIndexS+2])];

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
  } else if (dimName.indexOf("ClrH") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    return obj.shapes[s].color[0];
  } else if (dimName.indexOf("ClrS") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    return obj.shapes[s].color[1];
  } else if (dimName.indexOf("ClrL") >= 0) {
    var obj = getObjectByRole(trial, nameParts["Obj"], nameParts["Role"]);
    var s = nameParts["Shp"];
    return obj.shapes[s].color[2];
  }
};

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

  for (var i = 0; i < GRID_DIMENSION*GRID_DIMENSION; i++) {
    names.push(prefix + "_Shp" + i + "_ClrH");
    names.push(prefix + "_Shp" + i + "_ClrS");
    names.push(prefix + "_Shp" + i + "_ClrL");
  }

  return names;
};

var getObjectDimensionCount = function(obj) {
  return obj.shapes.length;
};

var sampleObject = function() {
  var obj = { cellLength : CELL_LENGTH, gridDimension : GRID_DIMENSION, shapes : [] };

  for (var i = 0; i < GRID_DIMENSION*GRID_DIMENSION; i++) {
    obj.shapes.push({ color : utils.randomColor({ fixedL : true })})
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

var makeNewRandomDimensionValue = function(obj, dimensionIndex) {
  return utils.randomColor({ fixedL : true });
};

var setObjectDimension = function(obj, dimensionIndex, dimensionValue) {
  var shape = obj.shapes[dimensionIndex];
  shape.color = dimensionValue;
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

if (typeof module !== 'undefined') {
  module.exports = {
    OBJECT_HEIGHT,
    OBJECT_WIDTH,
    GRID_DIMENSION,
    CELL_LENGTH,
    makeRandom,
    fromDimensionValueString,
    getDimensionNames,
    getDimensionNamesString,
    getDimensionValues,
    getDimensionValuesString,
    getDimensionValue
  };
}
