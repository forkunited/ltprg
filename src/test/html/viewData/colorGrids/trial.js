var CONDITION_CLOSE = "CLOSE";
var CONDITION_SPLIT = "SPLIT";
var CONDITION_FAR = "FAR";

var OBJECT_HEIGHT = 300;
var OBJECT_WIDTH = 300;
var CELL_LENGTH = 40;

var COLOR_DIFF_FLOOR = 5;
var COLOR_DIFF_FAR = 20;

var makeRandom = function(numObjs, condition, gridDimension) {
  var target = sampleTarget(condition, gridDimension);
  var objs = [];
  var speakerOrder = [0];
  var listenerOrder = [0];
  for (var i = 1; i < numObjs; i++) {
    objs.push(sampleDistractor(condition, gridDimension, target, objs));
    speakerOrder.push(i);
    listenerOrder.push(i);
  }

  shuffle(speakerOrder);
  shuffle(listenerOrder);
  shuffle(objs);

  var targetIndex = Math.floor(Math.random() * numObjs);
  objs.splice(targetIndex, 0, target);

  return { objs: objs,
           target : targetIndex,
           speakerOrder : speakerOrder,
           listenerOrder : listenerOrder,
           condition : { name : condition }
         }
};

// From https://stackoverflow.com/questions/6274339/how-can-i-shuffle-an-array
function shuffle(a) {
    var j, x, i;
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1));
        x = a[i];
        a[i] = a[j];
        a[j] = x;
    }
}

var getObjectByRole = function(trial, objectIndex, role) {
  if (role == "s") {
    return trial.objs[trial.speakerOrder[objectIndex]];
  } else if (role == "l") {
    return trial.objs[trial.listenerOrder[objectIndex]];
  }
};

var sampleTarget = function(condition, gridDimension) {
  var obj = { cellLength : CELL_LENGTH, gridDimension : gridDimension, shapes : [] };

  for (var i = 0; i < gridDimension*gridDimension; i++) {
      obj.shapes.push({ color : utils.randomColor({ fixedL : true })});
  }

  return obj;
};

var sampleDistractor = function(condition, gridDimension, target, distractors) {
  var obj = { cellLength : CELL_LENGTH, gridDimension : gridDimension, shapes : [] };

  for (var i = 0; i < gridDimension*gridDimension; i++) {
      var color = undefined;
      var colorMeetsCondition = false;
      do {
        color = utils.randomColor({ fixedL : true });
        colorMeetsCondition = true;

        var targetClose = false;
        if (condition === CONDITION_CLOSE || (condition === CONDITION_SPLIT && distractors.length % 2 == 0)) {
          targetClose = true;
        }

        var targetDiff = utils.colorDiff(target.shapes[i].color, color);
        if (targetDiff < COLOR_DIFF_FLOOR || (targetClose !== (targetDiff < COLOR_DIFF_FAR))) {
          colorMeetsCondition = false;
        }

        if (colorMeetsCondition) {
          var distractorClose = (condition === CONDITION_CLOSE);
          for (var j = 0; j < distractors.length; j++) {
            var distractorDiff = utils.colorDiff(distractors[j].shapes[i].color, color);
            if (distractorDiff < COLOR_DIFF_FLOOR || (distractorClose !== (distractorDiff < COLOR_DIFF_FAR))) {
              colorMeetsCondition = false;
              break;
            }
          }
        }
      } while (!colorMeetsCondition);

      obj.shapes.push({ color : color })
  }

  return obj;
};

if (typeof module !== 'undefined') {
  module.exports = {
    CONDITION_FAR,
    CONDITION_SPLIT,
    CONDITION_CLOSE,
    OBJECT_HEIGHT,
    OBJECT_WIDTH,
    CELL_LENGTH,
    makeRandom,
  };
}
