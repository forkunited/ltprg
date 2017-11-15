// drawing.js
// This file contains functions to draw on the HTML5 canvas

var clearScreen = function(game) {
  game.ctx.clearRect(0, 0, game.world.width, game.world.height);
};

var drawScreen = function(game, mouseX, mouseY) {
  var player = {}; //game.get_player(globalGame.my_id)

  // draw background
  game.ctx.fillStyle = "#FFFFFF";
  game.ctx.fillRect(0,0,game.viewport.width,game.viewport.height);

  // Draw message in center (for countdown, e.g.)
  if (player.message) {
    game.ctx.font = "bold 23pt Helvetica";
    game.ctx.fillStyle = 'blue';
    game.ctx.textAlign = 'center';
    wrapText(game, player.message,
             game.world.width/2, game.world.width/4,
             game.world.width*4/5,
             25);
  } else {
    drawObjects(game);
    if (game.my_role === game.playerRoleNames.role1) { // Speaker
      drawTargetBox(game);
    } else if (game.my_role ==game.playerRoleNames.role2) { // Listener
      drawHoverBox(game, mouseX, mouseY);
    }
    drawDividers(game);
  }
};

var drawDividers = function(game) {
  var numObjs = game.currStim.objs.length;
  var objWidth = game.world.width / numObjs;

  for (var i = 1; i < numObjs; i++) {
    game.ctx.strokeStyle = 'black';
    game.ctx.lineWidth = 5;
    game.ctx.beginPath();
    game.ctx.moveTo(objWidth*i, 0);
    game.ctx.lineTo(objWidth*i, game.world.height);
    game.ctx.closePath();
    game.ctx.stroke();
  }
};

var getHoverIndex = function(game, mouseX, mouseY) {
  var numObjs = game.currStim.objs.length;
  var objWidth = game.world.width / numObjs;
  return Math.floor(mouseX/objWidth);
};

var drawHoverBox = function(game, mouseX, mouseY) {
  if (typeof(mouseX) == 'undefined' && typeof(mouseY) == 'undefined')
    return undefined;

  var hoverObj = getHoverIndex(game, mouseX, mouseY);
  if (game.my_role !== game.playerRoleNames.role1) { // Listener
    drawBox(game, hoverObj, "rgba(0, 0, 255, 0.8)");
  }

  return hoverObj;
};

var drawTargetBox = function(game) {
  var world = game.currStim;
  var objIndex = 0;
  if (game.my_role === game.playerRoleNames.role1) { // Speaker
    objIndex = world.speakerOrder.indexOf(world.target);
  } else {
    objIndex = world.listenerOrder.indexOf(world.target);
  }

  drawBox(game, objIndex, "rgba(0, 255, 0, 0.8)")
}

var drawClickedCorrectBox = function(game, mouseX, mouseY) {
  var numObjs = game.currStim.objs.length;
  var objWidth = game.world.width / numObjs;
  var clickedObj = Math.floor(mouseX/objWidth);
  if (game.my_role !== game.playerRoleNames.role1) { // Listener
    var targetIndex = game.currStim.listenerOrder.indexOf(game.currStim.target);
    if (targetIndex == clickedObj) {
      drawBox(game, clickedObj, "rgba(255, 0, 0, 0.8)");
    } else {
      drawTargetBox(game);
    }
  }

  return clickedObj;
};

var drawBox = function(game, objIndex, color) {
  var numObjs = game.currStim.objs.length;
  var objWidth = game.world.width / numObjs;

  game.ctx.strokeStyle = color;
  game.ctx.lineWidth = 20;
  game.ctx.beginPath();
  game.ctx.rect(objWidth*objIndex + game.ctx.lineWidth / 2.0, // top-left x
                game.ctx.lineWidth / 2.0, // top-left y
                objWidth - game.ctx.lineWidth, // width
                game.world.height - game.ctx.lineWidth); // height
  game.ctx.closePath();
  game.ctx.stroke();
};

var drawObjects = function(game) {
  var trial = game.currStim;
  var objWidth = game.world.width / trial.objs.length;
  var order = trial.listenerOrder;
  if (game.my_role === game.playerRoleNames.role1) { // Speaker
    order = trial.speakerOrder;
  }

  for (var i = 0; i < trial.speakerOrder.length; i++) {
    var objShiftX = objWidth * i;
    var obj = trial.objs[order[i]];
    for (var j = 0; j < obj.shapeCount; j++) {
      var shape = obj.shapes[j];
      game.ctx.fillStyle = ('hsl(' + shape.color[0] + ',' + shape.color[1] + '%, ' + shape.color[2] + '%)');
      game.ctx.beginPath();
      game.ctx.moveTo(objShiftX + shape.points[0].x, shape.points[0].y);
      for (var k = 1; k < shape.points.length; k++) {
        game.ctx.lineTo(objShiftX + shape.points[k].x, shape.points[k].y);
      }
      game.ctx.closePath();
      game.ctx.fill();
    }
  }
};

// This is a helper function to write a text string onto the HTML5 canvas.
// It automatically figures out how to break the text into lines that will fit
// Input:
//    * game: the game object (containing the ctx canvas object)
//    * text: the string of text you want to writ
//    * x: the x coordinate of the point you want to start writing at (in pixels)
//    * y: the y coordinate of the point you want to start writing at (in pixels)
//    * maxWidth: the maximum width you want to allow the text to span (in pixels)
//    * lineHeight: the vertical space you want between lines (in pixels)
function wrapText(game, text, x, y, maxWidth, lineHeight) {
  var cars = text.split("\n");
  game.ctx.fillStyle = 'white';
  game.ctx.fillRect(0, 0, game.viewport.width, game.viewport.height);
  game.ctx.fillStyle = 'red';

  for (var ii = 0; ii < cars.length; ii++) {

    var line = "";
    var words = cars[ii].split(" ");

    for (var n = 0; n < words.length; n++) {
      var testLine = line + words[n] + " ";
      var metrics = game.ctx.measureText(testLine);
      var testWidth = metrics.width;

      if (testWidth > maxWidth) {
        game.ctx.fillText(line, x, y);
        line = words[n] + " ";
        y += lineHeight;
      }
      else {
        line = testLine;
      }
    }
    game.ctx.fillText(line, x, y);
    y += lineHeight;
  }
};
