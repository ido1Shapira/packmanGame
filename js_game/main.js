//
// Author: Ido Shapira
// date: 03/08/2021
//
var ctx = null;
var canvas = null;
var gameMap = [
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 1, 1, 1, 1, 0, 1, 1, 1, 0,
	0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
	0, 1, 1, 1, 0, 1, 1, 1, 1, 0,
	0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
	0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
	0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
	0, 1, 1, 1, 0, 1, 1, 1, 1, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0
];
var gameStatus = {
	inProgress: true,
	ended: false,
	stoped: false
}
const numOfAwards = 5;

var tileW = 50, tileH = 50;
var mapW = 10, mapH = 10;
var currentSecond = 0, frameCount = 0, framesLastSecond = 0, lastFrameTime = 0;

                    // tileFrom , tileTo, timeMoved, dimensions, dimensions_at_stay, position, delayMove
var human_player = new Character([3,4], [3,4], 0, [30,30], [20, 20], [160,215], 500);

initializeFirebase();
var computer_player = new Character([4,6], [4,6], 0, [30,30], [20, 20], [210,310], 500);
var computer_controller;
firebase.database().ref("chosen-controller").once('value',
    (snap) => {
        selectedBehavior = snap.val();
        computer_controller = new PlayerController(computer_player, selectedBehavior);

		var type = computer_controller.getType();
        // Generate a reference to a new location and add some data using push()
        var newPostRef = firebase.database().ref("all-games").push({
            behavior: type
        });
        // Get the unique ID generated by push() by accessing its key
        postID = newPostRef.key;
        // console.log("postID: "+postID);
    });
 

function position(tile, dimensions)
{
	return [Math.round((tile[0] * tileW) + ((tileW-dimensions[0])/2)),
			Math.round((tile[1] * tileH) + ((tileH-dimensions[1])/2))];
}
function randomValidTiles(n) {
	var indexs = []; // find valid indexes
	gameMap.filter(function(elem, index, array){
		if(elem == 1) {
			var i = Math.floor(index / 10);
			var j = index % 10;
			if(j != computer_player.tileFrom[0] && i != computer_player.tileFrom[1]
				&& j != human_player.tileFrom[0] && i != human_player.tileFrom[1]) {
					// award can not be in player tile
				indexs.push([j,i]);
			}
		}
	});
	var result = [];
	// choose randomly n indexs
	for (var i = 0; i < n; i++) {
		var idx = Math.floor(Math.random() * indexs.length);
		result.push(indexs[idx]);
		indexs.splice(idx, 1);
	}
	return result;
}

var awards = [];
// It is not random anymore
// var validTiles = randomValidTiles(numOfAwards);
var validTiles = [[1,3],[1,6],[3,8],[7,8],[8,6]];
for(var i=0; i<numOfAwards; i++) {
	awards.push(new Award(validTiles[i], [15,15], position(validTiles[i], [15, 15]), 0.05, 5));
}
var human_awards = [];
var computer_awards = [];

function toIndex(x, y)
{
	return((y * mapW) + x);
}

function zeros(dimensions) { // dimensions = [r,c] 
    var array = [];
    for (var i = 0; i < dimensions[0]; ++i) {
        array.push(dimensions.length == 1 ? 0 : zeros(dimensions.slice(1)));
    }
    return array;
}

var prev_state = [[], zeros([mapW, mapH]), zeros([mapW, mapH]), [], [], []];
prev_state[1][human_player.tileFrom[1]][human_player.tileFrom[0]] = 1;
prev_state[2][computer_player.tileFrom[1]][computer_player.tileFrom[0]] = 1;

function getBoardState() {
	var state = [
			[], // the board
			[], // human trace
			[], // computer trace
			zeros([mapW, mapH]), // awards collected by human
			zeros([mapW, mapH]), // awards collected by computer
			zeros([mapW, mapH]) // all awards
			];
	
	var board = gameMap.slice();
	while(board.length) state[0].push(board.splice(0,10)); // reshape board

	//human trace
	for(var i=0; i<prev_state[1].length; i++) {
		state[1].push(prev_state[1][i].map(x => Number((x * 0.9).toFixed(2))));
	}
	state[1][human_player.tileFrom[1]][human_player.tileFrom[0]] = 1;
	//computer trace 
	for(var i=0; i<prev_state[2].length; i++) {
		state[2].push(prev_state[2][i].map(x => Number((x * 0.9).toFixed(2))));
	}
	state[2][computer_player.tileFrom[1]][computer_player.tileFrom[0]] = 1;
	//human collected awards
	for(award of human_awards) {
		state[3][award.tile[1]][award.tile[0]] = 1;
	}
	//computer collected awards
	for(award of computer_awards) {
		state[4][award.tile[1]][award.tile[0]] = 1;
	}
	//allawards
	for(award of awards) {
		state[5][award.tile[1]][award.tile[0]] = 1;
	}
	prev_state = state.slice();	
	return state;
}

var humanMove = null;
var computerMove = null;
var handleKeyUp = function(e) {
	if((e.keyCode>=37 && e.keyCode<=40) || e.keyCode==32) {
		var currentFrameTime = Date.now();
		if((currentFrameTime-human_player.timeMoved>=human_player.delayMove)) {
			var validHumanAction = false;
			switch(e.keyCode) {
				case 32:
						//stay
						validHumanAction = true;
						humanMove = e.keyCode;
					break;
				case 37:
					if(human_player.tileFrom[0]>0 && gameMap[toIndex(human_player.tileFrom[0]-1, human_player.tileFrom[1])]==1) {
						human_player.tileTo[0]-= 1; //left
						validHumanAction = true;
						humanMove = e.keyCode;
					}
					break;
				case 38:
					if(human_player.tileFrom[1]>0 && gameMap[toIndex(human_player.tileFrom[0], human_player.tileFrom[1]-1)]==1) {
						human_player.tileTo[1]-= 1; //up
						validHumanAction = true;
						humanMove = e.keyCode;
					}
					break;
				case 39:
					if(human_player.tileFrom[0]<(mapW-1) && gameMap[toIndex(human_player.tileFrom[0]+1, human_player.tileFrom[1])]==1) {
						human_player.tileTo[0]+= 1; //right
						validHumanAction = true;
						humanMove = e.keyCode;
					}
					break;
				case 40:
					if(human_player.tileFrom[1]<(mapH-1) && gameMap[toIndex(human_player.tileFrom[0], human_player.tileFrom[1]+1)]==1) {
						human_player.tileTo[1]+= 1; //down
						validHumanAction = true;
						humanMove = e.keyCode;
					}
					break;
			}
			human_player.timeMoved = currentFrameTime;
			human_player.keysDown[e.keyCode] = true;

			if(validHumanAction) {
				//blue player move
				var state = getBoardState();
				saveToFirebase(state, humanMove);
				computerMove = computer_controller.move(state);
				computer_player.keysDown[computerMove] = true;
			}
		}
	}
}
window.onload = function()
{
	canvas = document.getElementById('game');
	ctx = canvas.getContext("2d");
	requestAnimationFrame(drawGame);
	ctx.font = "bold 10pt sans-serif";

	// window.addEventListener("keyup", handleKeyUp);
};

function drawGame()
{
	if(ctx==null) { return; }

	var currentFrameTime = Date.now();
	var timeElapsed = currentFrameTime - lastFrameTime;
	
	var sec = Math.floor(Date.now()/1000);
	if(sec!=currentSecond)
	{
		currentSecond = sec;
		framesLastSecond = frameCount;
		frameCount = 1;
	}
	else { frameCount++; }

	if(!human_player.processMovement(currentFrameTime)) //move human player on board
	{
		human_player.keysDown[humanMove] = false;

		if(human_player.tileFrom[0]!=human_player.tileTo[0] || human_player.tileFrom[1]!=human_player.tileTo[1])
		{ human_player.timeMoved = currentFrameTime; }
	}
	

	if(!computer_player.processMovement(currentFrameTime)) //move computer player on board
	{
		if(computer_player.keysDown[38] && computer_player.tileFrom[1]>0 && gameMap[toIndex(computer_player.tileFrom[0], computer_player.tileFrom[1]-1)]==1) { computer_player.tileTo[1]-= 1; }
		else if(computer_player.keysDown[40] && computer_player.tileFrom[1]<(mapH-1) && gameMap[toIndex(computer_player.tileFrom[0], computer_player.tileFrom[1]+1)]==1) { computer_player.tileTo[1]+= 1; }
		else if(computer_player.keysDown[37] && computer_player.tileFrom[0]>0 && gameMap[toIndex(computer_player.tileFrom[0]-1, computer_player.tileFrom[1])]==1) { computer_player.tileTo[0]-= 1; }
		else if(computer_player.keysDown[39] && computer_player.tileFrom[0]<(mapW-1) && gameMap[toIndex(computer_player.tileFrom[0]+1, computer_player.tileFrom[1])]==1) { computer_player.tileTo[0]+= 1; }
		else if(human_player.keysDown[32]) { }
		// computer_player.resetKeyPress();
		computer_player.keysDown[computerMove] = false;

		if(computer_player.tileFrom[0]!=computer_player.tileTo[0] || computer_player.tileFrom[1]!=computer_player.tileTo[1])
		{ computer_player.timeMoved = currentFrameTime; }
	}

	//check for eaten awards
	for(var i=0; i<awards.length; i++) {
		temp_award = awards[i];
		if(temp_award.tile[0] == human_player.tileFrom[0] && temp_award.tile[1] == human_player.tileFrom[1]) {
			awards.splice(i, 1);
			human_player.score = human_player.score + temp_award.value;
			human_player.scoreToView = human_player.scoreToView + temp_award.valueToView;
			if(human_player.tileFrom[0] == computer_player.tileFrom[0] && human_player.tileFrom[1] == computer_player.tileFrom[0]) {
				computer_player.score = computer_player.score + award.value;
				computer_player.scoreToView = computer_player.scoreToView + award.valueToView;
				computer_awards.push(temp_award);
			}
			human_awards.push(temp_award);
		}
		if(temp_award.tile[0] == computer_player.tileFrom[0] && temp_award.tile[1] == computer_player.tileFrom[1]) {
			awards.splice(i, 1);
			computer_player.score = computer_player.score + temp_award.value;
			computer_player.scoreToView = computer_player.scoreToView + temp_award.valueToView;
			if(human_player.tileFrom[0] == computer_player.tileFrom[0] && human_player.tileFrom[1] == computer_player.tileFrom[0]) {
				// two player got to the award at the same time
				human_player.score = human_player.score + award.value;
				human_player.scoreToView = human_player.scoreToView + award.valueToView;
				human_awards.push(temp_award);
			}
			computer_awards.push(temp_award);
			
		}
	}

	if(awards.length == 0 && gameStatus.inProgress) { 
		// game ended when their is no awards on board
		gameStatus.inProgress = false;
		gameStatus.ended = true;
	}

	if(gameStatus.ended) {
		gameStatus.ended = false;
		gameStatus.stoped = true;

		human_player.score = human_player.score + human_player.scores.finish;
		human_player.scoreToView = human_player.scoreToView + human_player.scoresToView.finish;

		computer_player.score = computer_player.score + computer_player.scores.finish;
		computer_player.scoreToView = computer_player.scoreToView + computer_player.scoresToView.finish;
		
		console.log('1) Computer score: '+ computer_player.scoreToView);
		console.log('2) Computer score: '+ computer_player.score);
		console.log('1) Human score: '+ human_player.scoreToView);
		console.log('2) Human score: '+ human_player.score);
		window.removeEventListener("keyup", handleKeyUp);

		finishGame(getBoardState(), humanMove); //sending the last state
	}

	for(var y = 0; y < mapH; ++y) // draw the board
	{
		for(var x = 0; x < mapW; ++x)
		{
			switch(gameMap[((y*mapW)+x)])
			{
				case 0:
					ctx.fillStyle = "#685b48"; // color: brown
					break;
				default:
					ctx.fillStyle = "#5aa457"; // color: green
			}

			ctx.fillRect( x*tileW, y*tileH, tileW, tileH);
		}
	}
	for(award of awards) {
		ctx.fillStyle = "#FF8000"; // award color: orange
		ctx.beginPath();
		ctx.arc(award.position[0] + 5, award.position[1] + 5, 10, 0, 2 * Math.PI, false);
		ctx.fill();
		// ctx.fillRect(award.position[0], award.position[1],
		// 	award.dimensions[0], award.dimensions[1]);
	}
	
	var borderWidth = 1;   
	var offset = borderWidth * 2;
	
	ctx.fillStyle = "black"; // draw player borders
	if(computerMove == 32 && human_player.keysDown[humanMove]) {
		ctx.fillRect( computer_player.position[0] - borderWidth, computer_player.position[1] -borderWidth,
			computer_player.dimensions_at_stay[0] + offset, computer_player.dimensions_at_stay[1] + offset);
		ctx.fillStyle = "#0000ff"; // computer color: blue
		ctx.fillRect(computer_player.position[0], computer_player.position[1],
			computer_player.dimensions_at_stay[0], computer_player.dimensions_at_stay[1]);
		drawBubble(ctx, computer_player.position[0], computer_player.position[1] + 20 , 120, 50, 20);
		ctx.fillStyle = "#0000ff";
		ctx.fillText("I'm staying", computer_player.position[0] + 10, computer_player.position[1]+40);
		ctx.fillText("and not helping.", computer_player.position[0] + 10, computer_player.position[1]+60);
	}
	else {
		ctx.fillRect( computer_player.position[0] - borderWidth, computer_player.position[1] -borderWidth,
		computer_player.dimensions[0] + offset, computer_player.dimensions[1] + offset);
		ctx.fillStyle = "#0000ff";
		ctx.fillRect(computer_player.position[0], computer_player.position[1],
			computer_player.dimensions[0], computer_player.dimensions[1]);
	}

	ctx.fillStyle = "black"; 
	if (humanMove == 32 && 	human_player.keysDown[humanMove]) {
		ctx.fillRect( human_player.position[0] - borderWidth, human_player.position[1] -borderWidth,
			human_player.dimensions_at_stay[0] + offset, human_player.dimensions_at_stay[1] + offset);
		ctx.fillStyle = "#FF5050"; // human color: red
		ctx.fillRect(human_player.position[0], human_player.position[1],
			human_player.dimensions_at_stay[0], human_player.dimensions_at_stay[1]);
		drawBubble(ctx, human_player.position[0], human_player.position[1] + 20 , 120, 50, 20);
		ctx.fillStyle = "#FF5050";
		ctx.fillText("I'm staying", human_player.position[0] + 10, human_player.position[1]+40);
		ctx.fillText("and not helping.", human_player.position[0] + 10, human_player.position[1]+60);
	}
	else {
		ctx.fillRect( human_player.position[0] - borderWidth, human_player.position[1] -borderWidth,
			human_player.dimensions[0] + offset, human_player.dimensions[1] + offset);
		ctx.fillStyle = "#FF5050";
		ctx.fillRect(human_player.position[0], human_player.position[1],
			human_player.dimensions[0], human_player.dimensions[1]);
	}

	ctx.fillStyle = "#FFFFFF"; // title color: white
	// ctx.fillText("FPS: " + framesLastSecond, 10, 20);
    ctx.fillText("Red score: " + human_player.scoreToView, 20, 25);
	ctx.fillText("Blue score: " + computer_player.scoreToView, 355, 25);

	ctx.fillText("Red action: " + human_player.codeToAction[humanMove], 20, 475);
    ctx.fillText("Blue action: " + computer_player.codeToAction[computerMove], 355, 475);

	lastFrameTime = currentFrameTime;
	requestAnimationFrame(drawGame);
}

function drawBubble(ctx, x, y, w, h, radius) {
    var r = x + w;
    var b = y + h;
    ctx.beginPath();
    ctx.strokeStyle = "black";
    ctx.lineWidth = "2";

    var handle = {
        x1: x + radius,
        y1: y,
        x2: x + radius / 2,
        y2: y - 10,
        x3: x + radius * 2,
        y3: y
    }
	
    ctx.moveTo(handle.x1, handle.y1);
    ctx.lineTo(handle.x2, handle.y2);
    ctx.lineTo(handle.x3, handle.y3);

    ctx.lineTo(r - radius, y);
    ctx.quadraticCurveTo(r, y, r, y + radius);
    ctx.lineTo(r, y + h - radius);
    ctx.quadraticCurveTo(r, b, r - radius, b);
    ctx.lineTo(x + radius, b);
    ctx.quadraticCurveTo(x, b, x, b - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);

	ctx.fillStyle = "white";
	ctx.fill();
    ctx.stroke();
    
    return handle;
}
