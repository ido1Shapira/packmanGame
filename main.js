//
// Author: Ido Shapira
// date: 03/08/2021
//
var ctx = null;
var gameMap = [
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
	0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	0, 1, 0, 1, 0, 0, 0, 1, 1, 0,
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

var tileW = 40, tileH = 40;
var mapW = 10, mapH = 10;
var currentSecond = 0, frameCount = 0, framesLastSecond = 0, lastFrameTime = 0;

                    // tileFrom , tileTo, timeMoved, dimensions, position, delayMove
var human_player = new Character([1,1], [1,1], 0, [30,30], [45,45], 700);

var computer_player = new Character([7,7], [7,7], 0, [30,30], [285,285], 700);
var computer_controller = new PlayerController(computer_player, "farthest");

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
var validTiles = randomValidTiles(numOfAwards);
for(var i=0; i<numOfAwards; i++) {
	awards.push(new Award(validTiles[i], [15,15], position(validTiles[i], [15, 15]), 2));
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
			zeros([mapW, mapH]), // all awards
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

var computerMove = null;
var handleKeyDown = function(e) {
	if((e.keyCode>=37 && e.keyCode<=40) || e.keyCode==32) { //move players on key press
		human_player.keysDown[e.keyCode] = true;
	}
}
var handleKeyUp = function(e) {
	if((e.keyCode>=37 && e.keyCode<=40) || e.keyCode==32) {
		human_player.keysDown[e.keyCode] = false;
		computer_player.keysDown[computerMove] = false;
	}
	// console.log(e.keyCode);
}
window.onload = function()
{
	ctx = document.getElementById('game').getContext("2d");
	requestAnimationFrame(drawGame);
	ctx.font = "bold 10pt sans-serif";

	window.addEventListener("keydown", handleKeyDown);
	window.addEventListener("keyup", handleKeyUp);
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
		var human_player_moves = false;
		if(human_player.keysDown[38] && human_player.tileFrom[1]>0 && gameMap[toIndex(human_player.tileFrom[0], human_player.tileFrom[1]-1)]==1) {
			human_player.tileTo[1]-= 1; //up
			human_player_moves = true;
		}
		else if(human_player.keysDown[40] && human_player.tileFrom[1]<(mapH-1) && gameMap[toIndex(human_player.tileFrom[0], human_player.tileFrom[1]+1)]==1) {
			human_player.tileTo[1]+= 1; //down
			human_player_moves = true;
		}
		else if(human_player.keysDown[37] && human_player.tileFrom[0]>0 && gameMap[toIndex(human_player.tileFrom[0]-1, human_player.tileFrom[1])]==1) {
			human_player.tileTo[0]-= 1; //left
			human_player_moves = true;
		}
		else if(human_player.keysDown[39] && human_player.tileFrom[0]<(mapW-1) && gameMap[toIndex(human_player.tileFrom[0]+1, human_player.tileFrom[1])]==1) {
			human_player.tileTo[0]+= 1; //right
			human_player_moves = true;
		}
		else if(human_player.keysDown[32] && (currentFrameTime-human_player.timeMoved>=human_player.delayMove)) {
			human_player_moves = true; //stay
			human_player.timeMoved = currentFrameTime;
			human_player.score = human_player.score + human_player.scores.stay;
		}
		if(human_player_moves) {
			computer_player.resetKeyPress();
			computerMove = computer_controller.move(getBoardState());
			console.log("computerMove: "+ computerMove);
			computer_player.keysDown[computerMove] = true;
			if(computer_player.keysDown[32] && (currentFrameTime-computer_player.timeMoved>=computer_player.delayMove)) {
				// TODO: I think there is a problem with the stay score. it not always update the computer score
				computer_player.score = computer_player.score + computer_player.scores.stay;
			}
		}

		if(human_player.tileFrom[0]!=human_player.tileTo[0] || human_player.tileFrom[1]!=human_player.tileTo[1])
		{ human_player.timeMoved = currentFrameTime; }
	}

	if(!computer_player.processMovement(currentFrameTime)) //move computer player on board
	{
		if(computer_player.keysDown[38] && computer_player.tileFrom[1]>0 && gameMap[toIndex(computer_player.tileFrom[0], computer_player.tileFrom[1]-1)]==1) { computer_player.tileTo[1]-= 1; }
		else if(computer_player.keysDown[40] && computer_player.tileFrom[1]<(mapH-1) && gameMap[toIndex(computer_player.tileFrom[0], computer_player.tileFrom[1]+1)]==1) { computer_player.tileTo[1]+= 1; }
		else if(computer_player.keysDown[37] && computer_player.tileFrom[0]>0 && gameMap[toIndex(computer_player.tileFrom[0]-1, computer_player.tileFrom[1])]==1) { computer_player.tileTo[0]-= 1; }
		else if(computer_player.keysDown[39] && computer_player.tileFrom[0]<(mapW-1) && gameMap[toIndex(computer_player.tileFrom[0]+1, computer_player.tileFrom[1])]==1) { computer_player.tileTo[0]+= 1; }

		if(computer_player.tileFrom[0]!=computer_player.tileTo[0] || computer_player.tileFrom[1]!=computer_player.tileTo[1])
		{ computer_player.timeMoved = currentFrameTime; }
	}

	//check for eaten awards
	for(var i=0; i<awards.length; i++) {
		temp_award = awards[i];
		if(temp_award.tile[0] == human_player.tileFrom[0] && temp_award.tile[1] == human_player.tileFrom[1]) {
			awards.splice(i, 1);
			human_player.score = human_player.score + temp_award.value;
			if(human_player.tileFrom[0] == computer_player.tileFrom[0] && human_player.tileFrom[1] == computer_player.tileFrom[0]) {
				computer_player.score = computer_player.score + award.value;
				computer_awards.push(temp_award);
			}
			human_awards.push(temp_award);
		}
		if(temp_award.tile[0] == computer_player.tileFrom[0] && temp_award.tile[1] == computer_player.tileFrom[1]) {
			awards.splice(i, 1);
			computer_player.score = computer_player.score + temp_award.value;
			if(human_player.tileFrom[0] == computer_player.tileFrom[0] && human_player.tileFrom[1] == computer_player.tileFrom[0]) {
				// two player got to the award at the same time
				human_player.score = human_player.score + award.value;
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
		computer_player.score = computer_player.score + computer_player.scores.finish;
		
		window.removeEventListener("keydown", handleKeyDown);
		window.removeEventListener("keyup", handleKeyUp);
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
		ctx.fillRect(award.position[0], award.position[1],
			award.dimensions[0], award.dimensions[1]);
	}
	
	ctx.fillStyle = "#0000ff"; // computer color: blue
	ctx.fillRect(computer_player.position[0], computer_player.position[1],
		computer_player.dimensions[0], computer_player.dimensions[1]);

	ctx.fillStyle = "#FF5050"; // human color: red
	ctx.fillRect(human_player.position[0], human_player.position[1],
		human_player.dimensions[0], human_player.dimensions[1]);

	ctx.fillStyle = "#000000"; // title color: black
	// ctx.fillText("FPS: " + framesLastSecond, 10, 20);
    ctx.fillText("Red score: " + human_player.score, 110, 20);
	ctx.fillText("Blue score: " + computer_player.score, 210, 20);

	lastFrameTime = currentFrameTime;
	requestAnimationFrame(drawGame);
}
