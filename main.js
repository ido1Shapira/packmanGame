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
	0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
	0, 1, 1, 1, 0, 1, 1, 1, 1, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0
];

var tileW = 40, tileH = 40;
var mapW = 10, mapH = 10;
var currentSecond = 0, frameCount = 0, framesLastSecond = 0, lastFrameTime = 0;

                    // tileFrom , tileTo, timeMoved, dimensions, position, delayMove
var human_player = new Character([1,1], [1,1], 0, [30,30], [45,45], 700);

var computer_player = new Character([7,7], [7,7], 0, [30,30], [285,285], 700);
var computer_controller = new PlayerController(computer_player);
		
function position(tile, dimensions)
{
	return [Math.round((tile[0] * tileW) + ((tileW-dimensions[0])/2)),
			Math.round((tile[1] * tileH) + ((tileH-dimensions[1])/2))];
}
function randomValidTile() {
	var indexs = []; // find valid indexes
	gameMap.filter(function(elem, index, array){
		if(elem == 1) {
			var i = Math.floor(index / 10);
			var j = index % 10;
			indexs.push([i,j]);
		}
	});
	console.log(indexs);
	return indexs[indexs.length * Math.random() | 0];
}

var awards = [];
for(var i=1; i<=5; i++) {
	var generateTile = randomValidTile();
	console.log(generateTile);
						// tile, dimensions, position, value
	awards.push(new Award(generateTile, [15,15], position(generateTile, [15, 15]), 3));
}

function toIndex(x, y)
{
	return((y * mapW) + x);
}

var computerMove = null;
window.onload = function()
{
	ctx = document.getElementById('game').getContext("2d");
	requestAnimationFrame(drawGame);
	ctx.font = "bold 10pt sans-serif";

	window.addEventListener("keydown", function(e) {
		if(e.keyCode>=37 && e.keyCode<=40) { //move players on key press
			human_player.keysDown[e.keyCode] = true;
			computer_player.resetKeyPress();
			computerMove = computer_controller.move();
			computer_player.keysDown[computerMove] = true;
		}
	});
	window.addEventListener("keyup", function(e) {
		if(e.keyCode>=37 && e.keyCode<=40) {
			human_player.keysDown[e.keyCode] = false;
			computer_player.keysDown[computerMove] = false;
		}
	});
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
		if(human_player.keysDown[38] && human_player.tileFrom[1]>0 && gameMap[toIndex(human_player.tileFrom[0], human_player.tileFrom[1]-1)]==1) { human_player.tileTo[1]-= 1; }
		else if(human_player.keysDown[40] && human_player.tileFrom[1]<(mapH-1) && gameMap[toIndex(human_player.tileFrom[0], human_player.tileFrom[1]+1)]==1) { human_player.tileTo[1]+= 1; }
		else if(human_player.keysDown[37] && human_player.tileFrom[0]>0 && gameMap[toIndex(human_player.tileFrom[0]-1, human_player.tileFrom[1])]==1) { human_player.tileTo[0]-= 1; }
		else if(human_player.keysDown[39] && human_player.tileFrom[0]<(mapW-1) && gameMap[toIndex(human_player.tileFrom[0]+1, human_player.tileFrom[1])]==1) { human_player.tileTo[0]+= 1; }

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

	var gameEnd = false;
	for(award of awards) {
		if(award.tile === human_player.tileFrom) {
			human_player.score = human_player.score + award.value;
			awards.pop(award);
			if(human_player.tileFrom == computer_player.tileFrom) {
				computer_player.score = computer_player.score + award.value;
			}
		}
		if(award.tile === computer_player.tileFrom) {
			computer_player.score = computer_player.score + award.value;
			awards.pop(award);
			if(human_player.tileFrom == computer_player.tileFrom) {
				human_player.score = human_player.score + award.value;
			}
		}
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

	ctx.fillStyle = "#FF5050"; // human color: red
	ctx.fillRect(human_player.position[0], human_player.position[1],
		human_player.dimensions[0], human_player.dimensions[1]);
	
	ctx.fillStyle = "#0000ff"; // computer color: blue
	ctx.fillRect(computer_player.position[0], computer_player.position[1],
		computer_player.dimensions[0], computer_player.dimensions[1]);

	ctx.fillStyle = "#000000"; // title color: black
	ctx.fillText("FPS: " + framesLastSecond, 10, 20);
    ctx.fillText("Red score: " + human_player.score, 110, 20);
	ctx.fillText("Blue score: " + computer_player.score, 210, 20);

	lastFrameTime = currentFrameTime;
	requestAnimationFrame(drawGame);
}
