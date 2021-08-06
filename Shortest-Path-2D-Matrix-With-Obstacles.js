// https://github.com/rativardhan/Shortest-Path-2D-Matrix-With-Obstacles

var dupLot = [];
var allPath = [];
var obstacleDistance = 0;
var foundObstacleDistance = -1;
function removeObstacle(numRows,numCol, lot)
{

	for(var i in lot)
    {
		var temp = [];
		for(var j in lot[i])
        {
			temp.push(0)
		}
		dupLot.push(temp)
	}
	
	findShortestPath(0,0,lot)
}
function findShortestPath(x,y,lot)
{
	obstacleDistance++;
	
	if(obstacleDistance > foundObstacleDistance && foundObstacleDistance!= -1)
    {
		obstacleDistance--;
		return false;
    }
	if(lot[x][y] !== 0 && lot[x][y] != 1)
    {
		 // find obstacle 
		foundObstacleDistance = obstacleDistance;
		
		var temp2D = [];

		for (var i = 0; i < dupLot.length; i++)
    	temp2D[i] = dupLot[i].slice();
		
		temp2D[x][y] = lot[x][y];
		allPath.push(temp2D)

		obstacleDistance--;
		return true;
	}
	if(dupLot[x][y] == 1)
    {
		obstacleDistance--;
		return false; // already visited node
	}
	dupLot[x][y] = 1;
	
	// Top 
	if(typeof lot[x-1] != 'undefined' && typeof lot[x-1][y] != 'undefined' && lot[x-1][y] !==0 )
	findShortestPath(x-1,y,lot)

	// Right 
	if(typeof lot[x] != 'undefined' && typeof lot[x][y+1] != 'undefined' && lot[x][y+1] !==0 )
	findShortestPath(x,y+1,lot)

	// Bottom 
	if(typeof lot[x+1] != 'undefined' && typeof lot[x+1][y] != 'undefined' && lot[x+1][y] !==0 )
	findShortestPath(x+1,y,lot)

	// LEFT 
	if(typeof lot[x] != 'undefined' && typeof lot[x][y-1] != 'undefined' && lot[x][y-1] !==0 )
	findShortestPath(x,y-1,lot)

	obstacleDistance--;
	dupLot[x][y] = 0;
}
removeObstacle(3, 3, [
        [1,1,1],
        [1,0,1,0],
        [1,0,1,0],
        [1,0,9,0],
        [1,0,1,0,0],
        [1,1,1,0,0],
        [1,0,0,0,0],
        [0,0,0,0,0]
    ])
console.log(allPath)
console.log(foundObstacleDistance-1) // Because we are including obstacle start and end both