// https://github.com/rativardhan/Shortest-Path-2D-Matrix-With-Obstacles

class ShortestPathAlgo {
	constructor(b) {
		this.boardWithoutTarget = b.map(function(arr) {
			return arr.slice();
		});
	}
	getMinDistance() {
		return this.shortestPathLength - 1;
	}
	getSortestPath() {
		return this.flipCoordinates(this.shortestXY);
	}
	setTarget(x_target, y_target) {
		this.board[x_target][y_target] = 9;
	}
	clear() {
		this.board = this.boardWithoutTarget.map(function(arr) {
			return arr.slice();
		});
		this.allObstacle2DMatrix = []; // All Possible path in 2D Matrix Form
		this.duplicateVirtualPath = []; // A virtual 2D matrix path to find the shortest path
		this.allObstacleXY = []; // A coordinates XY path collections  for all possible obstacle Path
		this.currentObstacleXY = []; // Current obstacle path XY coordinates
		this.shortestPathLength = '';  // shortest path length
		this.shortestXY = []; // Shortest path coordinates
	}
	run(current_pos, target_pos) {
		this.clear();
		this.setTarget(target_pos[1], target_pos[0]);
		this.shortestPathLengthProcess(current_pos[1], current_pos[0]);
	}
	
	flipCoordinates(path) {
		var flip = [];
		for(var pos of path) {
			flip.push([pos[1], pos[0]]);
		}
		return flip
	}

	/**
	 * Function to find the shortest path, All X, Y Coordinates
	 * @param x
	 * @param y
	 */
	shortestPathLengthProcess(x,y)
	{
		// Assign 0 for all X, Y coordinates of virtual 2D Matrix
		for(var i in this.board)
		{
			var temp = [];
			for(var j in this.board[i])
			{
				temp.push(0);
			}
			this.duplicateVirtualPath.push(temp);
		}
		this.shortestObstaclePath(x,y)
		this.shortestPath(this.allObstacleXY);
	}
	/**
	 * Recursive function using backtracking to find shortest path coordinates and related information.
	 * @param x
	 * @param y
	 * @returns {boolean}
	 */
	shortestObstaclePath(x,y)
	{
		// Check if Value is not 1 and 0 and it is an obstacle, Store result in all solution array
		if(this.board[x][y] !== 1 && this.board[x][y] !== 0)
		{
			// We found obstacle
			// Because Javascript work as Pass by reference and to a
				// Javascript equal to= operator use pass by reference, so we have deep copy all value of an array
			var temp2D = [];
			for(var i in this.duplicateVirtualPath)
			{
				var temp = [];
				for(var j in this.duplicateVirtualPath[i])
				{
					temp.push(this.duplicateVirtualPath[i][j]);
				}
				temp2D.push(temp);
			}
			temp2D[x][y] = this.board[x][y];
			this.allObstacle2DMatrix.push(temp2D);
			// Because Javascript work as Pass by reference and to a
				// Javascript equal to= operator use pass by reference, so we have deep copy all value of an array
			var tempXY = [];
			for(var i in this.currentObstacleXY)
			{
				tempXY.push(this.currentObstacleXY[i]);
			}
			tempXY.push([x,y]);
			this.allObstacleXY.push(tempXY);
			return true;
		}
		// If we already traverse through this path, avoid the infinite loop of Matrix
		// example : 00,01,10,11 is infinite loop
		// [
		// [1, 1, 0]
		// [1, 1, 0]
		// ]
		if(this.duplicateVirtualPath[x][y] == 1)
		{
			return false;
		}
		// Store Path coordinates X, Y, And mark Virtual Path 2D Matrix  with 1
		this.currentObstacleXY.push([x,y]);
		this.duplicateVirtualPath[x][y] = 1;
		// We can take 4 path; Top, Right, Bottom, Left
		// If you need Directional Path, Please store TOP, RIGHT, Bottom, Left in array
		// console.log('TOP')
		if(x-1 > 0 && typeof this.board[x-1][y] != 'undefined' && this.board[x-1][y] != 0) // TOP
		{
			this.shortestObstaclePath(x-1,y);
		}
		//console.log('RIGHT')
		if(y+1 < this.board[x].length  &&  typeof this.board[x][y+1] != 'undefined' && this.board[x][y+1] != 0) // RIGHT
		{
			this.shortestObstaclePath(x,y+1);
		}
		//console.log('BOTTOM')
		if(x+1 < this.board.length &&  typeof this.board[x+1][y] != 'undefined' && this.board[x+1][y] != 0) // BOTTOM
		{
			this.shortestObstaclePath(x+1,y);
		}
		//console.log('LEFT')
		if(y-1>0 && this.board[x].length > y-1 &&  typeof this.board[x][y-1] != 'undefined'  && this.board[x][y-1] != 0) // LEFT
		{
			this.shortestObstaclePath(x,y-1);
		}
		// Backtracking remove remove and replace last info
		this.currentObstacleXY.pop();
		this.duplicateVirtualPath[x][y] = 0;
		return true;
	}
	shortestPath(allObstacleXY)
	{
		for(var i in allObstacleXY)
		{
			if(this.shortestPathLength == '')
			{
				this.shortestXY = allObstacleXY[i];
				this.shortestPathLength = allObstacleXY[i].length;
			}
			else if(this.shortestPathLength > allObstacleXY[i].length)
			{
				this.shortestXY = allObstacleXY[i];
				this.shortestPathLength = allObstacleXY[i].length;
			}
		}
	}

}