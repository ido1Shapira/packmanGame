
// https://www.geeksforgeeks.org/travelling-salesman-problem-greedy-approach/

class TSPGreedy {
 
    constructor() {
        
    }
    // Function to find the minimum
    // cost path for all the paths
    findMinRoute(tsp)
    {
        console.log(tsp);
        var sum = 0;
        var counter = 0;
        var j = 0, i = 0;
        var min = Number.NEGITIVE_INFINITY;
        var visitedRouteList = [];
 
        // Starting from the 0th indexed
        // city i.e., the first city
        visitedRouteList.push(0);
        var route = [];
 
        // Traverse the adjacency
        // matrix tsp[][]
        while (i < tsp.length && j < tsp[i].length) {
            // Corner of the Matrix
            if (counter >= tsp[i].length - 1) {
                break;
            }
            // If this path is unvisited then
            // and if the cost is less then
            // update the cost
            if (j != i
                && !(j in visitedRouteList)) {
                if (tsp[i][j] < min) {
                    min = tsp[i][j];
                    route[counter] = j + 1;
                    console.log("route: " + route);
                }
            }
            j++;
            // Check all paths from the
            // ith indexed city
            if (j == tsp[i].length) {
                sum += min;
                min = Number.POSITIVE_INFINITY;
                visitedRouteList.push(route[counter] - 1);
                j = 0;
                i = route[counter] - 1;
                console.log("counter: " + counter);
                console.log("route[counter]: " + route[counter]);
                console.log("i: "+ i);
                counter++;
            }
        }
        // Update the ending city in array
        // from city which was last visited
        i = route[counter - 1] - 1;
        for (var j = 0; j < tsp.length; j++) {
            console.log("i: "+ i);
            console.log("j: "+ j);
            if ((i != j) && tsp[i][j] < min) {
                min = tsp[i][j];
                route[counter] = j + 1;
                console.log("route: " + route);
            }
        }
        sum += min;
        // Started from the node where
        // we finished as well.
        console.log("Minimum Cost is : ");
        console.log(sum);
    }
}

// Input Matrix
var tsp = [
[ -1, 10, 15, 20 ],
[ 10, -1, 35, 25 ],
[ 15, 35, -1, 30 ],
[ 20, 25, 30, -1 ]
];

var tsp_solver = new TSPGreedy();
// Function Call
tsp_solver.findMinRoute(tsp);