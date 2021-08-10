class PlayerController{
    TYPES = {
        "random": false,
        "selfish": false,
        "farthest": false,
        "closest": false,
        "TSP": false,
        "mix": false
    }
    // players_controlled = []; // option for more than one agent
    constructor(player, type) {
        //5 kinds of type:
        // 1. random - Moves randomly
        // 2. selfish - stay in place
        // 3. farthest - Moves to the farthest award that its still closer than the other agent
        // 4. closest - Go to the closest award
        // 5. TSP - Moves by the solution of the TSP problem
        // 6. mix of all controllers
        
        // this.players_controlled.push(player);
        this.TYPES[type] = true;
        this.type = type;
        this.player_controlled = player;
    }
    move(state) {
        switch(this.type) {
            case "random":
                return this.random(state);
            case "selfish":
                return this.selfish(state);
            case "closest":
                return this.closest(state);
            case "farthest":
                return this.farthest(state);
            case "TSP":
                return this.TSP(state);
            case "mix":
                return this.mix(state);
            default:
                throw "move(state): not a valid baseline"
        }
    }

    validAction(action, board) {
        // console.log(action);
        // console.log(this.player_controlled.tileFrom);
        switch(action) {
            case 38: //up
                return this.player_controlled.tileFrom[1]>0 && board[this.player_controlled.tileFrom[1]-1][this.player_controlled.tileFrom[0]]==1;
            case 40: //down
                return this.player_controlled.tileFrom[1]<(mapH-1) && board[this.player_controlled.tileFrom[1]+1][this.player_controlled.tileFrom[0]]==1;
            case 37: //left
                return this.player_controlled.tileFrom[0]>0 && board[this.player_controlled.tileFrom[1]][this.player_controlled.tileFrom[0]-1]==1;
            case 39: //right
                return this.player_controlled.tileFrom[0]<(mapW-1) && board[this.player_controlled.tileFrom[1]][this.player_controlled.tileFrom[0]+1]==1;
            case 32: //stay
                return true;
            default:
                return false;
        }
    }
    getValidActions(board) {
        console.log(board);
        var actions = Object.keys(this.player_controlled.keysDown).map((i) => Number(i));
        var valid_actions = [];
        for(var i=0; i<actions.length; i++) {
            if(this.validAction(actions[i], board)) {
                valid_actions.push(actions[i]);
            }
        }
        console.log("valid_actions: "+valid_actions)
        return valid_actions;
    }

    whereis(sub_state) {
        var indexs = [];
        sub_state.filter(function(row, i){
            row.filter(function(row, j) {
                if(row == 1) {
                    indexs.push([j,i]);
                }
            });
        });
        return indexs;
    }
    distance(from, to) {
        //number of moves to get from 'from' to 'to'
        // console.log("distance: " + (Math.abs(from[0] - to[0]) + Math.abs(from[1] - to[1])));
        return Math.abs(from[0] - to[0]) + Math.abs(from[1] - to[1]);
    }
    takeActionTo(from, to) {
        // take a action that make it close to the award

        // maybe use here astar to find the shortest path:
        // https://github.com/bgrins/javascript-astar
        // or this:
        // https://github.com/rativardhan/Shortest-Path-2D-Matrix-With-Obstacles
        if(from[1] < to[1]) {
            return 40 //down
        }
        else if(from[1] > to[1]){
            return 38 //up
        }
        else if(from[0] < to[0]) {
            return 39 //right
        }
        else if(from[0] > to[0]) {
            return 37 //left
        }
        throw "takeActionTo("+from +","+ to+ "): could not found the action";
    }

    ////////////////////////////// All baselines ////////////////////////////////////////////////
    random(state) {
        var valid_actions = this.getValidActions(state[0]);
        var randomAction = valid_actions[Math.floor(valid_actions.length * Math.random())];
        return randomAction;
    }
    selfish(state) {
        return 32;
    }
    closest(state) {
        var player_pos = this.player_controlled.tileFrom;
        var all_awards_positions = this.whereis(state[5]); // whereis all awards
        
        const SPA = new ShortestPathAlgo(state[0]);
        
        SPA.run(player_pos, all_awards_positions[0]);
        var min_d = SPA.getMinDistance();
        var min_path = SPA.getSortestPath();
        
        for (var i=1; i<all_awards_positions.length; i++) {
            var award_pos = all_awards_positions[i];
            SPA.run(player_pos, award_pos);
            var d = SPA.getMinDistance();
            // what happen when min_d == d ? maybe to consider the human player here
            // TODO: ask Amos
            if(d < min_d) {
                min_d = d;
                min_path = SPA.getSortestPath();
            }
        }
        return this.takeActionTo(min_path[0], min_path[1]);
    }
    farthest(state) {
        var player_pos = this.player_controlled.tileFrom;
        var all_awards_positions = this.whereis(state[5]); // whereis all awards
        var human_pos = this.whereis(state[1])[0];
        
        const SPA = new ShortestPathAlgo(state[0]);

        SPA.run(player_pos, all_awards_positions[0]);
        var max_d = SPA.getMinDistance();
        var max_path = SPA.getSortestPath();

        for (var i=1; i<all_awards_positions.length; i++) {
            var award_pos = all_awards_positions[i];
            SPA.run(player_pos, award_pos);
            var d_c = SPA.getMinDistance(); //computer distance
            SPA.run(human_pos, award_pos);
            var d_h = SPA.getMinDistance(); //human distance
            if(d_c < d_h) {
                //Check if the computer comes before the human 
                if(d_c > max_d) {
                    max_d = d_c;
                    max_path = SPA.getSortestPath();
                }
            }
        }
        return this.takeActionTo(max_path[0], max_path[1]);
    }
    TSP(state) {
        var player_pos = this.player_controlled.tileFrom;
        var all_awards_positions = this.whereis(state[5]);
        all_awards_positions.unshift(player_pos);
        var map_cost = new Map();
        for(var point of all_awards_positions) {
            map_cost.set(point, new Map());
        }
        var permutations = generateCityRoutes(all_awards_positions);
        var min_cost = Number.POSITIVE_INFINITY;
        var awards_order = [];
        for(var path of permutations) {
            // console.log(p);
            var temp_cost = 0;
            for(var i=0; i<path.length-1; i++) {
                var point1 = path[i];
                var point2 = path[i+1];
                
                var cost = map_cost.get(point1).get(point2);
                if(cost === undefined) {
                    //insert cost
                    map_cost.get(point1).set(point2, this.distance(point1, point2));
                    cost = map_cost.get(point1).get(point2);
                }
                temp_cost += cost;
            }
            if(temp_cost < min_cost) {
                min_cost = temp_cost;
                awards_order = path;
            }
        }
        const SPA = new ShortestPathAlgo(state[0]);
        SPA.run(awards_order[0], awards_order[1]);
        var optimal_path = SPA.getSortestPath();
        return this.takeActionTo(optimal_path[0], optimal_path[1]);
    }
    mix(state) {
        var baselines = Object.keys(this.TYPES);
        const index = baselines.indexOf("mix");
        if (index > -1) {
            baselines.splice(index, 1);
        }
        console.log(baselines);
        var random_baseline = baselines[Math.floor(baselines.length * Math.random())];
        switch(random_baseline) {
            case "random":
                return this.random(state);
            case "selfish":
                return this.selfish(state);
            case "closest":
                return this.closest(state);
            case "farthest":
                return this.farthest(state);
            case "TSP":
                return this.TSP(state);
        }
    }
    
}