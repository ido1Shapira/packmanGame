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
                console.log("im here");
        }
    }

    validAction(action, board) {
        switch(action) {
            case 38: //left
                return this.player_controlled.tileFrom[1]>0 && board[toIndex(this.player_controlled.tileFrom[0], this.player_controlled.tileFrom[1]-1)]!=0;
            case 40: //up
                return this.player_controlled.tileFrom[1]<(mapH-1) && board[toIndex(this.player_controlled.tileFrom[0], this.player_controlled.tileFrom[1]+1)]!=0;
            case 37: //right
                return this.player_controlled.tileFrom[0]>0 && board[toIndex(this.player_controlled.tileFrom[0]-1, this.player_controlled.tileFrom[1])]!=0;
            case 39: //down
                return this.player_controlled.tileFrom[0]<(mapW-1) && board[toIndex(this.player_controlled.tileFrom[0]+1, this.player_controlled.tileFrom[1])]!=0;
            case 32:
                return true;
            default:
                return false;
        }
    }
    whereis(sub_state) {
        var indexs = [];
        sub_state.filter(function(row, i){
            row.filter(function(row, j) {
                if(row == 1) {
                    indexs.push([i,j]);
                }
            });
        });
        return indexs;
    }
    distance(from, to) {
        //number of moves to get from 'from' to 'to'
        return Math.abs(from[0] - to[0]) + Math.abs(from[1] - to[1]);
    }
    takeActionTo(from, to) {
        // maybe use here astar to find the shortest path:
        // https://github.com/bgrins/javascript-astar
        // or this:
        // https://github.com/rativardhan/Shortest-Path-2D-Matrix-With-Obstacles
        console.log("from: "+ from);
        console.log("to: "+ to);
        
        // need to check that the action is valid
        if(from[0] < to[0]) {
            return 37; //left
        }
        else if(from[0] > to[0]) {
            return 39; //right
        }
        if(from[1] < to[1]) {
            return 38; //up
        }
        else {
            return 40; //down
        }

    }
    ////////////////////////////// All baselines ////////////////////////////////////////////////
    random(state) {
        console.log(state);
        var actions = Object.keys(this.player_controlled.keysDown).map((i) => Number(i));
        var valid_actions = [];
        for(var i=0; i<actions.length; i++) {
            if(this.validAction(actions[i], state[0])) {
                valid_actions.push(actions[i]);
            }
        }
        var randomAction = valid_actions[Math.floor(valid_actions.length * Math.random())];
        return randomAction;
    }
    selfish(state) {
        return 32;
    }
    closest(state) {
        var player_pos = this.player_controlled.tileFrom;
        var all_awards_positions = this.whereis(state[5]); // whereis all awards
        var min_d = all_awards_positions[0];
        for (var award_pos of all_awards_positions) {
            var d = this.distance(player_pos, award_pos);
            console.log("d: "+ d);
            // what happen when min_d == d ? maybe to consider the human player here
            // TODO: ask Amos
            if(d < min_d) {
                min_d = d;
            }
        }
        return this.takeActionTo(player_pos, min_d);
    }
    farthest(state) {
        // var award_map = 
        // for()
        
    }
    TSP(state) {
        
    }
    mix(state) {
        var baselines = Object.keys(this.TYPES);
        const index = baselines.indexOf("mix");
        if (index > -1) {
            baselines.splice(index, 1);
        }
        console.log(baselines);
    }
    
}