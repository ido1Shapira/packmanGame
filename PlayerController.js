class PlayerController{
    // players_controlled = []; // option for more than one agent
    constructor(player, type) {
        //5 kinds of type:
        // 1. random - Moves randomly
        // 2. selfish - stay in place
        // 3. farthest - Moves to the farthest award that its still closer than the other agent
        // 4. closest - Go to the closest award
        // 5. TSP - Moves by the solution of the TSP problem
        
        // this.players_controlled.push(player);
        this.player_controlled = player;
    }
    move(state) {
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
}