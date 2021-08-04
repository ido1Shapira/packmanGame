class PlayerController{
    // players_controlled = []; // option for more than one agent
    constructor(player) {
        // this.players_controlled.push(player);
        this.player_controlled = player;
    }
    move(state) {
        var actions = Object.keys(this.player_controlled.keysDown).map((i) => Number(i));
        var valid_actions = [];
        for(var i=0; i<actions.length; i++) {
            if(this.validAction(actions[i], state)) {
                valid_actions.push(actions[i]);
            }
        }
        var randomAction = valid_actions[Math.floor(valid_actions.length * Math.random())];
        return randomAction;
    }
    validAction(action, state) {
        switch(action) {
            case 38: //left
                return this.player_controlled.tileFrom[1]>0 && state[toIndex(this.player_controlled.tileFrom[0], this.player_controlled.tileFrom[1]-1)]!=0;
            case 40: //up
                return this.player_controlled.tileFrom[1]<(mapH-1) && state[toIndex(this.player_controlled.tileFrom[0], this.player_controlled.tileFrom[1]+1)]!=0;
            case 37: //right
                return this.player_controlled.tileFrom[0]>0 && state[toIndex(this.player_controlled.tileFrom[0]-1, this.player_controlled.tileFrom[1])]!=0;
            case 39: //down
                return this.player_controlled.tileFrom[0]<(mapW-1) && state[toIndex(this.player_controlled.tileFrom[0]+1, this.player_controlled.tileFrom[1])]!=0;
            case 32:
                return true;
            default:
                return false;
        }
    }
}