class PlayerController{
    // players_controlled = []; // option for more than one agent
    constructor(player) {
        // this.players_controlled.push(player);
        this.player_controlled = player;
    }
    move(state) {
        // if(Object.values(this.player_controlled.keysDown).some(item => item.name === true)) {return };

        var actions = Object.keys(this.player_controlled.keysDown).map((i) => Number(i));
        var randomAction = actions[actions.length * Math.random() | 0];
//         while(!checkAction(randomAction, "blue")) { //find valid action 
//             randomAction = actions[actions.length * Math.random() | 0];
//         }
        console.log("randomAction: "+ randomAction);
        return randomAction;
    }
}