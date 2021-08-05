// class not done
class Environment {
    gameMap = [
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
    gameStatus = {
        inProgress: true,
        ended: false,
        stoped: false
    }
    numOfAwards = 5;
    
    tileW = 40
    tileH = 40;
    mapW = 10
    mapH = 10;
    constructor(human_player, computer_player) {
        this.human_player = human_player;
        this.computer_player = computer_player;
        console.log(this.computer_player);
        this.awards = [];

        var validTiles = [];
        var indexs = []; // find valid indexes
        this.gameMap.filter(function(elem, index, array){
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
        // choose randomly numOfAwards indexs
        for (var i = 0; i < this.numOfAwards; i++) {
            var idx = Math.floor(Math.random() * indexs.length);
            validTiles.push(indexs[idx]);
            indexs.splice(idx, 1);
        }
        for(var i=0; i<this.numOfAwards; i++) {
	        this.awards.push(new Award(validTiles[i], [15,15], this.position(validTiles[i], [15, 15]), 2));
        }
        this.prev_state = [[],[],[],[],[],[]];
    }
    
    position(tile, dimensions)
    {
        return [Math.round((tile[0] * this.tileW) + ((this.tileW-dimensions[0])/2)),
                Math.round((tile[1] * this.tileH) + ((this.tileH-dimensions[1])/2))];
    }

    getBoardState() {
        // state is 6d array contains:
        // the board
        // human trace
        // computer trace
        // awards collected by human
        // awards collected by computer
        // all awards
        var state = [[],[],[],[],[],[]];
        var board = this.gameMap.slice();
        board[((this.human_player.tileFrom[1]*this.mapW)+this.human_player.tileFrom[0])] = 'h';
        board[((this.computer_player.tileFrom[1]*this.mapW)+this.computer_player.tileFrom[0])] = 'c';
        for(award of this.awards) {
            board[((award.tile[1]*this.mapW)+award.tile[0])] = 'a';
        }
        while(board.length) state[0].push(board.splice(0,10)); // reshape board
        return state;
    }

}