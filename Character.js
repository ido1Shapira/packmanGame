class Character{
    keysDown = {
        37 : false, //left
        38 : false, //up
        39 : false, //right
        40 : false, //down
        32 : false //enter (stay)
    };
    scores = {
        stay : -1,
        step : -2,
        collide : 0,
        finish : 100,
    }
    constructor(tileFrom , tileTo, timeMoved, dimensions, position, delayMove) {
        this.tileFrom	= tileFrom;
        this.tileTo		= tileTo;
        this.timeMoved	= timeMoved;
        this.dimensions	= dimensions;
        this.position	= position;
        this.delayMove	= delayMove;
        this.score = 50;
    }
    placeAt(x,y) {
        this.score = this.score + this.scores.step;

        this.tileFrom	= [x,y];
	    this.tileTo		= [x,y];
	    this.position	= [((tileW*x)+((tileW-this.dimensions[0])/2)),
		    ((tileH*y)+((tileH-this.dimensions[1])/2))];
    }
    processMovement(t) {
        if(this.tileFrom[0]==this.tileTo[0] && this.tileFrom[1]==this.tileTo[1]) { return false; }

        if((t-this.timeMoved)>=this.delayMove)
        {
            this.placeAt(this.tileTo[0], this.tileTo[1]);
        }
        else
        {
            this.position[0] = (this.tileFrom[0] * tileW) + ((tileW-this.dimensions[0])/2);
            this.position[1] = (this.tileFrom[1] * tileH) + ((tileH-this.dimensions[1])/2);

            if(this.tileTo[0] != this.tileFrom[0])
            {
                var diff = (tileW / this.delayMove) * (t-this.timeMoved);
                this.position[0]+= (this.tileTo[0]<this.tileFrom[0] ? 0 - diff : diff);
            }
            if(this.tileTo[1] != this.tileFrom[1])
            {
                var diff = (tileH / this.delayMove) * (t-this.timeMoved);
                this.position[1]+= (this.tileTo[1]<this.tileFrom[1] ? 0 - diff : diff);
            }

            this.position[0] = Math.round(this.position[0]);
            this.position[1] = Math.round(this.position[1]);           
        }

        return true;

    }
    resetKeyPress() {
        this.keysDown = {
            37 : false, //left
            38 : false, //up
            39 : false, //right
            40 : false, //down
            32 : false //stay
        };
    }
}