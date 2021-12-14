function getDOM(id) {
    return document.getElementById(id);
}


///////////////////////////////// buttons /////////////////////////////////////////////////

// Display the quiz
function showQuiz() {
    var quiz = getDOM("quiz");
    quiz.style.display = "block";
    var span = document.getElementsByClassName("close")[0];
    span.onclick = function() {
        quiz.style.display = "none";
        getDOM("showgameButton").style.display = "";
    }
    var returnToInstructions = getDOM("returnToInstructions");
    returnToInstructions.onclick = function() {
        quiz.style.display = "none";
        getDOM("showgameButton").style.display = "";
    }
    getDOM("quiz").style.display = "";
}

function checkQuiz() {
    var q1 = getRating("q1");
    var q2 = getRating("q2");
    var q3 = getRating("q3");

    if(q1 === undefined || q2 === undefined || q3 === undefined) {
            getDOM("notFillAll1").innerHTML = "Did not submit, please fill all fields."
            getDOM("notFillAll1").style.display = "";
    }
    else {
        if(q1 == "true" && q2 == "true" && q3 == "true") {
            getDOM("continueToGame").style.display = "";
        }
        else {
            getDOM("notFillAll1").innerHTML = "At least one answer is incorrect, please read the instructions again."
            getDOM("notFillAll1").style.display = "";
        }
    }
}

function startGame() {
    window.addEventListener("keyup", handleKeyUp);
    getDOM("quiz").style.display = "none";
    getDOM("showgameButton").style.display = "none";
    getDOM("showgameLabel").style.display = "none";

    getDOM("left").setAttribute("style","width:25%;font-size: 11pt;");
}

function submitSurvey() {
    //Get values
    var birthYear = getDOM("yBirth").value;
    var gender = getDOM("gender").value;
    var education = getDOM("education").value;
    
    var selfishly_value = getRating("selfishly_rating");
    var collaborative_value = getRating("collaborative_rating");
    var wisely_value = getRating("wisely_rating");
    var computer_value = getRating("computer_rating");
    var predictable_value = getRating("predictable_rating");

    var additional_comments = getDOM("additionalcomments").value;

    if(birthYear == '' || gender == '' || education == '' ||
        selfishly_value === undefined || collaborative_value === undefined ||
        wisely_value === undefined || computer_value === undefined || predictable_value === undefined) {
            getDOM("notFillAll2").innerHTML = "Did not submit, please fill all fields."
            getDOM("notFillAll2").style.display = "";
    }
    else if(birthYear < 1930 || birthYear > 2021) {
        getDOM("notFillAll2").innerHTML = "Enter your vallid birth of year."
        getDOM("notFillAll2").style.display = "";
    }
    else {
        firebase.database().ref("all-games/"+postID+"/birth_year").set(birthYear);
        firebase.database().ref("all-games/"+postID+"/gender").set(gender);
        firebase.database().ref("all-games/"+postID+"/education").set(education);
        firebase.database().ref("all-games/"+postID+"/selfishly_value").set(selfishly_value);
        firebase.database().ref("all-games/"+postID+"/collaborative_value").set(collaborative_value);
        firebase.database().ref("all-games/"+postID+"/wisely_value").set(wisely_value);
        firebase.database().ref("all-games/"+postID+"/predictable_value").set(predictable_value);
        firebase.database().ref("all-games/"+postID+"/computer_value").set(computer_value);
        firebase.database().ref("all-games/"+postID+"/additional_comments").set(additional_comments);

        // var leadsRef = firebase.database().ref("all-games/"+postID);
        // leadsRef.once('value', function(snapshot) {
        //     snapshot.forEach(function(childSnapshot) {
        //         var childKey = childSnapshot.key;
        //         var childData = childSnapshot.val();
        //         firebase.database().ref("complete-games/"+postID+"/"+childKey).set(childData);
        //     });
        // });
        survey.style.display = "none";
        
        // what to do when anwers the survey:
        survey.style.display = "none";
        // getDOM("instructions-h").innerHTML = "Thank you for your participating!";
        getDOM("container").style.display = "none"
        // getDOM("game").style.display = "none";
        getDOM("end-game-code-div").style.display = "";
        getDOM("code").value = postID.substring(1) + "ido";
    }
}

function copytoclipboard() {
    var copyText = getDOM("code");
    copyText.select();
    copyText.setSelectionRange(0, 99999);
    document.execCommand("copy");    
    getDOM("myTooltip").innerHTML = "Copied: " + copyText.value;
}

function getRating(s_rating) {
    var rating = document.getElementsByName(s_rating);
    for(var i = 0; i < rating.length; i++){
        if(rating[i].checked){
            return rating[i].value;
        }
    }
    return undefined;
}

///////////////////////////////// firebase /////////////////////////////////////////////////
let postID;
let steps = 1;

function finishGame(state, action) { //update database
    saveToFirebase(state, action); //saving the last state

    firebase.database().ref("all-games/"+postID+"/human_score").set(human_player.score.toFixed(3));
    firebase.database().ref("all-games/"+postID+"/computer_score").set(computer_player.score.toFixed(3));
    // Get the survey
    getDOM("survey_title").innerHTML = "Well done, you reached your destination safely!<br>Your score is: "+human_player.scoreToView + " point(s).\n"+ "<br>Please fill the following survey:";
    getDOM("survey").style.display = "block";
    keyEnable = false;
}

function saveToFirebase(state, humanMove) {
    // var url=canvas.toDataURL("image/png").replace("image/png", "image/octet-stream")
	// var img=new Image();
	// img.src=url
    // firebase.database().ref("all-games/"+postID+"/img/"+steps).set({
    //     img
    // });

    // firebase.database().ref("all-games/"+postID+"/log/"+steps).set({
    //     state
    // });
    firebase.database().ref("humanModel/"+postID+"/"+steps).set({
        state: state,
        action: humanMove
    });
    // firebase.database().ref("all-games/"+postID+"/log/"+steps).set({
    //     "board": state[0],
    //     "human_trace": state[1],
    //     "computer_trace": state[2],
    //     "human_rewards": state[3],
    //     "computer_rewards": state[4],
    //     "all_rewards": state[5]
    // });
    steps++;
}

function initializeFirebase() {
    // Firebase configuration
    var firebaseConfig = {
        apiKey: "AIzaSyAy6TmnVcLWjkpSpQFtCnX-PVGignQFsiw",
        authDomain: "packman-game.firebaseapp.com",
        projectId: "packman-game",
        storageBucket: "packman-game.appspot.com",
        messagingSenderId: "819894936980",
        appId: "1:819894936980:web:eef6650be9f4caa0d81b81"
    };
    // Initialize Firebase
    firebase.initializeApp(firebaseConfig);
}