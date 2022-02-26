# Rock-Scissors-Paper

### Sketch

This project uses [leap motion](https://developer.leapmotion.com/) to recognize our gesture(Rock / Scissors / Paper), and it was used as an interactive game once a time at the entrance of **No.5 courtyard of Jingyuan, Peking University**.

As we must provide API to leap motion, ./protos uses the open-source code of [tensorflow models](https://github.com/tensorflow/models).

### Structure

    /image ---- Pictures of the game.
    
    /music ---- Music of the game.
    
    /hand_classifier ---- Open-source code of tensorflow（including optimization of parameters)
    
    master.py ---- The entrance of the game.
    
    game.py  ---- Details of internal implementation during the course of a game.
    
    hand-recogniter.py ---- Classifier of Rock / Scissors / Paper.
    
### Demonstration

