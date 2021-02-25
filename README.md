# QuantumChessEngine

[Medium article: Asking a Quantum Computer To Learn Chess](https://gabriel-mueller.medium.com/asking-a-quantum-computer-to-learn-chess-8670891770a0?sk=05ee4f50cb5f4ce868a686c170a3ca3d)

The main goals of this project are the visualization of the main principles of the quantum computer as well as to demonstrate its performance. To show this, we apply our quantum algorithms to one of the most famous board games of all time, namely chess. We construct a variational algorithm to classify the best possible moves in a game of [Microchess](https://en.wikipedia.org/wiki/Minichess).

Microchess features a 5x4 board on which 5 of the 6 different chess pieces are placed (there are still trillions of possible constellations). Thus, it makes it perfectly suitable for a quantum computer as it can be implemented using only 21 qubits (20 + 1 ancillary qubit). The piece of each square is encoded into a single qubit showing its remarkable capability of representing much more than the classical bit could do. We then construct a variational quantum algorithm (VQA) to operate on these qubits to classify the board's score. The score is determined by the mobility (how many moves are possible) and material (how many pieces are there) of each player. The output of the circuit is received through a single measurement collapsing all the information of the circuit into a single classical bit. Repeating this measurement gives a good estimate of the board's value. To learn how to quantify the given board, we train the VQA using a special combination of gradient descent and reinforcement learning.

## Requirements
<code>pip install pennylane</code><br>
<code>pip install pennylane-cirq==0.14.0</code><br>
<code>pip install python-chess</code>


Optional:
Install [stockfish](https://stockfishchess.org/download/).
