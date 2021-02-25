import remote_cirq
import pennylane as qml
import numpy as np
import chess 
import math
from minichess import *
import plotting as plt

# FLOQ setup
FLOQ_API_KEY = "API_KEY"
SIM = remote_cirq.RemoteSimulator(FLOQ_API_KEY)

NUM_WIRES = MICROCHESS_SIZE[0]*MICROCHESS_SIZE[1] + 1

# Device definition
floq=True
if floq:
    # floq
    DEV = qml.device('cirq.simulator', wires=NUM_WIRES, simulator=SIM, analytic=False)
else:
    # pennylane
    DEV = qml.device('default.qubit', wires=NUM_WIRES)

LAYERS = 1
CURRENT_BOARD = []
CIRCUIT = None
TURN = 'white'

def get_rotation_angles(turn='white', params=None):
    global CURRENT_BOARD
    board = CURRENT_BOARD
    if params is None:
        params = [
            # Rx, Rz
            # Own pieces
            np.pi*3/8,np.pi, # PAWN (1)
            np.pi*2/8,np.pi*3/2, # KNIGHT (3)
            np.pi*2/8,np.pi*1/2, # BISHOP (3)
            np.pi*1/8,np.pi, # ROOK -> (5)
            0, 0, # QUEEN -> not needed
            0, 0, # KING -> exp_val(PauliZ) = 1
            # Opponent's pieces
            np.pi*5/8,np.pi, # PAWN (1)
            np.pi*6/8,np.pi*3/2, # KNIGHT (3)
            np.pi*6/8,np.pi*1/2, # BISHOP (3)
            np.pi*7/8,np.pi, # ROOK -> (5)
            0,0, # QUEEN -> not needed
            np.pi, 0, # KING -> exp_val(PauliZ) = -1
            # Other
            np.pi/2, 0 # empty square
            ]
    
    angles_dict = {
        'P': 0,
        'N': 1,
        'B': 2,
        'R': 3,
        'Q': 4,
        'K': 5
    }
    
    def criteria_turn(p):
        if turn == 'white':
            return 6 if p.islower() else 0
        return 6 if p.isupper() else 0
    
    pieces = [angles_dict.get(p.upper()) + criteria_turn(p) if p != '' else 12 for p in board]
     
    return [[params[2*p-1], params[2*p]] for p in pieces]
    
def initialize_fields(angles, wires):
    qml.broadcast(qml.RX, wires=wires, pattern='single', parameters=np.array(angles)[:,0])
    qml.broadcast(qml.RZ, wires=wires, pattern='single', parameters=np.array(angles)[:,1])

def analysis_circuit(params, wires, layers, circuit):
    # Possible circuit functions:
    # 1. random_circuit
    # 2. qaoa_circuit
    circuit(params, wires, layers)
        
def random_circuit(params, wires, layers):
    layer_params = np.split(params, layers)
    qml.templates.layers.RandomLayers(layer_params, wires)
    
def qaoa_circuit(params, wires, layers):
    cost_params, mixer_params = np.split(params, 2)
    cost_h = qml.qaoa.bit_driver(wires, b=1)
    mixer_h = qml.qaoa.x_mixer(wires)
    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)
    qml.layer(qaoa_layer, layers, cost_params, mixer_params)
        
def evaluate_to_ancillary(params, wires):
    unitary_params = np.split(params, len(wires)-1)
    pattern = [[i,wires[-1]] for i in wires[:-1]]
    if floq:
        for i,used_wires in enumerate(pattern):
            CRX(params[i], used_wires)
    else:
        qml.broadcast(unitary=qml.CRX, pattern=pattern, wires=wires, parameters=unitary_params)

# own implementation of a controlled RX gate (only necessary when using floq)
def CRX(param, wires):
    qml.RZ(-np.pi/2, wires=wires[1])
    # Identity on qubit 1
    qml.CNOT(wires=wires)
    qml.RY(-param/2, wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RY(param/2, wires=wires[1])
    # Identity on qubit 1
    qml.RZ(np.pi/2, wires=wires[1])
    # Identity on qubit 1

@qml.qnode(DEV)
def full_circuit(params):
    # Initialize board
    board_init = get_rotation_angles(turn=TURN)
    initialize_fields(board_init, range(NUM_WIRES-1))
    # Circuit to analyse given board
    if len(params) != NUM_WIRES-1:
        analysis_circuit(params[:-(NUM_WIRES-1)], range(NUM_WIRES-1), LAYERS, circuit=CIRCUIT)
        evaluate_to_ancillary(params[-(NUM_WIRES-1):], range(NUM_WIRES))
    else:
        evaluate_to_ancillary(params, range(NUM_WIRES))

    # Evaluate circuit, where probability 1 for state |1> is best
    return qml.probs([NUM_WIRES-1]) # -> probability for ['0','1'] of ancillary qubit

def circuit_score(params, legal_boards):
    global CURRENT_BOARD
    if not isinstance(legal_boards[0], list):
        legal_boards = [legal_boards]
    cost = 0
    for board in legal_boards:
        CURRENT_BOARD = board
        quantum_rating = full_circuit(params)[1] * 2 - 1
        classical_rating = MiniBoard(board=board).analyze_board()
        cost += (quantum_rating-classical_rating/10)**2
        print('Quantum: {}, Classical: {}, Cost: {}'.format(quantum_rating, classical_rating, cost))
    return cost/len(legal_boards)

def make_diff_params(p, i, step):
    new_p = p.copy()
    new_p[i] += step
    return new_p
    
class GradientDescent():
    def __init__(self, epsilon=np.pi, eta=0.1):
        self.epsilon = epsilon
        self.eta = eta
        
    def step(self, params, boards, circuit=circuit_score):
        new_params = params.copy()
        for i in range(len(params)):
            params_plus = make_diff_params(params, i, self.epsilon)
            params_minus = make_diff_params(params, i, -self.epsilon)
            cost_plus = circuit(params_plus, boards)
            cost_minus = circuit(params_minus, boards)
            param_step = (cost_minus - cost_plus) / (2*self.epsilon)
            new_params[i] += param_step * self.eta
            print('Param step: {} with +: {} and -: {}.'.format(param_step * self.eta, cost_plus, cost_minus))
        return new_params
    
    def step_and_cost(self, params, boards, circuit=circuit_score):
        new_params = self.step(params, boards, circuit)
        cost = circuit(new_params, boards)
        return new_params, cost
    
    def update_hyperparameters(epsilon=None, eta=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if eta is not None:
            self.eta = eta

def game_two_circuits(circuit1, circuit2, num_moves=10, num_train_epochs=10):
    global CIRCUIT
    global TURN
    global LAYERS
    board = MiniBoard(board_preset='Microchess')
    for move in range(num_moves):
        for turn in ['white', 'black']:
            legal_boards = board.legal_boards(turn=turn[0], print_boards=False)
            if len(legal_boards) == 0:
                    print('\n No legal moves: Draw or {} loses.\n'.format(turn))
                    return board
            turn_circuit = circuit1 if 'white' else circuit2
            params = turn_circuit.get('params')
            # SET GLOBAL VARIABLES
            TURN = turn
            CIRCUIT = turn_circuit.get('fun')
            LAYERS = turn_circuit.get('layers')
            # If num_train_epochs > 0: train parameters with gradient descent
            params, training_cost = train_own_game(board, params, epochs=num_train_epochs, legal_boards=legal_boards)
            turn_circuit['params'] = params
            # Choose next move by quantum evaluation of all possible boards
            board_after_move = choose_board(legal_boards, params)
            # Make move and set new board
            board.set_board(board_after_move)
            print("Move {}, {}'s turn".format(move, turn), board)
    # Return board after num_moves for each side
    return board

def train_circuit_against_itself(circuit, num_epochs, num_moves, epsilon=0.1, param_change_ratio=0.3, eta=0.1):
    filename_i = plt.filename_nmbr() # Set plotting filename
    param_list = [circuit.get('params')]
    for epoch in range(num_epochs):
        # Define two circuits in analogy to +/- epsilon (number of changing params w.r.t. param_change_ratio)
        circuit_plus = circuit.copy()
        circuit_minus = circuit.copy()
        params = circuit.get('params')
        param_change = [1 if np.random.uniform() < param_change_ratio else 0 for _ in params]
        circuit_plus['params'] = [param + epsilon * factor for param, factor in zip(params, param_change)]
        circuit_minus['params'] = [param - epsilon * factor for param, factor in zip(params, param_change)]
        # Execute game between these two circuits
        board = game_two_circuits(circuit_plus, circuit_minus, num_moves=num_moves, num_train_epochs=0)
        # Calculate gradient
        result = board.analyze_board()/(2*epsilon)
        # Update parameters accordingly
        new_params = [param + result * eta for param in params]
        circuit['params'] = new_params
        # PRINTS and BOOKKEEPING
        print('Epoch {}: Board result = {}'.format(epoch, result))
        param_list.append(new_params)
        print('New parameters:', new_params)
        plt.plot_params(param_list, "params-{}.pdf".format(filename_i))
        
        

def train_own_game(board, params, epochs=10, legal_boards=None, epsilon=0.1, eta=0.1):
    if epochs == 0: return params, None
    legal_boards = legal_boards or board.legal_boards(turn=turn[0], print_boards=False)
    # Define optimizer
    opt = GradientDescent(epsilon, eta)
    # Train the circuit
    for epoch in range(epochs):
        if epoch < epochs-1:
            params = opt.step(params, legal_boards, circuit_score)
        else:
            params, cost = opt.step_and_cost(params, legal_boards, circuit_score)
    return params, cost

def train_circuit(circuit, num_epochs=10, num_moves=5, num_train_epochs=5, max_legal_boards=2, epsilon=0.1, eta=0.1):
    global CIRCUIT
    global TURN
    global LAYERS
    # Plotting filename
    filename_i = plt.filename_nmbr()
    # Set global variables
    TURN = 'white'
    CIRCUIT = circuit.get('fun')
    LAYERS = circuit.get('layers')
    # Define initial params
    params = circuit.get('params')
    # list for BOOKKEEPING
    training_cost_list = []
    validation_cost_list = []
    param_list = []
    for epoch in range(num_epochs):
        # Generate inititial board
        board = MiniBoard(board_preset='Microchess')
        for move in range(num_moves):
            for turn in ['white', 'black']:
                legal_boards = board.legal_boards(turn=turn[0], print_boards=False)
                print("Number of legal boards for {}'s move #{}: {}".format(turn, move, len(legal_boards)))
                if len(legal_boards) == 0:
                    print('\n No legal moves: Draw or {} loses.\n'.format(turn))
                    break
                if turn == 'white':
                    # np.random.shuffle(legal_boards)
                    # Train parameters with gradient descent
                    params, training_cost = train_own_game(board, params, epochs=num_train_epochs, legal_boards=legal_boards[0:max_legal_boards], epsilon=epsilon, eta=eta)
                    # Choose next move by quantum evaluation of all possible boards
                    board_after_move = choose_board(legal_boards, params)
                    # Validate training with all legal boards
                    validation_cost = circuit_score(params, legal_boards)
                    # PRINTS and BOOKKEEPING
                    training_cost_list.append(training_cost)
                    validation_cost_list.append(validation_cost)
                    param_list.append(params)
                    print('Move {}: Validation cost with {} unknown boards = {}'.format(move, max(0,len(legal_boards)-max_legal_boards), validation_cost))
                else:
                    board_after_move = legal_boards[np.random.randint(0,len(legal_boards))]
                # Make move and set new board
                board.set_board(board_after_move)
                print("Epoch {}, Move {}, {}'s turn".format(epoch+1, move+1, turn), board)
        plt.plot_cost(training_cost_list, validation_cost_list, "cost-{}.pdf".format(filename_i))
        plt.plot_params(param_list, "params-{}.pdf".format(filename_i))
    circuit['params'] = params
    
        
def choose_board(legal_boards, params):
    global CURRENT_BOARD
    rating = []
    for legal_board in legal_boards:
        CURRENT_BOARD = legal_board
        # Evaluate global CURRENT_BOARD with global CIRCUIT and LAYERS
        rating.append(full_circuit(params)[1]*2-1)
    # Find and return board with highest rating
    index_board = rating.index(max(rating))
    return legal_boards[index_board]

def get_ancillary_params(num_wires):
    return np.random.uniform(low=0, high=2*np.pi, size=num_wires-1)
    
class QAOA():
    def __init__(self, layers, circuit_params=None, ancillary_params=None, num_wires=NUM_WIRES):
        qaoa_params = circuit_params or np.random.uniform(low=-1, high=1, size=layers*2)
        ancillary_params = ancillary_params or get_ancillary_params(num_wires)
        self.circuit = {
            'fun': qaoa_circuit,
            'params': np.concatenate((qaoa_params, ancillary_params)),
            'layers': layers
        }
        
class RANDOM():
    def __init__(self, layers, circuit_params=None, ancillary_params=None, num_wires=NUM_WIRES):
        random_circuit_params = circuit_params or np.random.uniform(low=-1, high=1, size=layers)
        ancillary_params = ancillary_params or get_ancillary_params(num_wires)
        self.circuit = {
            'fun': random_circuit,
            'params': np.concatenate((random_circuit_params, ancillary_params)),
            'layers': layers
        }

if __name__ == '__main__':
    print('Hello Chess World!')
    
    qaoa = QAOA(layers=1).circuit
    random = RANDOM(layers=1).circuit

    # Choose on of the three possible applications
    
    # 1. Game between two (possibly pretrained) quantum circuits
    # 2. Train single circuit as white against random legal moves via gradient descent (learns according to a special score system)
    # 3. Train one circuit against itself by playing white vs black with two slightly different parameter sets
    
    application = 3
    
    if application == 1:
        game_two_circuits(qaoa, random, num_moves=30, num_train_epochs=1)
    elif application == 2:
        train_circuit(qaoa, num_epochs=100, num_moves=1, num_train_epochs=1, max_legal_boards=3, epsilon=0.25, eta=0.25)
    elif application == 3:
        train_circuit_against_itself(qaoa, 100, num_moves=5, epsilon=0.25, param_change_ratio=0.1, eta=0.01)
    