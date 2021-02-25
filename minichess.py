
import numpy as np
import chess
import chess.engine
import chess.pgn

COLUMN_LABELS = ["a","b","c","d","e","f","g","h"]
MICROCHESS_SIZE = (5,4)
STOCKFISH_DIR = "/usr/local/bin/stockfish"

class MiniBoard():
    
    def __init__(self, 
                 height=5, 
                 width=4, 
                 board=[], 
                 periodic_boundaries=False, 
                 board_preset="Microchess",
                 name="MicrochessBoard",
                 castling_rights=[True,True]):
        assert (len(board) == 0 or height*width == len(board)), "Board size does not match given board."
        self.size = (height, width)
        if len(board) > 0:
            self.board = board
        elif board_preset == "Microchess":
            self.size = MICROCHESS_SIZE
            self.board = [""]*20
            for c in range(2):
                self.board[19*c] = "r" if c else "R"
                self.board[19*c + (1-2*c)*1] = "b" if c else "B"
                self.board[19*c + (1-2*c)*2] = "n" if c else "N"
                self.board[19*c + (1-2*c)*3] = "k" if c else "K"
                self.board[19*c + (1-2*c)*7] = "p" if c else "P"
        else:
            self.board = [""]*height*width
        self.periodic_boundaries = True
        self.board_preset = board_preset
        self.castling_rights = castling_rights
        self.name = name
        
    def __str__(self):
        board_str = "  ." + "_"*(4*self.size[1]-1) + ".\n"
        for h in range(self.size[0]):
            board_str += str(self.size[0]-h) + " |"
            for w in range(self.size[1]):
                piece = self.board[(self.size[0]-h-1)*self.size[1]+w]
                board_str += "_{}_|".format(piece if piece else "_")
            board_str += "\n"
        board_str += "  "
        for i in range(self.size[1]):
            board_str += "  {} ".format(COLUMN_LABELS[i])
        return board_str
        
    def copy(self):
        miniboard_copy = MiniBoard()
        miniboard_copy.size = self.size
        miniboard_copy.board = self.board.copy()
        miniboard_copy.periodic_boundaries = self.periodic_boundaries
        miniboard_copy.board_preset = self.board_preset
        miniboard_copy.castling_rights = self.castling_rights.copy()
        miniboard_copy.name = self.name + "Copy"
        return miniboard_copy

    def convert_to_board_index(self, move):
        from_vec, to_vec = convert_to_vector(move)
        from_i = from_vec[1]*self.size[1] + from_vec[0]
        if to_vec:
            to_i = to_vec[1]*self.size[1] + to_vec[0]
            return from_i, to_i
        return from_i

    def make_move(self, move, print_move=False):
        if print_move:
            print("Move: ", move)
            self.print_move(move)
        from_i, to_i = self.convert_to_board_index(move)
        moving_piece = self.board[from_i]
        if (self.board_preset == "Minichess" and any(*self.castling_rights) and move in ["d1b1", "a5c5"]):
            # castling
            if (moving_piece == "K" and self.castling_rights[0]):
                # move white rook
                self.board[0] = ""
                self.board[2] = "R"
            if (moving_piece == "k" and self.castling_rights[1]):
                # move black rook
                self.board[-1] = ""
                self.board[-3] = "r"
                
        self.board[from_i] = ""
        self.board[to_i] = self.check_for_pawn_promotion(moving_piece, to_i)
        
        if self.castling_rights[0] and (moving_piece == "K" or moving_piece == "R"):
            self.castling_rights[0] = False
        if self.castling_rights[1] and (moving_piece == "k" or moving_piece == "r"):
            self.castling_rights[1] = False
    
    def random_move(self, turn="w", print_move=False):
        possible_moves = self.legal_moves(turn=turn)
        rndm_mv = possible_moves[np.random.randint(0,len(possible_moves))]
        self.make_move(rndm_mv, print_move=print_move)
    
    def place_random_king(self, white=True):
        self.board[np.random.randint(0,len(self.board))] = "K" if white else "k"
        
    def place_king(self, i, white=True):
        self.board[i] = "K" if white else "k"
        
    def is_check(self, move=""):
        mb = self.copy()
        if move:
            mb.make_move(move)
        fen = mb.convert_board_to_FEN()
        cb = chess.Board(fen)
        return cb.is_check()
        
    def is_checkmate(self, move=""):
        mb = self.copy()
        if move:
            mb.make_move(move)
        return len(mb.legal_moves("w")) == 0 or len(mb.legal_moves("b")) == 0
        
    def legal_moves(self, turn="w"):
        lgl_mvs = []
        fen = self.convert_board_to_FEN(turn)
        chess_board = chess.Board(fen)
        for move in chess_board.legal_moves:
            if not (any(a in str(move) for a in COLUMN_LABELS[self.size[1]:]) or any(str(i+1) in str(move) for i in range(self.size[0],8))):
                lgl_mvs.append(str(move))
        # castling
        if self.board_preset == "Microchess":
            if turn == "w" and self.castling_rights[0] and self.board[0] == "R" and self.board[self.size[1]-1] == "K" and all(self.board[self.size[1]-2-i] == "" for i in range(2)):
                if not self.is_check() and not self.is_check("d1b1"):
                    lgl_mvs.append("d1b1")
            if turn == "b" and self.castling_rights[1] and self.board[-self.size[1]] == "k" and self.board[-1] == "r" and all(self.board[1+i] == "" for i in range(2)):
                if not self.is_check() and not self.is_check("a5c5"):
                    lgl_mvs.append("a5c5")
            
        return lgl_mvs
    
    def print_move(self, move):
        from_i, to_i = self.convert_to_board_index(move)
        new_miniboard = self.copy()
        moving_piece = new_miniboard.board[from_i]
        target_piece = new_miniboard.board[to_i]
        new_miniboard.board[from_i] = moving_piece or "0"
        new_miniboard.board[to_i] = "x" if target_piece else "*"
        print(new_miniboard)
    
    def check_for_pawn_promotion(self, moving_piece, i):
        if moving_piece == "P" and i > len(self.board)-self.size[1]-1:
            return "Q"
        if moving_piece == "p" and i < self.size[1]:
            return "q"
        return moving_piece
        
    
    def legal_boards(self, turn="w", print_boards=False):
        # Return list of MiniBoard.board lists
        lgl_mvs = self.legal_moves(turn=turn)
        lgl_brds = []
        for mv in lgl_mvs:
            from_i, to_i = self.convert_to_board_index(mv)
            new_board = self.board.copy()
            moving_piece = new_board[from_i]
            new_board[from_i] = ""
            new_board[to_i] = self.check_for_pawn_promotion(moving_piece, to_i)
            lgl_brds.append(new_board)
            if print_boards:
                self.print_move(mv)
        return lgl_brds
            
        
    def convert_board_to_FEN(self, turn="w"):
        fen = "8/"*(8-self.size[0])
        for h in range(self.size[0]):
            row = ""
            space = -1
            for w in range(self.size[1]):
                piece = self.board[(self.size[0]-h-1)*self.size[1]+w]
                space += 1
                if piece:
                    row += "{}{}".format(space or "", piece)
                    space = -1
            if not row:
                fen += "8/"
            else:
                row += str(space+1+8-self.size[1])
                fen += row + "/"
        fen = fen[:-1]
        fen += " {} - - 0 1".format(turn)
        return fen
    
    def set_board(self, new_board):
        self.board = new_board.copy()
        
    def clear_board(self):
        self.board = [""]*self.size[0]*self.size[1]
    
    def is_legal_move(old_board, new_board):
        old_b = old_board if(isinstance(old_board, MiniBoard)) else MiniBoard(old_board)
        lgl_brds = old_b.legal_boards()
        new_b = new_board.board if(isinstance(new_board, MiniBoard)) else new_board
        return new_b in lgl_brds
    
    def convert_to_board_notation(self, square):
        return COLUMN_LABELS[square % self.size[1]] + str(square // self.size[1] + 1)
    
    def convert_to_chess_square(self, square):
        return (square//self.size[1])*8 + (square % self.size[1])
    
    def convert_to_move(self, origin_square, target_square):
        return self.convert_to_board_notation(origin_square) + self.convert_to_board_notation(target_square)
    
    def convert_to_vector(self, square):
        return np.array([square % self.size[1], square // self.size[1]])
    
    def distance_pbc(self, origin_square, target_square):
        o_vec = self.convert_to_vector(origin_square)
        t_vec = self.convert_to_vector(target_square)
        d = np.abs(o_vec - t_vec)
        for i in range(2):
            if (d[i] > self.size[1-i]/2):
                d[i] = abs(d[i] - self.size[1-i])
        return max(d)
            
    
    def legality(self, origin_square, target_square):
        if not self.periodic_boundaries:
            origin_chess_square = self.convert_to_chess_square(origin_square)
            target_chess_square = self.convert_to_chess_square(target_square)
            distance = chess.square_distance(origin_chess_square, target_chess_square)
            valid_piece_chess_square = self.convert_to_chess_square(self.board.index("K"))
            distance_to_valid_piece = chess.square_distance(origin_chess_square,valid_piece_chess_square)
            return distance_to_valid_piece, distance
       
        distance = self.distance_pbc(origin_square, target_square)
        distance_to_valid_piece = self.distance_pbc(origin_square, self.board.index("K"))
        return distance_to_valid_piece, distance
        
    def material_score(self, white=True):
        return 10*self.board.count("K" if white else "k") \
               + 9*self.board.count("Q" if white else "q") \
               + 5*self.board.count("R" if white else "r") \
               + 3*self.board.count("N" if white else "n") \
               + 3*self.board.count("B" if white else "b") \
               + 1*self.board.count("P" if white else "p") \
        
    def stockfish_score(self, turn="w"):
        fen = self.convert_board_to_FEN(turn)
        root_moves = [chess.Move.from_uci(mv) for mv in self.legal_moves(turn)]
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_DIR)
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(time=0.5, depth=1), root_moves=root_moves)
        engine.quit()
        score = info.get("score")
        return score.white(), score.black()
    
    def stockfish_move(self, turn="w"):
        fen = self.convert_board_to_FEN(turn=turn)
        root_moves = [chess.Move.from_uci(mv) for mv in self.legal_moves(turn)]
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_DIR)
        board = chess.Board(fen)
        result = engine.play(board, chess.engine.Limit(time=0.5, depth=1), root_moves=root_moves)
        engine.quit()
        return str(result.move)
        
    def analyze_board(self, turn="w"):
        material_score_diff = self.material_score(white=True) - self.material_score(white=False)
        mobility_score_diff = len(self.legal_moves("w")) - len(self.legal_moves("b"))
        return (1-2*int(turn!="w")) * (material_score_diff + mobility_score_diff)

    def best_move(self, turn="w"):
        lgl_mvs = self.legal_moves(turn=turn)
        lgl_brds = self.legal_boards(turn=turn)
        max_i = 0
        max_score = -100
        for i, brd in enumerate(lgl_brds):
            mb = MiniBoard(board=brd)
            score = mb.analyze_board(turn=turn)
            if score > max_score:
                max_score = score
                max_i = i
        return lgl_mvs[max_i] if len(lgl_mvs) > 0 else ""        
        
    def convert_moves_to_pgn(self, moves, write_to_file=False, print_it=True):
        pgn_string = '[Variant "From Position"]\n'
        fen = self.convert_board_to_FEN()
        pgn_string += '[FEN "{}"]\n'.format(fen) 
        chessboard = chess.Board(fen)
        mvs = chessboard.variation_san([chess.Move.from_uci(mv) for mv in moves])
        pgn_string += mvs
        
        if write_to_file:
            with open("output/pgn.txt", "w") as f:
                f.write(pgn_string)
        
        if print_it:
            print("=== PGN STRING ===\n")
            print(pgn_string)
            print("\n==================")
        
def miniboard_combinations(size=(5,4), piece="K"):
    miniboards = []
    for i in range(size[0]*size[1]):
        new_miniboard = MiniBoard(*size)
        new_miniboard.board[i] = piece
        miniboards.append(new_miniboard)
    return miniboards

def convert_to_vector(move):
    from_i = [COLUMN_LABELS.index(move[0]), int(move[1])-1]
    if len(move) > 2:
        to_i = [COLUMN_LABELS.index(move[-2]), int(move[-1])-1]
        return from_i , to_i
    return from_i

def is_draw(moves):
    if len(moves) < 10: return False
    return moves[-1] == moves[-5] == moves[-9]


if __name__ == '__main__':
    mini_board = MiniBoard()
    print(mini_board)
    
    made_moves = []
    i = 0
    while (not mini_board.is_checkmate() and not is_draw(made_moves) and i < 50):
        turn = "w" if i % 2 == 0 else "b"
        
        # PLAY AGAINST STOCKFISH ---------------------------        
        # if turn == "b":
        #     try:
        #         mv = mini_board.stockfish_move(turn)
        #     except:
        #         mv = mini_board.random_move(turn)
                
        # else:
        #     mv = mini_board.best_move(turn)
        # --------------------------------------------------
            
        # PLAY AGAINST ITSELF ------------------------------
        mv = mini_board.best_move(turn)
        # --------------------------------------------------
        
        if mv != "":
            made_moves.append(mv)
            mini_board.make_move(mv)

        i += 1
    
    # new board with initial configuration
    new_mini_board = MiniBoard()
    
    # convert moves to pgn to view moves using https://lichess.org/paste
    new_mini_board.convert_moves_to_pgn(made_moves)
    