import torch
import models
import chess
import chess.engine
import data_preparation
import os
from pathlib import Path

# Find Stockfish executable
stockfish_paths = [
    "stockfish/stockfish-windows-x86-64-avx2.exe",
    "stockfish/stockfish.exe",
    "stockfish/src/stockfish.exe"
]

stockfish_path = None
for path in stockfish_paths:
    if os.path.exists(path):
        stockfish_path = path
        break

if stockfish_path is None:
    raise Exception("Stockfish executable not found. Please ensure it's in the stockfish folder.")

print(f"Using Stockfish at: {stockfish_path}")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.BoardInputBigClassifier(389, 15).to(device)

if not os.path.exists('model.pth'):
    raise Exception("model.pth not found. Train the model first.")

model.load_state_dict(torch.load('model.pth'))
model.eval()
print("Model loaded successfully")

# Evaluation to centipawn mapping (same as training)
class_borders = [i-0.5 for i in range(-7, 0, 1)] + [i+0.5 for i in range(1, 8, 1)]
# Class centers in centipawns: -800, -650, -550, ..., 0, ..., 550, 650, 800
class_values = torch.tensor([-800, -650, -550, -450, -350, -250, -100, 0, 100, 250, 350, 450, 550, 650, 800], dtype=torch.float).to(device)

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

def material_count(board):
    """Count material balance"""
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    return score

def model_evaluate_position(board):
    """Hybrid evaluation: material + neural network positional score"""
    # Material component
    mat = material_count(board)

    # Neural network positional evaluation
    fen = board.fen()
    fen_with_eval = fen + ",0"
    inputs, _ = data_preparation.turn_fen_to_board_inputs_15_outputs(fen_with_eval)

    if inputs is None:
        return mat

    with torch.no_grad():
        X = torch.tensor([inputs], dtype=torch.float).to(device)
        output = model(X)[0]
        shifted = output - output.min()
        weights = shifted / (shifted.sum() + 1e-6)
        nn_eval = torch.sum(weights * class_values).item()

    # Combine: material is primary, NN gives positional bonus
    return mat + nn_eval * 0.5

def minimax(board, depth, alpha, beta, maximizing):
    """Minimax with alpha-beta pruning using neural network evaluation"""
    if depth == 0 or board.is_game_over():
        if board.is_checkmate():
            return -9999 if maximizing else 9999
        eval_score = model_evaluate_position(board)
        return eval_score if eval_score is not None else 0

    # Move ordering - check captures first
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    if maximizing:
        max_eval = -10000
        for move in moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = 10000
        for move in moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def get_best_move_by_evaluation(board, engine, skill_level=20, debug=False):
    """Get best move using minimax search with neural network evaluation"""
    legal_moves = list(board.legal_moves)

    if len(legal_moves) == 0:
        return None

    best_move = None
    best_eval = -10000 if board.turn == chess.WHITE else 10000
    search_depth = 2  # Keep at 2 for reasonable speed

    move_evals = []
    for move in legal_moves:
        board.push(move)
        # After our move, opponent plays - so we minimize if we're white (opponent is black)
        eval_score = minimax(board, search_depth - 1, -10000, 10000, board.turn == chess.WHITE)
        board.pop()

        move_evals.append((move, eval_score))

        if board.turn == chess.WHITE:
            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move
        else:
            if eval_score < best_eval:
                best_eval = eval_score
                best_move = move

    if debug and len(move_evals) > 0:
        move_evals.sort(key=lambda x: x[1], reverse=(board.turn == chess.WHITE))
        print(f"    Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        print(f"    Top 3 moves: {[(str(m), f'{e:.1f}') for m, e in move_evals[:3]]}")
        print(f"    Bottom 3: {[(str(m), f'{e:.1f}') for m, e in move_evals[-3:]]}")
        print(f"    Chosen: {best_move} (eval: {best_eval:.1f})")

    return best_move if best_move else legal_moves[0]

def play_game(skill_level, model_color=chess.WHITE, debug=False):
    """Play a game against Stockfish at a given skill level"""
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": skill_level})

    move_count = 0
    max_moves = 200  # Prevent infinite games

    while not board.is_game_over() and move_count < max_moves:
        if board.turn == model_color:
            # Model's turn
            move = get_best_move_by_evaluation(board, engine, debug=(debug and move_count < 10))
            if move is None:
                break
            board.push(move)
        else:
            # Stockfish's turn
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)

        if debug and move_count < 10:
            print(f"  Move {move_count+1}: {board.peek()}")

        move_count += 1

    engine.quit()

    if debug:
        print(f"  Final position:\n{board}")
        print(f"  Result: {board.result()}")

    # Determine result
    if board.is_checkmate():
        if board.turn != model_color:
            return 1.0  # Model won
        else:
            return 0.0  # Model lost
    elif board.is_stalemate() or board.is_insufficient_material() or move_count >= max_moves:
        return 0.5  # Draw
    else:
        return 0.5  # Draw

# Stockfish skill level to approximate Elo mapping
skill_to_elo = {
    0: 800,
    1: 900,
    2: 1000,
    3: 1100,
    4: 1200,
    5: 1300,
    6: 1400,
    7: 1500,
    8: 1600,
    9: 1700,
    10: 1800,
    11: 1900,
    12: 2000,
    13: 2100,
    14: 2200,
    15: 2300,
    16: 2400,
    17: 2500,
    18: 2600,
    19: 2700,
    20: 3000
}

def estimate_elo():
    """Test model against different Stockfish levels"""
    skill_levels = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    games_per_level = 10  # 5 as white, 5 as black

    results = {}

    for skill in skill_levels:
        print(f"\nTesting against Stockfish Skill Level {skill} (≈{skill_to_elo[skill]} Elo)")
        wins = 0
        draws = 0
        losses = 0

        for game_num in range(games_per_level):
            model_color = chess.WHITE if game_num < games_per_level // 2 else chess.BLACK
            color_str = "White" if model_color == chess.WHITE else "Black"

            print(f"  Game {game_num + 1}/{games_per_level} (Model as {color_str})...", end=" ")
            result = play_game(skill, model_color)

            if result == 1.0:
                wins += 1
                print("Win")
            elif result == 0.5:
                draws += 1
                print("Draw")
            else:
                losses += 1
                print("Loss")

        score = wins + 0.5 * draws
        percentage = (score / games_per_level) * 100
        results[skill] = {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'score': score,
            'percentage': percentage,
            'elo': skill_to_elo[skill]
        }

        print(f"  Results: {wins}W {draws}D {losses}L = {score}/{games_per_level} ({percentage:.1f}%)")

    # Estimate Elo based on results
    print("\n" + "="*60)
    print("ELO ESTIMATION SUMMARY")
    print("="*60)

    for skill in skill_levels:
        r = results[skill]
        print(f"Skill {skill:2d} ({r['elo']:4d} Elo): {r['wins']}W {r['draws']}D {r['losses']}L = {r['percentage']:5.1f}%")

    # Find the skill level where model scores around 50%
    estimated_elo = None
    for skill in skill_levels:
        if results[skill]['percentage'] >= 45 and results[skill]['percentage'] <= 55:
            estimated_elo = results[skill]['elo']
            break

    if estimated_elo:
        print(f"\nEstimated model Elo: ~{estimated_elo}")
    else:
        # Interpolate
        for i in range(len(skill_levels) - 1):
            if results[skill_levels[i]]['percentage'] > 50 and results[skill_levels[i+1]]['percentage'] < 50:
                # Linear interpolation
                skill_low = skill_levels[i]
                skill_high = skill_levels[i+1]
                perc_low = results[skill_low]['percentage']
                perc_high = results[skill_high]['percentage']
                elo_low = results[skill_low]['elo']
                elo_high = results[skill_high]['elo']

                estimated_elo = elo_low + (50 - perc_low) / (perc_high - perc_low) * (elo_high - elo_low)
                print(f"\nEstimated model Elo (interpolated): ~{int(estimated_elo)}")
                break

if __name__ == "__main__":
    # Run 6 games against Stockfish Skill 0
    print("="*60)
    print("Testing Model vs Stockfish Skill 0 - 6 games")
    print("="*60)

    wins, draws, losses = 0, 0, 0
    for i in range(6):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        print(f"Game {i+1} (Model as {'White' if color == chess.WHITE else 'Black'})...", end=" ", flush=True)
        result = play_game(0, color, debug=False)
        if result == 1.0:
            wins += 1
            print("WIN")
        elif result == 0.5:
            draws += 1  
            print("DRAW")
        else:
            losses += 1
            print("LOSS")

    print(f"\nResults: {wins}W {draws}D {losses}L")
