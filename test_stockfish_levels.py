import chess
import chess.engine
import os

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
    raise Exception("Stockfish executable not found.")

print(f"Using Stockfish at: {stockfish_path}")

def play_stockfish_vs_stockfish(skill_white, skill_black):
    """Play Stockfish vs Stockfish at different skill levels"""
    board = chess.Board()

    engine_white = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine_white.configure({"Skill Level": skill_white})

    engine_black = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine_black.configure({"Skill Level": skill_black})

    move_count = 0
    max_moves = 200

    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            result = engine_white.play(board, chess.engine.Limit(time=0.1))
        else:
            result = engine_black.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
        move_count += 1

    engine_white.quit()
    engine_black.quit()

    if board.is_checkmate():
        if board.turn == chess.BLACK:
            return 1.0  # White won
        else:
            return 0.0  # Black won
    else:
        return 0.5  # Draw

def test_skill_levels():
    matchups = [
        (0, 5),
        (0, 10),
        (5, 10),
        (10, 15),
        (15, 20),
    ]

    games_per_matchup = 4

    for skill_white, skill_black in matchups:
        print(f"\nSkill {skill_white} (White) vs Skill {skill_black} (Black)")
        white_wins = 0
        draws = 0
        black_wins = 0

        for game in range(games_per_matchup):
            print(f"  Game {game+1}/{games_per_matchup}...", end=" ")
            result = play_stockfish_vs_stockfish(skill_white, skill_black)
            if result == 1.0:
                white_wins += 1
                print("White wins")
            elif result == 0.5:
                draws += 1
                print("Draw")
            else:
                black_wins += 1
                print("Black wins")

        print(f"  Result: Skill {skill_white}: {white_wins}W, Skill {skill_black}: {black_wins}W, Draws: {draws}")

if __name__ == "__main__":
    test_skill_levels()
