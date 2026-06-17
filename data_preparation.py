def turn_fen_to_board_inputs(fen_position: str):
    inputs = []
    
    [board, side, castle, pawn, halfclock, moves] = fen_position.split(' ')
    ranks = board.split('/')
    
    for i in range(8):  # ranks are given from 8 to 1 in FEN
        for char in ranks[i]:
            if char == 'p':
                inputs.extend([-1, 0, 0, 0, 0, 0])
            elif char == 'n':
                inputs.extend([0, -1, 0, 0, 0, 0])
            elif char == 'b':
                inputs.extend([0, 0, -1, 0, 0, 0])
            elif char == 'r':
                inputs.extend([0, 0, 0, -1, 0, 0])
            elif char == 'q':
                inputs.extend([0, 0, 0, 0, -1, 0])
            elif char == 'k':
                inputs.extend([0, 0, 0, 0, 0, -1])
            elif char == 'P':
                inputs.extend([1, 0, 0, 0, 0, 0])
            elif char == 'N':
                inputs.extend([0, 1, 0, 0, 0, 0])
            elif char == 'B':
                inputs.extend([0, 0, 1, 0, 0, 0])
            elif char == 'R':
                inputs.extend([0, 0, 0, 1, 0, 0])
            elif char == 'Q':
                inputs.extend([0, 0, 0, 0, 1, 0])
            elif char == 'K':
                inputs.extend([0, 0, 0, 0, 0, 1])
            elif int(char) in range(1, 9):  # from 1 to 8
                for file in range(int(char)):
                    inputs.extend([0, 0, 0, 0, 0, 0])
    
    # pure sanity check
    assert(len(inputs) == 384)
    
    if side == 'w':
        inputs.append(1)
    elif side == 'b':
        inputs.append(-1)
        
    castle_rights = [0, 0, 0, 0]
    for right in castle:
        if right == 'K':
            castle_rights[0] = 1
        if right == 'Q':
            castle_rights[1] = 1
        if right == 'k':
            castle_rights[2] = 1
        if right == 'q':
            castle_rights[3] = 1
    inputs.extend(castle_rights)
    
    return inputs

def turn_fen_to_board_inputs_3_outputs(position: str):
    [fen_position, evaluation] = position.split(',')
    
    if evaluation == "ERROR":
        return None, None

    inputs = turn_fen_to_board_inputs(fen_position)
    
    # apparently, this is bad, as torch.nn.CrossEntropyLoss cannot be applied
    #if evaluation > 150:
    #    outputs = [0, 0, 1]
    #elif evaluation >= -150:
    #    outputs = [0, 1, 0]
    #elif evaluation < -150:
    #    outputs = [1, 0, 0]
    if 'M' in evaluation:
        if evaluation[1] == '-':
            output = 0
        #elif evaluation[0] == 'M':
        else:
            output = 2
    else:
        evaluation = int(evaluation)
        if evaluation > 150:
            output = 2
        elif evaluation >= -150:
            output = 1
        elif evaluation < -150:
            output = 0
    
    return inputs, output
    
    
def turn_fen_to_board_inputs_12_outputs(position: str):
    [fen_position, evaluation] = position.split(',')
    
    if evaluation == "ERROR":
        return None, None

    inputs = turn_fen_to_board_inputs(fen_position)
    
    # [-8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8]
    class_borders = [-8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8]
    if 'M' in evaluation:
        if evaluation[1] == '-':
            output = 0
        else:
            output = 11
    else:
        evaluation = int(evaluation)
        cls = 0
        while cls < 11 and evaluation > class_borders[cls]*100:
            cls += 1
        output = cls
    
    return inputs, output
    
    
def turn_fen_to_board_inputs_15_outputs(position: str):
    [fen_position, evaluation] = position.split(',')
    
    if evaluation == "ERROR":
        return None, None

    inputs = turn_fen_to_board_inputs(fen_position)
    
    # [-7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    class_borders = [i-0.5 for i in range(-7, 0, 1)] + [i+0.5 for i in range(1, 8, 1)]
    if 'M' in evaluation:
        if evaluation[1] == '-':
            output = 0
        else:
            output = 14
    else:
        evaluation = int(evaluation)
        cls = 0
        while cls < 14 and evaluation > class_borders[cls]*100:
            cls += 1
        output = cls
    
    return inputs, output
    
    
def turn_fen_to_board_inputs_20_outputs(position: str):
    [fen_position, evaluation] = position.split(',')
    
    if evaluation == "ERROR":
        return None, None

    inputs = turn_fen_to_board_inputs(fen_position)
    
    # [-7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    class_borders = [i-0.5 for i in range(-7, 0, 1)] + [i/2 for i in range(-2, 3, 1)] + [i+0.5 for i in range(1, 8, 1)]
    if 'M' in evaluation:
        if evaluation[1] == '-':
            output = 0
        else:
            output = 19
    else:
        evaluation = int(evaluation)
        cls = 0
        while cls < 19 and evaluation > class_borders[cls]*100:
            cls += 1
        output = cls
    
    return inputs, output
    
def turn_fen_to_board_inputs_raw_output(position: str):
    [fen_position, evaluation] = position.split(',')
    
    if evaluation == "ERROR":
        return None, None

    inputs = turn_fen_to_board_inputs(fen_position)
    
    if 'M' in evaluation:
        if evaluation[1] == '-':
            output = -10000
        else:
            output = 10000
    else:
        output = int(evaluation)
    
    return inputs, output


if __name__ == "__main__":
    print(turn_fen_to_board_inputs_3_outputs("rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPPN1PPP/R1BQKBNR b KQkq - 1 3,35"))
    print(turn_fen_to_board_inputs_raw_output("rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPPN1PPP/R1BQKBNR b KQkq - 1 3,35"))
