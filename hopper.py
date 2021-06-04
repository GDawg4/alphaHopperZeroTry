import math
from collections import namedtuple
from random import choice
from montecarlo import MCTS, Node

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")


def min_max(node, depth, alpha, beta, player):
    if depth == 0:
        return node
    if player:
        value = -math.inf
        best_state = None
        for child in node.find_children():
            child_value = min_max(child, depth-1, alpha, beta, False)
            if child_value.evaluate() > value:
                value = child_value.evaluate()
                best_state = child
            value = max(value, child_value.evaluate())
            alpha = max(alpha, value)
            if alpha>=beta:
                break
        return best_state
    else:
        value = math.inf
        best_state = None
        for child in node.find_children():
            child_value = min_max(child, depth-1, alpha, beta, True)
            if child_value.evaluate() < value:
                value = child_value.evaluate()
                best_state = child
            value = min(value, child_value.evaluate())
            alpha = min(beta, value)
            if alpha>=beta:
                break
        return best_state

def new_hopper_board():
    # first_player = [0, 1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22, 30, 31, 33]
    # second_player = [59, 68, 69, 77, 78, 79, 86, 87, 88, 89, 55, 96, 97, 98, 99]
    first_player = [0, 1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22, 30, 31, 40]
    second_player = [59, 68, 69, 77, 78, 79, 86, 87, 88, 89, 95, 96, 97, 98, 99]
    board = tuple(True if j in first_player else False if j in second_player else None for j in range(100))
    return HopperBoard(board, turn=True, winner=None, terminal=False)

def play_min_max():
    return min_max(new_hopper_board(), 10, math.inf, -math.inf, True)

def _find_winner(tup):
    first_player = [0, 1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22, 30, 31, 40]
    second_player = [59, 68, 69, 77, 78, 79, 86, 87, 88, 89, 95, 96, 97, 98, 99]
    first_player_values = [not tup[i] for i in first_player if tup[i] is not None]
    second_player_values = [tup[i] for i in second_player if tup[i] is not None]
    #print(hopper_to_pretty_string(tup))
    if len(first_player_values) == len(first_player) and any(first_player_values):
        return False
    if len(second_player_values) == len(second_player) and any(second_player_values):
        return True
    return None

far_spaces = [2, 18, 20, 22]

def hopper_play_game():
    tree = MCTS()
    hopper = new_hopper_board()
    print(hopper.hopper_to_pretty_string())
    while True:
        row_col_initial = input('Enter row,col for the initial chip: ')
        row_initial, col_initial = map(int, row_col_initial.split(","))
        index_initial = 10 * row_initial + col_initial
        print(hopper.get_possible_moves(index_initial))
        row_col_final = input('Enter row,col for the final chip: ')
        row_final, col_final = map(int, row_col_final.split(","))
        index_final = 10 * row_final + col_final
        if not hopper.can_get_there(index_initial, index_final):
            raise RuntimeError("Invalid move")
        hopper = hopper.hopper_make_move(index_initial, index_final)
        print(hopper.hopper_to_pretty_string())
        if hopper.terminal:
            break
        hopper = min_max(hopper, 1, -math.inf, math.inf, True)
        hopper.hopper_to_pretty_string()
        print(hopper.hopper_to_pretty_string())
        if hopper.terminal:
            break


class HopperBoard(_TTTB, Node):
    def hopper_to_pretty_string(board):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else "-"))
        rows = [
            [
                to_char(board.tup[10 * row + col]) for col in range(10)
            ] for row in range(10)
        ]
        return (
            "\n  0 1 2 3 4 5 6 7 8 9\n"
            + "\n".join(str(i) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )

    def find_children(board):
        if board.terminal:
            return set()
        own_spaces = [i for i, value in enumerate(board.tup) if value is board.turn]
        return [
            board.hopper_make_move(i, j) for i in own_spaces for j in board.get_possible_moves(i)
        ]

    def evaluate(board):
        score_list = [i for i in range(len(board.tup)) if board.tup[i] is True]
        score = sum(score_list)
        score_list2 = [100 - i for i in range(len(board.tup)) if board.tup[i] is False]
        score2 = sum(score_list2)
        moves = len(board.find_children())
        total_score = sum(score_list)-sum(score_list2)
        return -total_score

    def is_terminal(board):
        return board.terminal

    def is_empty(board, coord):
        if coord < 99:
            return board.tup[coord] is None
        return False

    def is_occupied(board, coord):
        return board.tup[coord] is not None

    def is_occupied_by_current(board, coord):
        return board.tup[coord] == board.turn

    def is_adjacent_vertical(board, start, end):
        return end == start + 10 or end == start - 10

    def is_adjacent_horizontal(board, start, end):
        return (end == start + 1 or end == start - 1) and round(start // 10) == round(end // 10)

    def is_adjacent_diagonal(board, start, end):
        return (end == start + 11 or end == start - 11 or end == start + 9 or end == start - 9) and abs(round(end//10) - round(start//10)) == 1

    def is_two_spaces_away_vertical(board, start, end):
        return end == start + 20 or end == start - 20 and end >= 0

    def is_two_spaces_away_horizontal(board, start, end):
        return (end == start + 2 or end == start - 2) and round(start // 10) == round(end // 10) and end >= 0

    def is_two_spaces_away_diagonal(board, start, end):
        return (end == start + 22 or end == start - 22 or end == start + 18 or end == start - 18) and abs(round(end//10) - round(start//10)) == 2 and end >= 0

    def is_two_spaces_away(self, start, end):
        return self.is_two_spaces_away_vertical(start, end) \
               or self.is_two_spaces_away_horizontal(start, end) \
               or self.is_two_spaces_away_diagonal(start, end)

    def is_adjacent(self, start, end):
        return self.is_adjacent_vertical(start, end) \
               or self.is_adjacent_horizontal(start, end)
               #or self.is_adjacent_diagonal(start, end)

    def can_start_checking(board, start, end):
        return board.tup[end] is None and board.is_occupied_by_current(start)

    def get_valid_far_spaces(board, start):
        available_spaces = []
        for i in far_spaces:
            if board.is_two_spaces_away(start, start + i) and board.is_empty(start + i) and board.is_occupied((int)(start + i / 2)):
                available_spaces.append(start + i)
            if board.is_two_spaces_away(start, start - i) and board.is_empty(start - i) and board.is_occupied((int)(start - i / 2)):
                available_spaces.append(start - i)
        return available_spaces

    def can_hop_there_at_once(board, start, end):
        return end in board.get_valid_far_spaces(start)

    def can_hop_there(board, start, end, visited):
        if board.can_hop_there_at_once(start, end):
            return True
        possible_moves = board.get_valid_far_spaces(start)
        for i in possible_moves:
            if i not in visited:
                local = visited
                local.append(i)
                if board.can_hop_there(i, end, local):
                    return True
        return False

    def can_get_there(board, start, end, visited=[]):
        #return board.can_start_checking(start, end) and (board.is_adjacent(start, end) or board.can_hop_there(start, end))
        # if end == 66:
        #     print('start checking', board.can_start_checking(start, end))
        #     print('adjacent', board.is_adjacent(start, end))
        #     print('hop', board.can_hop_there(start, end))
        # var = None
        return board.can_start_checking(start, end) and (board.is_adjacent(start, end) or board.can_hop_there(start, end, visited))

    def get_possible_moves(board, start):
        possible_moves_list = [position for position in range(len(board.tup)) if board.can_get_there(start, position, [])]
        return possible_moves_list

    def hopper_make_move(board, start, end):
        tup = board.tup[:start] + (None,) + board.tup[start+1:]
        tup = tup[:end] + (board.turn,) + tup[end+1:]
        turn = not board.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        return HopperBoard(tup, turn, winner, is_terminal)