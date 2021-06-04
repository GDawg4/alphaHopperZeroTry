
class GameBoard:
    def __init__(self):
        self.first_player = [0, 1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22, 30, 31, 40]
        self.second_player = [59, 68, 69, 77, 78, 79, 86, 87, 88, 89, 95, 96, 97, 98, 99]
        self.turn = True
        self.far_spaces = [2, 18, 20, 22]

    def print_board(self):
        board = ['1' if j in self.first_player else '2' if j in self.second_player else '+' for j in range(100)]
        string = ''
        count = 0
        for i in board:
            string += str(i)
            if count % 10 == 9:
                print(string)
                string = ''
            else:
                string += '|'
            count += 1

    def is_empty(self, coord):
        return not (coord in self.first_player or coord in self.second_player)

    def is_occupied(self, coord):
        return coord in self.first_player or coord in self.second_player

    def is_occupied_by_current(self, coord):
        return coord in self.first_player if self.turn else coord in self.second_player

    def is_occupied_by_other(self, coord):
        return coord in self.second_player if self.turn else coord in self.first_player

    def is_adjacent_vertical(self, start, end):
        return end == start + 10 or end == start - 10

    def is_adjacent_horizontal(self, start, end):
        return (end == start + 1 or end == start - 1) and start // 10 == end // 10

    def is_adjacent_diagonal(self, start, end):
        return (end == start + 11 or end == start - 11 or end == start + 9 or end == start - 9) and abs(end//10 - start//10) == 1

    def is_adjacent(self, start, end):
        return self.is_adjacent_vertical(start, end) \
               or self.is_adjacent_horizontal(start, end) \
               or self.is_adjacent_diagonal(start, end)

    def is_two_spaces_away_vertical(self, start, end):
        return end == start + 20 or end == start - 20 and end >= 0

    def is_two_spaces_away_horizontal(self, start, end):
        return (end == start + 2 or end == start - 2) and start // 10 == end // 10 and end >= 0

    def is_two_spaces_away_diagonal(self, start, end):
        return (end == start + 22 or end == start - 22 or end == start + 18 or end == start - 18) and abs(end//10 - start//10) == 2 and end >= 0

    def is_two_spaces_away(self, start, end):
        return self.is_two_spaces_away_vertical(start, end) \
               or self.is_two_spaces_away_horizontal(start, end) \
               or self.is_two_spaces_away_diagonal(start, end)

    def get_far_spaces(self, start):
        available_spaces = []
        for i in self.far_spaces:
            if self.is_two_spaces_away(start, start + i) and self.is_occupied(start + i/2):
                available_spaces.append(start + i)
            if self.is_two_spaces_away(start, start - i) and self.is_occupied(start - i/2):
                available_spaces.append(start - i)
        return available_spaces

    def get_valid_far_spaces(self, start):
        available_spaces = []
        for i in self.far_spaces:
            if self.is_two_spaces_away(start, start + i) and self.is_empty(start + i) and self.is_occupied(start + i / 2):
                available_spaces.append(start + i)
            if self.is_two_spaces_away(start, start - i) and self.is_empty(start - i) and self.is_occupied(start - i / 2):
                available_spaces.append(start - i)
        return available_spaces

    def can_hop_there_at_once(self, start, end):
        return end in self.get_valid_far_spaces(start)

    def can_hop_there(self, start, end):
        if self.can_hop_there_at_once(start, end):
            return True
        possible_moves = self.get_valid_far_spaces(start)
        for i in possible_moves:
            self.can_get_there(i, end)
        return False

    def can_start_checking(self, start, end):
        return self.is_occupied_by_current(start) and self.is_empty(end)

    def can_get_there(self, start, end):
        return self.can_start_checking(start, end) and (self.is_adjacent(start, end) or self.can_hop_there(start, end))