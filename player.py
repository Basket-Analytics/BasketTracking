class Player:
    def __init__(self, ID, team, color):
        self.ID = ID
        self.team = team
        self.color = color
        self.previous_bb = None
        # dict of tuples {timestamp: (position_y, position_x), ...}
        self.positions = {}
