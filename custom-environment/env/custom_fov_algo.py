import math
from fractions import Fraction

def compute_fov(origin, angle, fov, is_blocking, mark_visible):
    mark_visible(*origin)

    #determine both rays in fov
    ls = angle+fov/2.0
    rs = angle-fov/2.0
    if(ls>360): ls -= 360
    if(rs<0):   rs += 360

    q = []

    start_slope = 0
    if(ls>45 and ls<=135):
        start_slope = math.tan(math.radians(90-ls))
        q.append(0)
    elif(ls>135 and ls<=225):
        start_slope = math.tan(math.radians(180-ls))
        q.append(1)
    elif(ls>225 and ls<=315):
        start_slope = math.tan(math.radians(270-ls))
        q.append(2)
    elif(ls>315 or ls<=45):
        start_slope = math.tan(math.radians(360-ls))
        q.append(3)

    end_slope = 0
    if(rs>=45 and rs<135):
        end_slope = math.tan(math.radians(90-rs))
        q.append(0)
    elif(rs>=135 and rs<225):
        end_slope = math.tan(math.radians(180-rs))
        q.append(1)
    elif(rs>=225 and rs<315):
        end_slope = math.tan(math.radians(270-rs))
        q.append(2)
    elif(rs>=315 or rs<45):
        end_slope = math.tan(math.radians(360-rs))
        q.append(3)

    # #determine quadrants to be searched and also
    # q = []
    # if( (rs>=45 and rs<135) or (ls>45 and ls<=135) or ((rs<=45 and rs>=315) and (ls>=135 or ls<=225) )): q.append(0)
    # if( (rs>=135 and rs<225) or (ls>135 and ls<=225) or (rs<=135 and ls>=225 and ls>=rs) ):    q.append(1)
    # if( (rs>=225 and rs<315) or (ls>225 and ls<=315) or ((rs<=225 and rs>=135) and (ls>=315 or ls<=45)) ):   q.append(2)
    # if( (rs>=315 or rs<45) or (ls>315 or ls<=45) or ((rs<=315 and ls>=45 and rs>=ls) )):  q.append(3)
    # print(angle,fov,q)

    #Utility functions
    def reveal(tile):
        x, y = quadrant.transform(tile)
        mark_visible(x, y)

    def is_wall(tile):
        if tile is None:
            return False
        x, y = quadrant.transform(tile)
        return is_blocking(x, y)

    def is_floor(tile):
        if tile is None:
            return False
        x, y = quadrant.transform(tile)
        return not is_blocking(x, y)

    def scan(row):
        prev_tile = None
        for tile in row.tiles():
            if is_wall(tile) or is_symmetric(row, tile):
                reveal(tile)
            if is_wall(prev_tile) and is_floor(tile):
                row.start_slope = slope(tile)
            if is_floor(prev_tile) and is_wall(tile):
                next_row = row.next()
                next_row.end_slope = slope(tile)
                scan(next_row)
            prev_tile = tile
        if is_floor(prev_tile):
            scan(row.next())

    if(len(q)==1):
        quadrant = Quadrant(q[0], origin)
        first_row = Row(1, start_slope, end_slope)
        scan(first_row)
    else:
        if(q[0] == 1 or q[0] == 2):
            quadrant = Quadrant(q[0], origin)
            first_row = Row(1, Fraction(-1), -start_slope)
            scan(first_row)
        else:
            quadrant = Quadrant(q[0], origin)
            first_row = Row(1, start_slope, Fraction(1))
            scan(first_row)


        if(q[1] == 1 or q[1] == 2):
            quadrant = Quadrant(q[1], origin)
            first_row = Row(1, -end_slope, Fraction(1))
            scan(first_row)
        else:
            quadrant = Quadrant(q[1], origin)
            first_row = Row(1, Fraction(-1), end_slope)
            scan(first_row)

class Quadrant:

    north = 0
    west  = 1
    south = 2
    east  = 3

    def __init__(self, cardinal, origin):
        self.cardinal = cardinal
        self.ox, self.oy = origin

    def transform(self, tile):
        row, col = tile
        if self.cardinal == self.north:
            return (self.ox + col, self.oy - row)
        if self.cardinal == self.south:
            return (self.ox + col, self.oy + row)
        if self.cardinal == self.east:
            return (self.ox + row, self.oy + col)
        if self.cardinal == self.west:
            return (self.ox - row, self.oy + col)

class Row:
    def __init__(self, depth, start_slope, end_slope):
        self.depth = depth
        self.start_slope = start_slope
        self.end_slope = end_slope

    def tiles(self):
        min_col = round_ties_up(self.depth * self.start_slope)
        max_col = round_ties_down(self.depth * self.end_slope)
        for col in range(min_col, max_col + 1):
            yield (self.depth, col)

    def next(self):
        return Row(
            self.depth + 1,
            self.start_slope,
            self.end_slope)

def slope(tile):
    row_depth, col = tile
    return Fraction(2 * col - 1, 2 * row_depth)

def is_symmetric(row, tile):
    row_depth, col = tile
    return (col >= row.depth * row.start_slope
        and col <= row.depth * row.end_slope)

def round_ties_up(n):
    return math.floor(n + 0.5)

def round_ties_down(n):
    return math.ceil(n - 0.5)

def scan_iterative(row):
    rows = [row]
    while rows:
        row = rows.pop()
        prev_tile = None
        for tile in row.tiles():
            if is_wall(tile) or is_symmetric(row, tile):
                reveal(tile)
            if is_wall(prev_tile) and is_floor(tile):
                row.start_slope = slope(tile)
            if is_floor(prev_tile) and is_wall(tile):
                next_row = row.next()
                next_row.end_slope = slope(tile)
                rows.append(next_row)
            prev_tile = tile
        if is_floor(prev_tile):
            rows.append(row.next())
