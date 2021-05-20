class Region:
    def __init__(self, name, x1, y1, x2, y2, t=-1):
        # <file_name>;<x1>;<y1>;<x2>;<y2>;<type>
        # When instantiating a Region object without type parameter, it is defined as -1 because we want to classify it later
        self.file_name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.type = t

    def contains(self, other):
        # Returns true if the compared "other" region has at least one point inside the "self" region
        in_vertex = 0
        if self.x1 < other.x1 < self.x2 and self.y1 < other.y1 < self.y2:
            in_vertex += 1
        if self.x1 < other.x1 < self.x2 and self.y1 < other.y2 < self.y2:
            in_vertex += 1
        if self.x1 < other.x2 < self.x2 and self.y1 < other.y1 < self.y2:
            in_vertex += 1
        if self.x1 < other.x2 < self.x2 and self.y1 < other.y2 < self.y2:
            in_vertex += 1
        return in_vertex != 0 and in_vertex != 2

    def __eq__(self, other):
        # Overriding equality of Region class for permitting to ask a regions set from the ground-truth
        # if a possible region detected during training is present in the set in O(1) complexity
        if not isinstance(other, Region):
            return False

        return (int(self.x1)-5 <= int(other.x1) <= int(self.x1)+5 and
                int(self.y1)-5 <= int(other.y1) <= int(self.y1)+5 and
                int(self.x2)-5 <= int(other.x2) <= int(self.x2)+5 and
                int(self.y2)-5 <= int(other.y2) <= int(self.y2)+5)

    def __hash__(self):
        # Overriding hash function for Region class to store its instantiated objects in a regions set
        return hash((self.file_name, int(self.x1), int(self.y1), int(self.x2), int(self.y2)))

    def show(self):
        return self.x1, self.y1, self.x2, self.y2
