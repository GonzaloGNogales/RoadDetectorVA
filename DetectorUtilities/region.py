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

    def __eq__(self, other):
        # Overriding equality of Region class for permitting to ask a regions set from the ground-truth
        # if a possible region detected during training is present in the set in O(1) complexity
        if not isinstance(other, Region):
            return False

        return (int(self.x1)-15 <= int(other.x1) <= int(self.x1)+15 and
                int(self.y1)-15 <= int(other.y1) <= int(self.y1)+15 and
                int(self.x2)-15 <= int(other.x2) <= int(self.x2)+15 and
                int(self.y2)-15 <= int(other.y2) <= int(self.y2)+15)

    def __hash__(self):
        # Overriding hash function for Region class to store its instantiated objects in a regions set
        return hash((self.file_name, int(self.x1), int(self.y1), int(self.x2), int(self.y2)))

    def show(self):
        return self.x1, self.y1, self.x2, self.y2
