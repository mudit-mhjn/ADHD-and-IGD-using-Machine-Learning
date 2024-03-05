class HashTable:
    def __init__(self, capacity):
        self.values = [None]*capacity

    def __len__(self):
        return len(self.values)

