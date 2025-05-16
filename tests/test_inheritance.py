"""
Test inheritance for splitting off plotting functions
"""
class PlotBox:
    """
    Plotting class for twobox objects
    """

    def plot(self):
        print(f"Plotting box1: {self.box1}, box2: {self.box2}")
        print(f"The box sum: {self.box_sum()}")

class TwoBox(PlotBox):
    """
    Base class for TwoBox
    """

    def __init__(self, box1, box2):
        self.box1 = box1
        self.box2 = box2

    def box_sum(self):
        return self.box1 + self.box2
    


test_box = TwoBox(1, 2)
print(test_box.box_sum())  # Should return 3
test_box.plot()  # Should print "Plotting box1: 1, box2: 2"
test_box.box1 = 2
test_box.plot()  # Should print "Plotting box1: 2, box2: 2"