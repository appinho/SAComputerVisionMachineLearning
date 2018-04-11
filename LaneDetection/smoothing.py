class Smoothing():
    def smooth(self,old_value,new_value):
        return old_value + 0.1*new_value