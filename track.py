import numpy as np

class Track(object):
    def __init__(self,object):
        # todo.change to dictonary
        self.number_of_states = 4 # x,y,vx,vy
        self.x = [object.x]
        self.xp = self.x
        self.P = [np.matrix([
            [0.3    ,    0  ,    0  ,       0],
            [0      ,0.3    ,    0  ,       0],
            [0      ,0      ,2      ,       0],
            [0      ,0      ,0      ,       2]
        ])]
        self.Pp = self.P
        self.z = [np.matrix([])]
        self.width = [object.width]
        self.length = [object.length]
        self.age = 0
        self.not_updated=0
    """
    def display_track(self):
        print " X "
        print self.x
        print " XP "
        print self.xp
        print " P "
        print self.P
        print " PP "
        print self.Pp
    """