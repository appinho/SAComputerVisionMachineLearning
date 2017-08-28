import numpy as np

class Track(object):
    def __init__(self,object):
        # todo.change to dictonary
        self.number_of_states = 4 # x,y,vx,vy
        self.x = object.x
        self.xp = object.x
        self.P = np.matrix(np.eye(self.number_of_states))
        self.Pp = np.matrix(np.eye(self.number_of_states))
        self.z = np.matrix([])
        self.width = object.width
        self.length = object.length
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