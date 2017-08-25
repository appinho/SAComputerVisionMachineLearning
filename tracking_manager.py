from track import Track

class TrackingManager(object):
    def __init__(self):
        self.number_of_tracks = 0
        self.list_of_tracks = []
        self.list_of_objects = []
    def process(self,objects):
        self.list_of_objects = objects
        if not self.list_of_tracks:
            for object in self.list_of_objects:
                self.init_track(object)
        else:
            pass
    def init_track(self,object):
        tr = Track(object)
        self.list_of_tracks.append(tr)
    def predict(self):
        pass
    def associate(self):
        pass
    def update(self):
        pass
    def get_tracks(self):
        return self.list_of_tracks