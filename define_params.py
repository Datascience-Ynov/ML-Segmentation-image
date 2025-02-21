import abc

class DefineParams(abc.ABC):    
    @abc.abstractmethod
    def resize_image(self):
        pass

    @abc.abstractmethod
    def segment_algo(self):
        pass

    @abc.abstractmethod
    def create_segmented_image(self):
        pass

    @abc.abstractmethod
    def color_image(self):
        pass

    @abc.abstractmethod
    def plot_3d(self):
        pass