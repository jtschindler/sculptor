#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class SpecModelCanvas(FigureCanvas):

    """A FigureCanvas for plotting an astronomical spectrum from a SpecModel
    object.

    This class provides the plotting routine for the SpecModelWidget.

    Attributes:
        fig (matplotlib.figure.Figure): Figure object for the plot.
        ax_main (matplotlib.axes.Axis): Axis for main plot
    """


    def __init__(self, specmodel):
        """ Initilization method for the SpecFitCanvas

        :param (SpecModel) specmodel: SpecModel object, which holds
        information on the astronomical spectrum and its fit.
        """


        self.fig = plt.figure()
        super().__init__(self.fig)

        self.ax_main = self.fig.add_subplot()

        if hasattr(specmodel, 'spec'):
            if specmodel.spec is not None:
                specmodel._plot_specmodel(self.ax_main)

        self.ax_main.set_xlim(specmodel.xlim)
        self.ax_main.set_ylim(specmodel.ylim)


        self.draw()


    def plot(self, specmodel):
        """
        Plot the spectrum and the model

        :param (SpecModel) specmodel: SpecModel object, which holds
        information on the astronomical spectrum and its fit.

        :return: None
        """

        # Clear all axes
        self.ax_main.clear()


        if hasattr(specmodel, 'spec'):
            if specmodel.spec is not None:
                specmodel._plot_specmodel(self.ax_main)


        self.ax_main.set_xlim(specmodel.xlim)
        self.ax_main.set_ylim(specmodel.ylim)

        self.draw()

