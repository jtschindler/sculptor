#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class SpecFitCanvas(FigureCanvas):

    """A FigureCanvas for plotting an astronomical spectrum from a SpecFit
    object.

    This class provides the plotting routine for the SpecFitGui.

    Attributes:
        fig (matplotlib.figure.Figure): Figure object for the plot.
        ax_main (matplotlib.axes.Axis): Axis for main plot
        ax_resid (matplotlib.axes.Axis): Axis for the residual plot
    """

    def __init__(self, specfit=None):
        """ Initilization method for the SpecFitCanvas

        :param (SpecFit) specfit: SpecFit object, which holds information on
            the astronomical spectrum and its fit.
        """


        self.fig = plt.figure()

        super().__init__(self.fig)
        # FigureCanvas.__init__(self, self.fig)

        gs = gridspec.GridSpec(4, 1)
        self.ax_main = self.fig.add_subplot(gs[:3, 0])
        self.ax_resid = self.fig.add_subplot(gs[3:4, 0], sharex=self.ax_main)


        if specfit is not None:
            self.plot(specfit)

        self.ax_main.set_xlim(specfit.xlim)
        self.ax_main.set_ylim(specfit.ylim)

    def plot(self, specfit):
        """
        Plot the spectrum and the models

        :param (SpecFit) specfit: SpecFit object, which holds information on
            the astronomical spectrum and its fit.

        :return: None
        """

        # Clear all axes
        self.ax_main.clear()
        self.ax_resid.clear()

        if specfit.spec is not None:
            specfit._plot_specfit(self.ax_main,
                                  self.ax_resid)

        self.ax_main.set_xlim(specfit.xlim)
        self.ax_main.set_ylim(specfit.ylim)

        self.ax_resid.set_xlim(specfit.xlim)
        dylim = abs(specfit.ylim[1] - specfit.ylim[0])/4
        self.ax_resid.set_ylim([-dylim, dylim])

        self.draw()