
import numpy as np

from PyQt5 import QtWidgets, QtCore

from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction, \
    QLabel, QTabWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QLineEdit, \
    QComboBox, QCheckBox, QGroupBox, QScrollArea

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
    as NavigationToolbar

from sculptor.specfit import SpecFit
from sculptor.specmodel import mask_presets, fitting_methods
from sculptor.specmodelwidget import SpecModelWidget
from sculptor.specfitcanvas import SpecFitCanvas
from sculptor.menu_dialogs import ResampleWindow, EmceeWindow

def update_float_variable_from_linedit(lineEdit, variable,
                                       expression='{:.2f}'):

    try:
        new_value = float(lineEdit.text())
        lineEdit.setText(expression.format(new_value))
        return new_value
    except:
        print('Input value is not convertable to float.')
        return variable



class SpecFitGui(QMainWindow):
    """

    """


    def __init__(self, spectrum=None, redshift=0):

        QMainWindow.__init__(self)

        self.resize(1150, 824)
        self.setWindowTitle("Sculptor - Interactive Modelling of astronomic electromagnetic spectra")

        # Add the SpecFit class to the GUI
        self.specfit = SpecFit(spectrum=spectrum, redshift=redshift)

        # Initialize class variables
        self.x_pos_a = 0
        self.x_pos_b = 0
        self.y_pos_a = 0
        self.y_pos_b = 0

        # Random seed variable


        # Resampling variables
        self.nsamples = 100
        self.resample_seed = 1234
        self.save_result_plots = True
        self.resample_foldername = '.'

        # MCMC variables


        # Setup the main menu
        # Add exit action
        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)
        # Add save action
        saveAction = QAction('&Save', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save to SpecFit Folder')
        saveAction.triggered.connect(self.save)
        # Add load action
        loadAction = QAction('&Load', self)
        loadAction.setShortcut('Ctrl+L')
        loadAction.setStatusTip('Load from SpecFit Folder')
        loadAction.triggered.connect(self.load)

        # Add import spectrum action
        importIrafSpecAction = QAction('Import IRAF spectrum', self)
        # loadAction.setShortcut('Ctrl+L')
        importIrafSpecAction.setStatusTip('Import IRAF Spectrum. Overrides '
                                          'current spectrum')
        importIrafSpecAction.triggered.connect(lambda: self.import_spectrum(
            mode='IRAF'))

        importPypeitSpecAction = QAction('Import PypeIT spectrum', self)
        # loadAction.setShortcut('Ctrl+L')
        importPypeitSpecAction.setStatusTip('Import IRAF Spectrum. Overrides '
                                          'current spectrum')
        importPypeitSpecAction.triggered.connect(lambda: self.import_spectrum(
            mode='PypeIT'))

        importSodSpecAction = QAction('Import SpecOneD spectrum', self)
        # loadAction.setShortcut('Ctrl+L')
        importSodSpecAction.setStatusTip('Import IRAF Spectrum. Overrides '
                                          'current spectrum')
        importSodSpecAction.triggered.connect(lambda: self.import_spectrum(
            mode='SpecOneD'))

        # SpecModel Actions
        addSpecModelAction = QAction('Add SpecModel', self)
        addSpecModelAction.setStatusTip('Add a SpecModel to the fit')
        addSpecModelAction.triggered.connect(self.add_specmodel)

        removeSpecModelAction = QAction('Remove current SpecModel', self)
        removeSpecModelAction.setStatusTip('Remove current SpecModel from fit')
        removeSpecModelAction.triggered.connect(self.remove_current_spec_model)

        resetSpecModelAction = QAction('Remove all SpecModels', self)
        resetSpecModelAction.setStatusTip('Remove all SpecModels from fit')
        resetSpecModelAction.triggered.connect(self.remove_all_models)

        # Fit Actions

        runResampleAction = QAction('Run resample and fit', self)
        runResampleAction.setStatusTip('Resample the spectrum, fit all models '
                                       'and save the posterior parameter '
                                       'distributions.')
        runResampleAction.triggered.connect(self.resample_and_fit)

        setResampleAction = QAction('Set resample and fit parameters', self)
        setResampleAction.setStatusTip('Set the resample and fit parameters')
        setResampleAction.triggered.connect(self.open_resample_dialog)

        setEmceeAction = QAction('Set MCMC parameters', self)
        setEmceeAction.setStatusTip('Set the MCMC fit parameters')
        setEmceeAction.triggered.connect(self.open_emcee_dialog)

        self.statusBar()

        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        fileMenu = mainMenu.addMenu('&File')
        # fileMenu.addSeparator()
        fileMenu.addAction(loadAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(exitAction)

        spectrumMenu = mainMenu.addMenu('&Spectrum')
        spectrumMenu.addAction(importIrafSpecAction)
        spectrumMenu.addAction(importPypeitSpecAction)
        spectrumMenu.addAction(importSodSpecAction)

        specModelMenu = mainMenu.addMenu('&SpecModel')
        specModelMenu.addAction(addSpecModelAction)
        specModelMenu.addAction(removeSpecModelAction)
        specModelMenu.addAction(resetSpecModelAction)

        fitMenu = mainMenu.addMenu('&Fit')
        fitMenu.addAction(setEmceeAction)
        fitMenu.addAction(setResampleAction)
        fitMenu.addAction(runResampleAction)


        # Setup the central widget and and the main tab widget
        self.centralwidget = QWidget()
        self.mainVLayout = QVBoxLayout(self.centralwidget)
        self.mainHLayout = QHBoxLayout()


        self.setCentralWidget(self.centralwidget)
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setTabShape(QTabWidget.Rounded)
        self.tabWidget.setUsesScrollButtons(False)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.currentChanged.connect(self.tabchanged)

        self.mainHLayout.addWidget(self.tabWidget)
        self.mainHLayout.setStretch(1, 2)

        self.mainVLayout.addLayout(self.mainHLayout)

        self.initialize_main_tab()

        # Initialize the Key Press Event for the SpecFitCanvas
        self.gcid = self.specFitCanvas.mpl_connect('key_press_event',
                                                   self.on_press)

        self.show()



    def initialize_main_tab(self):

        # Initialize the fit tab
        self.fittab = QWidget()
        self.hLayoutFit = QHBoxLayout(self.fittab)


        self.vLayoutFitProperties = QVBoxLayout()


        # Add positional input
        self.boxRegionSelect = QGroupBox('Region Select')
        self.labelDispersionPos = QLabel('Dispersion region (Shift + a/d)')
        self.leXposA = QLineEdit('{:.2f}'.format(self.specfit.xlim[0]))
        self.leXposB = QLineEdit('{:.2f}'.format(self.specfit.xlim[1]))
        self.hLayoutXpos = QHBoxLayout()
        self.hLayoutXpos.addWidget(self.leXposA)
        self.hLayoutXpos.addWidget(self.leXposB)
        self.labelFluxPos = QLabel('Flux region (Shift + s/d)')
        self.leYposA = QLineEdit('{:.2e}'.format(self.specfit.ylim[0]))
        self.leYposB = QLineEdit('{:.2e}'.format(self.specfit.ylim[1]))
        self.hLayoutYpos = QHBoxLayout()
        self.hLayoutYpos.addWidget(self.leYposA)
        self.hLayoutYpos.addWidget(self.leYposB)

        pos_le = [self.leXposA, self.leXposB, self.leYposA, self.leYposB]
        for le in pos_le:
            le.returnPressed.connect(self.update_region_from_ui)

        self.buttonSetX = QPushButton('Set X (x)')
        self.buttonSetX.clicked.connect(self.set_plot_dispersion)
        self.buttonSetY = QPushButton('Set Y (y)')
        self.buttonSetY.clicked.connect(self.set_plot_fluxden)
        self.buttonResetPlot = QPushButton('Reset plot (r)')
        self.buttonResetPlot.clicked.connect(self.reset_plot_region)

        self.hLayoutRegionButtons = QHBoxLayout()
        self.hLayoutRegionButtons.addWidget(self.buttonSetX)
        self.hLayoutRegionButtons.addWidget(self.buttonSetY)
        self.hLayoutRegionButtons.addWidget(self.buttonResetPlot)

        self.vLayoutBoxRegionSelect = QVBoxLayout(self.boxRegionSelect)
        self.vLayoutBoxRegionSelect.addWidget(self.labelDispersionPos)
        self.vLayoutBoxRegionSelect.addLayout(self.hLayoutXpos)
        self.vLayoutBoxRegionSelect.addWidget(self.labelFluxPos)
        self.vLayoutBoxRegionSelect.addLayout(self.hLayoutYpos)
        self.vLayoutBoxRegionSelect.addLayout(self.hLayoutRegionButtons)


        # Masking
        self.boxMaskSelect = QGroupBox('Masking')
        self.vLayoutBoxMaskSelect = QVBoxLayout(self.boxMaskSelect)

        self.boxMaskPreset = QComboBox()
        for mask_preset_key in mask_presets.keys():
            self.boxMaskPreset.addItem(mask_preset_key)

        self.buttonLoadPreset = QPushButton('Load mask preset')
        self.buttonLoadPreset.clicked.connect(lambda:
                                              self.load_mask_preset())

        self.hLayoutMaskPreset = QHBoxLayout()
        self.hLayoutMaskPreset.addWidget(self.boxMaskPreset)
        self.hLayoutMaskPreset.addWidget(self.buttonLoadPreset)

        self.buttonMask = QPushButton('Mask (m)')
        self.buttonMask.clicked.connect(lambda: self.update_mask(mode='mask'))
        self.buttonUnmask = QPushButton('Unmask (u)')
        self.buttonUnmask.clicked.connect(lambda: self.update_mask(mode='unmask'))
        self.hLayoutMaskAction = QHBoxLayout()
        self.buttonResetMask = QPushButton('Reset mask (R)')
        self.buttonResetMask.clicked.connect(lambda: self.reset_mask())

        self.hLayoutMaskAction.addWidget(self.buttonMask)
        self.hLayoutMaskAction.addWidget(self.buttonUnmask)
        self.hLayoutMaskAction.addWidget(self.buttonResetMask)

        self.vLayoutBoxMaskSelect.addLayout(self.hLayoutMaskAction)
        self.vLayoutBoxMaskSelect.addLayout(self.hLayoutMaskPreset)

        # Super parameters
        # Add global parameter
        self.boxSuperParam = QGroupBox('Super parameters')
        self.vLayoutSuperParam = QVBoxLayout(self.boxSuperParam)
        self.hLayoutAddSuperParam = QHBoxLayout()
        self.hLayoutDelSuperParam = QHBoxLayout()
        self.leSuperParamName = QLineEdit('super_param')
        self.leSuperParamName.setMaxLength(20)
        self.buttonAddSuperParam = QPushButton('Add super parameter')
        self.buttonAddSuperParam.clicked.connect(self.add_super_param)

        self.boxSelectSuperParam = QComboBox()
        self.buttonDelSuperParam = QPushButton('Remove super parameter')
        self.buttonDelSuperParam.clicked.connect(self.remove_super_param)

        self.hLayoutAddSuperParam.addWidget(self.leSuperParamName)
        self.hLayoutAddSuperParam.addWidget(self.buttonAddSuperParam)
        self.hLayoutDelSuperParam.addWidget(self.boxSelectSuperParam)
        self.hLayoutDelSuperParam.addWidget(self.buttonDelSuperParam)

        self.vLayoutSuperParam.addLayout(self.hLayoutAddSuperParam)
        self.vLayoutSuperParam.addLayout(self.hLayoutDelSuperParam)

        # Redshift parameter
        self.labelRedshift = QLabel('Redshift')
        self.leRedshift = QLineEdit()
        if self.specfit.redshift is not None:
            self.leRedshift.setText('{}'.format(self.specfit.redshift))
        else:
            self.leRedshift.setText('None')
        self.leRedshift.setMaxLength(20)
        self.leRedshift.returnPressed.connect(self.update_specfit_from_ui)

        self.hLayoutRedshift = QHBoxLayout()
        self.hLayoutRedshift.addWidget(self.labelRedshift)
        self.hLayoutRedshift.addWidget(self.leRedshift)

        # Fitting
        self.buttonFit = QPushButton('Fit all')
        self.buttonFit.clicked.connect(self.fit)
        self.buttonFitSaveResults = QPushButton('Fit all + Save results')
        self.buttonFitSaveResults.clicked.connect(lambda: self.fit(
            save_results=True))
        self.boxFittingMethod = QComboBox()
        for key in fitting_methods.keys():
            self.boxFittingMethod.addItem(key)
        self.boxFittingMethod.currentTextChanged.connect(
            self.update_fitting_method)

        # Build vLayoutFitProperties
        self.vLayoutFitProperties.addWidget(self.boxRegionSelect)
        self.vLayoutFitProperties.addWidget(self.boxMaskSelect)
        self.vLayoutFitProperties.addWidget(self.boxSuperParam)
        self.vLayoutFitProperties.addLayout(self.hLayoutRedshift)
        self.vLayoutFitProperties.addWidget(self.boxFittingMethod)
        self.vLayoutFitProperties.addWidget(self.buttonFit)
        self.vLayoutFitProperties.addWidget(self.buttonFitSaveResults)

        # Add SpecFitCanvas
        self.specFitCanvas = SpecFitCanvas(self.specfit)
        self.toolbar = NavigationToolbar(self.specFitCanvas, self)
        self.vLayoutCanvas = QVBoxLayout()
        self.vLayoutCanvas.addWidget(self.specFitCanvas)
        self.vLayoutCanvas.addWidget(self.toolbar)

        # Set the ClickFocus active on the SpecFitCanvas
        self.specFitCanvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.specFitCanvas.setFocus()

        # Super parameter scroll area widget
        self.build_super_param_widget()

        # Build hLayoutFit
        self.hLayoutFit.addLayout(self.vLayoutFitProperties)
        self.hLayoutFit.addLayout(self.vLayoutCanvas)
        self.hLayoutFit.setStretch(1, 2)

        # Add super parameter tab widget
        self.vLayoutCanvas.addWidget(self.ScArea)
        self.vLayoutCanvas.setStretch(0, 2)
        

        self.tabWidget.addTab(self.fittab, "Fit")


    def build_super_param_widget(self):

        superParamWidget = QWidget()
        hLayoutModel = QHBoxLayout(superParamWidget)

        superParams = self.specfit.super_params

        lineditlist = []
        varyboxlist = []
        widgetlist = []

        for jdx, param in enumerate(superParams):

            widgetlist = []

            groupBoxParam = QGroupBox(param)
            vLayoutGroupBoxParam = QVBoxLayout(groupBoxParam)

            label = QLabel(param)
            linedit = QLineEdit('{:.4E}'.format(superParams[param].value))
            linedit.setMaxLength(20)
            expr_linedit = QLineEdit('{}'.format(superParams[param].expr))
            expr_linedit.setMaxLength(20)
            min_label = QLabel("min")
            min_linedit = QLineEdit('{:.4E}'.format(superParams[param].min))
            min_linedit.setMaxLength(20)
            max_label = QLabel("max")
            max_linedit = QLineEdit('{:.4E}'.format(superParams[param].max))
            max_linedit.setMaxLength(20)
            vary_checkbox = QCheckBox("vary")
            vary_checkbox.setChecked(superParams[param].vary)

            widgetlist.extend(
                [label, linedit, expr_linedit, min_label, min_linedit,
                 max_label, max_linedit, vary_checkbox])
            lineditlist.extend(
                [linedit, expr_linedit, min_linedit, max_linedit])
            varyboxlist.append(vary_checkbox)

            for w in widgetlist:
                vLayoutGroupBoxParam.addWidget(w)

            hLayoutModel.addWidget(groupBoxParam)

            vary_checkbox.stateChanged.connect(self.update_super_params_from_ui)

        if widgetlist:
            self.super_params_widgetlist = widgetlist
            self.super_params_lineditlist = lineditlist
            self.super_params_varybox_list = varyboxlist

            # Activate input for lineEdit lists
            for l in lineditlist:
                l.returnPressed.connect(self.update_super_params_from_ui)

        # Add model parameters in scroll area
        if hasattr(self, 'ScArea'):
            self.vLayoutCanvas.removeWidget(self.ScArea)
            self.deleteItemsOfLayout(self.ScArea.layout())
        else:
            self.ScArea = QScrollArea()
            self.ScArea.setLayout(QHBoxLayout())
            self.ScArea.setWidgetResizable(True)
            self.ScArea.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOn)

        self.ScArea.setWidget(superParamWidget)

        self.vLayoutCanvas.addWidget(self.ScArea)

    def on_press(self, event):

        if event.key == 'm':

            self.update_mask(mode='mask')

        elif event.key == 'u':

            self.update_mask(mode='unmask')

        elif event.key == 'f':

            self.fit()

        elif event.key == 'A':

            self.x_pos_a = event.xdata
            self.leXposA.setText('{:.2f}'.format(self.x_pos_a))


        elif event.key == 'D':

            self.x_pos_b = event.xdata
            self.leXposB.setText('{:.2f}'.format(self.x_pos_b))

        elif event.key == 'W':

            self.y_pos_b = event.ydata
            self.leYposB.setText('{:.2e}'.format(self.y_pos_b))


        elif event.key == 'S':

            self.y_pos_a = event.ydata
            self.leYposA.setText('{:.2e}'.format(self.y_pos_a))

        # Zoom into X-axis region
        elif event.key == 'x':

            self.set_plot_dispersion()

        elif event.key == 'y':

            self.set_plot_fluxden()

        # Reset the region values
        elif event.key == 'r':

            self.reset_plot_region()

        # Full reset (region + mask)
        elif event.key == 'R':
            self.reset_mask()

    # --------------------------------------------------------------------------
    # Region Actions
    # --------------------------------------------------------------------------

    def set_plot_dispersion(self):

        self.update_region_from_ui()
        self.specfit.xlim = [self.x_pos_a, self.x_pos_b]
        self.update_specfit_plot()

    def set_plot_fluxden(self):

        self.update_region_from_ui()
        self.specfit.ylim = [self.y_pos_a, self.y_pos_b]
        self.update_specfit_plot()

    def reset_plot_region(self):

        self.reset_region()
        self.update_specfit_plot()

    def reset_region(self):

        if self.specfit.spec is not None:
            self.x_pos_a = min(self.specfit.spec.dispersion)
            self.x_pos_b = max(self.specfit.spec.dispersion)
            self.y_pos_a = min(self.specfit.spec.fluxden)
            self.y_pos_b = max(self.specfit.spec.fluxden)

        else:
            self.x_pos_a = 0
            self.x_pos_b = 1
            self.y_pos_a = 0
            self.y_pos_b = 1

        # Excluded, to get back to previous region
        # self.leXposA.setText('{:.2f}'.format(self.x_pos_a))
        # self.leXposB.setText('{:.2f}'.format(self.x_pos_b))
        # self.leYposA.setText('{:.2E}'.format(self.y_pos_a))
        # self.leYposB.setText('{:.2E}'.format(self.y_pos_b))

        self.specfit.xlim = [self.x_pos_a, self.x_pos_b]
        self.specfit.ylim = [self.y_pos_a, self.y_pos_b]

    def update_region_from_ui(self):

        self.x_pos_a = update_float_variable_from_linedit(self.leXposA,
                                                          self.x_pos_a)
        self.x_pos_b = update_float_variable_from_linedit(self.leXposB,
                                                          self.x_pos_b)
        self.y_pos_a = update_float_variable_from_linedit(self.leYposA,
                                                          self.y_pos_a,
                                                          expression='{:.2E}')
        self.y_pos_b = update_float_variable_from_linedit(self.leYposB,
                                                          self.y_pos_b,
                                                          expression='{:.2E}')

        self.specFitCanvas.setFocus()

    # --------------------------------------------------------------------------
    # Mask Actions
    # --------------------------------------------------------------------------

    def update_mask(self, mode='mask'):

        self.x_pos_a = float(self.leXposA.text())
        self.x_pos_b = float(self.leXposB.text())

        self.mask_region(mode)

        self.update_specfit_plot()

    def mask_region(self, mode='mask'):

        if self.specfit.spec is not None:

            mask_between = np.sort(np.array([self.x_pos_a,
                                             self.x_pos_b]))

            spec = self.specfit.spec
            lo_index = np.argmin(np.abs(spec.dispersion - mask_between[0]))
            up_index = np.argmin(np.abs(spec.dispersion - mask_between[1]))
            if mode == 'mask':
                self.specfit.spec.mask[lo_index:up_index] = False
            elif mode == 'unmask':
                self.specfit.spec.mask[lo_index:up_index] = True

        self.specfit.update_specmodel_spectra()


    def reset_mask(self):

        self.reset_region()
        self.specfit.spec.reset_mask()
        self.update_specfit_plot()

        self.specfit.update_specmodel_spectra()


    def load_mask_preset(self):

        mask_preset_key = self.boxMaskPreset.currentText()

        mask_preset = mask_presets[mask_preset_key]

        if mask_preset['rest_frame']:
            one_p_z = 1 + self.specfit.redshift
        else:
            one_p_z = 1

        for mask_range in mask_preset['mask_ranges']:
            wave_a = mask_range[0] * one_p_z
            wave_b = mask_range[1] * one_p_z

            if self.specfit.spec is not None:
                self.specfit.spec.mask_between([wave_a, wave_b])

        self.update_specfit_plot()

    def update_specfit_from_ui(self):

        # update redshift
        self.specfit.redshift = float(self.leRedshift.text())
        self.specfit.update_specmodels()
        self.leRedshift.setText('{:.4f}'.format(self.specfit.redshift))


        self.specFitCanvas.setFocus()

    # --------------------------------------------------------------------------
    # Super Param Actions
    # --------------------------------------------------------------------------

    def update_boxSelectSuperParam(self):
        """ Only used when loading a model to update the Combobox"""

        for param in self.specfit.super_params:
            if self.boxSelectSuperParam.findText(param) < 0:
                self.boxSelectSuperParam.addItem(param)

    def rebuild_super_params_widget(self):

        layout = self.ScArea.layout()
        self.deleteItemsOfLayout(layout)

        self.build_super_param_widget()

    def add_super_param(self):

        param_name = self.leSuperParamName.text()
        # Check if item already in QComboBox
        if self.boxSelectSuperParam.findText(param_name) < 0:
            # Add item to QComboBox
            self.boxSelectSuperParam.addItem(param_name)
            # Add param to SpecModel
            self.specfit.add_super_param(param_name)

        self.rebuild_super_params_widget()
        self.update_specmodelwidgets_for_global_params()

    def remove_super_param(self):

        param_name = self.boxSelectSuperParam.currentText()
        # Remove global param from SpecModel
        self.specfit.remove_super_param(param_name)
        # Remove item from QComboBox
        idx = self.boxSelectSuperParam.findText(param_name)
        self.boxSelectSuperParam.removeItem(idx)

        self.rebuild_super_params_widget()
        self.update_specmodelwidgets_for_global_params()


    def update_specmodelwidgets_for_global_params(self):

        for idx in range(self.tabWidget.count() - 1):
            self.tabWidget.setCurrentIndex(idx + 1)
            specmodel_widget = self.tabWidget.currentWidget()
            specmodel_widget.rebuild_global_params_tab()
            specmodel_widget.rebuild_model_tabs()

        self.tabWidget.setCurrentIndex(0)
        self.specFitCanvas.setFocus()

    # --------------------------------------------------------------------------
    # Fit Actions
    # --------------------------------------------------------------------------

    def update_fitting_method(self,value):

        self.specfit.fitting_method = value

    def fit(self, save_results=False):

        if len(self.specfit.specmodels) > 0 and self.specfit.spec is not None:

            if save_results:
                # Select save folder
                foldername = str(
                    QFileDialog.getExistingDirectory(self, "Select Directory"))
                if foldername:
                    self.specfit.fit(save_results=save_results,
                                     foldername=foldername)
                    self.specFitCanvas.fig.savefig(foldername+'/SpecFit.png')
                else:
                    self.statusBar().showMessage('No SAVE folder selected.')
            else:
                self.specfit.fit()

            self.rebuild_super_params_widget()
            self.update_specfit_plot()

    def resample_and_fit(self):

        if len(self.specfit.specmodels) == 0:
            self.statusBar().showMessage('[ERROR] No SpecModel found. Cannot '
                                         'resample and fit spectrum.')

        else:

            self.specfit.resample(n_samples=self.nsamples,
                                  foldername=self.resample_foldername,
                                  save_result_plots=self.save_result_plots,
                                  seed=self.resample_seed)

    def open_resample_dialog(self):
        self.resampleDialog = ResampleWindow(self)
        self.resampleDialog.show()


    def open_emcee_dialog(self):
        self.emceeDialog = EmceeWindow(self)
        self.emceeDialog.show()

    # --------------------------------------------------------------------------
    # File Actions
    # --------------------------------------------------------------------------

    def closeEvent(self, event):
        """This function modifies the standard closing of the GUI

        Parameters
        ----------
        event : event
            The closing event.
        """

        result = QtWidgets.QMessageBox.question(self,
                      "Exit Dialog",
                      "Are you sure you want to exit ?",
                      QtWidgets.QMessageBox.Yes| QtWidgets.QMessageBox.No)
        event.ignore()

        if result == QtWidgets.QMessageBox.Yes:
            event.accept()

    def save(self):

        # Select save folder
        foldername = str(
            QFileDialog.getExistingDirectory(self, "Select Directory"))

        if foldername:
            self.specfit.save(foldername)
        else:
            self.statusBar().showMessage('[WARNING] No save directory '
                                         'selected.')


    def load(self):

        # Select load folder
        foldername = str(
            QFileDialog.getExistingDirectory(self, "Select Directory"))

        if foldername:
            # Remove all SpecModels
            self.remove_all_models()

            self.specfit.load(foldername)

            # Rebuild TabWidget with loaded SpecModels
            self.add_specmodels_in_specfit()

            self.tabWidget.setCurrentIndex(0)
            # self.reset_region()
            self.update_specfit_plot()
            self.update_boxSelectSuperParam()
            self.rebuild_super_params_widget()

        else:
            self.statusBar().showMessage('[WARNING] No directory to '
                                         'load a model from selected.')


    # --------------------------------------------------------------------------
    # Spectrum Actions
    # --------------------------------------------------------------------------

    def import_spectrum(self, mode='IRAF'):
        """ Import spectrum """

        # Select spectrum
        fileName, fileFilter = QFileDialog.getOpenFileName(self, "Import "
                                                               "spectrum")


        self.specfit.import_spectrum(fileName, filetype=mode)

        # Re-plot main canvas -> later update fittab
        self.reset_region()
        self.specFitCanvas.plot(self.specfit)

        # Re-plot all SpecModel tabs
        for idx in range(self.tabWidget.count()-1):

            self.tabWidget.setCurrentIndex(idx+1)
            specmodel_widget = self.tabWidget.currentWidget()
            specmodel_widget.reset_plot_region()
            specmodel_widget.update_specmodel_plot()

    # --------------------------------------------------------------------------
    # SpecModel Actions
    # --------------------------------------------------------------------------

    def add_specmodel(self):

        # Add SpecModel to SpecFit
        self.specfit.add_specmodel()

        specmodel_widget = SpecModelWidget(self, self.specfit.specmodels[-1])

        self.add_specmodel_tab(specmodel_widget)

    def add_specmodel_tab(self, specmodel_widget, specmodel_name='SpecModel'):

        index = self.tabWidget.addTab(specmodel_widget, specmodel_name)
        self.tabWidget.setCurrentIndex(index)

    def add_specmodels_in_specfit(self):

        for specmodel in self.specfit.specmodels:
            specmodel_widget = SpecModelWidget(self, specmodel)
            self.add_specmodel_tab(specmodel_widget,
                                   specmodel_name=specmodel.name)

            specmodel_widget.rebuild_model_tabs()
            specmodel_widget.update_boxSelectGlobalParam()

    def remove_all_models(self):

        for tabindex in reversed(range(self.tabWidget.count())[1:]):
            self.remove_spec_model(tabindex)

    def remove_current_spec_model(self):

        # Get current SpecModel index from TabWidget
        tabindex = self.tabWidget.currentIndex()
        self.remove_spec_model(tabindex)

    #     TODO UPDATE ALL TABS WITH REGARD TO THE PROPAGATED SPECTRUM

    def remove_spec_model(self, index):

        if index > 0:

            # Remove SpecModel from SpecFit
            self.specfit.delete_specmodel(index-1)
            # Update the TabWidget accordingly
            specmodel_widget = self.tabWidget.currentWidget()
            specmodel_widget.deleteLater()
            self.tabWidget.removeTab(index)

        else:
            self.statusBar().showMessage('Current tab does not contain a '
                                         'SpecModel')

    # --------------------------------------------------------------------------
    # Update Actions (SpecModelWidget, SpecModel, TabWidget)
    # --------------------------------------------------------------------------

    def update_specfit_plot(self):
        self.specFitCanvas.plot(self.specfit)

    def tabchanged(self):

        # Get current tab widget
        idx = self.tabWidget.currentIndex()

        # Update the tab
        if idx == 0:
            self.specFitCanvas.plot(self.specfit)
            self.update_specfit_plot()
            self.specFitCanvas.setFocus()

        elif idx > 0:
            self.specfit.update_specmodel_spectra()
            self.update_specmodel_tab(idx)

    def update_specmodel_tab(self, index):

        self.tabWidget.setCurrentIndex(index)

        # UPDATE SpecMODEL???

        specmodel_widget = self.tabWidget.currentWidget()
        # specmodel_widget.reset_plot_region()
        specmodel_widget.update_specmodel_plot()
        specmodel_widget.specModelCanvas.setFocus()


    def update_super_params_from_ui(self):

        for jdx, param in enumerate(self.specfit.super_params):

            new_value = float(
                self.super_params_lineditlist[jdx * 4 + 0].text())
            new_expr = self.super_params_lineditlist[jdx * 4 + 1].text()
            new_min = float(
                self.super_params_lineditlist[jdx * 4 + 2].text())
            new_max = float(
                self.super_params_lineditlist[jdx * 4 + 3].text())
            new_vary = self.super_params_varybox_list[jdx].isChecked()

            if new_expr == 'None':
                new_expr = None

            for specmodel in self.specfit.specmodels:

                # Set the new parameter values
                specmodel.global_params[param].set(value=new_value,
                                                       expr=new_expr,
                                                       min=new_min,
                                                       max=new_max,
                                                       vary=new_vary)

        self.update_specmodelwidgets_for_global_params()


    def deleteItemsOfLayout(self, layout):
         if layout is not None:
             while layout.count():
                 item = layout.takeAt(0)
                 widget = item.widget()
                 if widget is not None:
                     widget.setParent(None)
                 else:
                     self.deleteItemsOfLayout(item.layout())