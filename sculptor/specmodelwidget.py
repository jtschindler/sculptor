

import numpy as np
import corner
import re
from lmfit import fit_report
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QPushButton, \
    QLabel, QTabWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QLineEdit, \
    QComboBox, QCheckBox, QGroupBox, QScrollArea, QMessageBox
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT \
    as NavigationToolbar

from sculptor.masksmodels import model_func_list, \
    model_setup_list, mask_presets
from sculptor.specmodelcanvas import SpecModelCanvas



def update_float_variable_from_linedit(lineEdit, variable,
                                       expression='{:.2f}'):

    try:
        new_value = float(lineEdit.text())
        lineEdit.setText(expression.format(new_value))
        return new_value
    except:
        print('[INFO] Input value is not convertable to float.')
        return variable




class SpecModelWidget(QWidget):

    def __init__(self, specfitgui, specmodel):

        super().__init__()


        # Initialize class variables
        self.x_pos_a = 0
        self.x_pos_b = 0
        self.y_pos_a = 0
        self.y_pos_b = 0

        self.specmodel = specmodel

        # Set up the SpecModelWidget
        self.vLayout = QVBoxLayout(self)
        self.hLayout = QHBoxLayout()

        self.specfitgui = specfitgui

        # Show Fit Report
        self.showFitReport = False

        self.specModelCanvas = SpecModelCanvas(specmodel)
        self.toolbar = NavigationToolbar(self.specModelCanvas, self)
        self.vLayoutCanvas = QVBoxLayout()
        self.vLayoutCanvas.addWidget(self.specModelCanvas)
        self.vLayoutCanvas.addWidget(self.toolbar)

        # SpecModel name
        self.labelName = QLabel('SpecModel name')
        self.leName = QLineEdit('{}'.format(self.specmodel.name))
        self.leName.setMaxLength(20)
        self.leName.returnPressed.connect(lambda: self.update_specmodel(specfitgui))
        self.hLayoutName = QHBoxLayout()
        self.hLayoutName.addWidget(self.labelName)
        self.hLayoutName.addWidget(self.leName)

        # Add positional input
        self.boxRegionSelect = QGroupBox('Region Select')
        self.labelDispersionPos = QLabel('Dispersion region (Shift + a/d)')
        self.leXposA = QLineEdit('{:.2f}'.format(self.specmodel.xlim[0]))
        self.leXposB = QLineEdit('{:.2f}'.format(self.specmodel.xlim[1]))
        self.hLayoutXpos = QHBoxLayout()
        self.hLayoutXpos.addWidget(self.leXposA)
        self.hLayoutXpos.addWidget(self.leXposB)
        self.labelFluxPos = QLabel('Flux region (Shift + w/s)')
        self.leYposA = QLineEdit('{:.2e}'.format(self.specmodel.ylim[0]))
        self.leYposB = QLineEdit('{:.2e}'.format(self.specmodel.ylim[1]))
        self.hLayoutYpos = QHBoxLayout()
        self.hLayoutYpos.addWidget(self.leYposA)
        self.hLayoutYpos.addWidget(self.leYposB)

        pos_le = [self.leXposA, self.leXposB, self.leYposA, self.leYposB]
        for le in pos_le:
            le.returnPressed.connect(self.update_region_from_ui)

        self.buttonSetX = QPushButton('Set dispersion range (x)')
        self.buttonSetX.clicked.connect(self.set_plot_dispersion)
        self.buttonSetY = QPushButton('Set flux range (y)')
        self.buttonSetY.clicked.connect(self.set_plot_flux)
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

        # Add model function to SpecModel
        self.boxModelSelect = QGroupBox('Model Select')
        self.vLayoutBoxModelSelect = QVBoxLayout(self.boxModelSelect)

        self.boxAddModel = QComboBox()
        for model_func in model_func_list:
            self.boxAddModel.addItem(model_func)

        self.buttonAddModel = QPushButton('Add')
        self.buttonAddModel.clicked.connect(lambda:
                                            self.add_model_to_specmodel(
                                                specmodel, specfitgui))
        self.hLayoutAddModel = QHBoxLayout()
        self.hLayoutAddModel.addWidget(self.boxAddModel)
        self.hLayoutAddModel.addWidget(self.buttonAddModel)

        # Model function prefix
        self.labelPrefix = QLabel('Model prefix')
        self.lePrefix = QLineEdit('modelA')
        self.lePrefix.setMaxLength(20)
        self.hLayoutPrefix = QHBoxLayout()
        self.hLayoutPrefix.addWidget(self.labelPrefix)
        self.hLayoutPrefix.addWidget(self.lePrefix)

        # Delete model function from SpecModel
        self.boxRemoveModel = QComboBox()
        self.buttonRemoveModel = QPushButton('Remove')
        self.buttonRemoveModel.clicked.connect(self.remove_model_from_specmodel)
        self.hLayoutRemoveModel = QHBoxLayout()
        self.hLayoutRemoveModel.addWidget(self.boxRemoveModel)
        self.hLayoutRemoveModel.addWidget(self.buttonRemoveModel)

        self.vLayoutBoxModelSelect.addLayout(self.hLayoutAddModel)
        self.vLayoutBoxModelSelect.addLayout(self.hLayoutRemoveModel)
        self.vLayoutBoxModelSelect.addLayout(self.hLayoutPrefix)


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
        self.buttonResetMask = QPushButton('Reset mask (Shift + r)')
        self.buttonResetMask.clicked.connect(lambda: self.reset_mask())

        self.hLayoutMaskAction.addWidget(self.buttonMask)
        self.hLayoutMaskAction.addWidget(self.buttonUnmask)
        self.hLayoutMaskAction.addWidget(self.buttonResetMask)

        self.vLayoutBoxMaskSelect.addLayout(self.hLayoutMaskAction)
        self.vLayoutBoxMaskSelect.addLayout(self.hLayoutMaskPreset)

        # Add global parameter
        self.boxGlobalParam = QGroupBox('Global parameters')
        self.vLayoutGlobalParam = QVBoxLayout(self.boxGlobalParam)
        self.hLayoutAddGlobalParam = QHBoxLayout()
        self.hLayoutDelGlobalParam = QHBoxLayout()
        self.leGlobalParamName = QLineEdit('global_param')
        self.leGlobalParamName.setMaxLength(20)
        self.buttonAddGlobalParam = QPushButton('Add global parameter')
        self.buttonAddGlobalParam.clicked.connect(self.add_global_param)

        self.boxSelectGlobalParam = QComboBox()
        self.buttonDelGlobalParam = QPushButton('Remove global parameter')
        self.buttonDelGlobalParam.clicked.connect(self.remove_global_param)

        self.hLayoutAddGlobalParam.addWidget(self.leGlobalParamName)
        self.hLayoutAddGlobalParam.addWidget(self.buttonAddGlobalParam)
        self.hLayoutDelGlobalParam.addWidget(self.boxSelectGlobalParam)
        self.hLayoutDelGlobalParam.addWidget(self.buttonDelGlobalParam)

        self.vLayoutGlobalParam.addLayout(self.hLayoutAddGlobalParam)
        self.vLayoutGlobalParam.addLayout(self.hLayoutDelGlobalParam)

        # Fit model button
        self.boxFitAction = QGroupBox('Fitting')
        self.vLayoutBoxFitAction = QVBoxLayout(self.boxFitAction)
        self.buttonFit = QPushButton('Fit SpecModel')
        self.buttonFit.clicked.connect(lambda: self.fit())
        self.buttonSaveResult = QPushButton('Save fit result')
        self.buttonSaveResult.clicked.connect(lambda: self.save_fit_result())
        self.checkboxUseWeights = QCheckBox('Use weights (fluxden errors)')
        self.checkboxUseWeights.setChecked(self.specmodel.use_weights)
        self.checkboxUseWeights.stateChanged.connect(
            lambda: self.update_specmodel(specfitgui))
        self.checkboxFitReport = QCheckBox('Show fit report')
        self.checkboxFitReport.setChecked(self.showFitReport)
        self.checkboxFitReport.stateChanged.connect(lambda:
                                                    self.update_specmodel(specfitgui))

        self.hLayoutFit = QHBoxLayout()
        self.hLayoutFit.addWidget(self.buttonFit)
        self.hLayoutFit.addWidget(self.buttonSaveResult)

        self.hLayoutFitCheckboxes = QHBoxLayout()
        self.hLayoutFitCheckboxes.addWidget(self.checkboxUseWeights)
        self.hLayoutFitCheckboxes.addWidget(self.checkboxFitReport)


        self.vLayoutBoxFitAction.addLayout(self.hLayoutFit)
        self.vLayoutBoxFitAction.addLayout(self.hLayoutFitCheckboxes)


        # Vertical Box Layout for Specmodel Actions/Properties
        self.vLayoutProperties = QVBoxLayout()
        self.vLayoutProperties.addLayout(self.hLayoutName)
        self.vLayoutProperties.addWidget(self.boxRegionSelect)
        self.vLayoutProperties.addWidget(self.boxMaskSelect)
        self.vLayoutProperties.addWidget(self.boxModelSelect)
        self.vLayoutProperties.addWidget(self.boxGlobalParam)
        self.vLayoutProperties.addWidget(self.boxFitAction)

        # Horizontal Main Widget Layout
        self.hLayout.addLayout(self.vLayoutProperties)
        self.hLayout.addLayout(self.vLayoutCanvas)
        self.hLayout.setStretch(1, 2)

        self.modelTabWidget = QTabWidget()
        self.build_global_params_tab()

        # Vertical Main Widget Layout
        self.vLayout.addLayout(self.hLayout)
        self.vLayout.addWidget(self.modelTabWidget)
        self.vLayout.setStretch(0, 2)


    #     TODO Check if these lists are necessary
        self.params_widgetlist = []
        self.params_lineditlist = []
        self.params_varybox_list = []
        # self.params_hbox_list = []

        self.global_params_lineditlist = []
        self.global_params_varybox_list = []

        # Set the ClickFocus active on the SpecFitCanvas
        self.specModelCanvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.specModelCanvas.setFocus()

        # Initialize the Key Press Event for the SpecFitCanvas
        self.gcid = self.specModelCanvas.mpl_connect('key_press_event',
                                                   self.on_press)


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

            self.set_plot_flux()

        # Reset the region values
        elif event.key == 'r':

            self.reset_plot_region()

        # Full reset (region + mask)
        elif event.key == 'R':
            self.reset_mask()



    def set_plot_dispersion(self):

        self.update_region_from_ui()
        self.specmodel.xlim = [self.x_pos_a, self.x_pos_b]
        self.update_specmodel_plot()

    def set_plot_flux(self):

        self.update_region_from_ui()
        self.specmodel.ylim = [self.y_pos_a, self.y_pos_b]
        self.update_specmodel_plot()

    def reset_plot_region(self):

        self.reset_region()
        self.update_specmodel_plot()

    def reset_region(self):

        if hasattr(self.specmodel, 'spec'):
            self.x_pos_a = min(self.specmodel.spec.dispersion)
            self.x_pos_b = max(self.specmodel.spec.dispersion)
            self.y_pos_a = min(self.specmodel.spec.fluxden)
            self.y_pos_b = max(self.specmodel.spec.fluxden)

        else:
            self.x_pos_a = 0
            self.x_pos_b = 1
            self.y_pos_a = 0
            self.y_pos_b = 1

        # Excluded, so that previous region can be selected very quickly
        # self.leXposA.setText('{:.2f}'.format(self.x_pos_a))
        # self.leXposB.setText('{:.2f}'.format(self.x_pos_b))
        # self.leYposA.setText('{:.2E}'.format(self.y_pos_a))
        # self.leYposB.setText('{:.2E}'.format(self.y_pos_b))

        self.specmodel.xlim = [self.x_pos_a, self.x_pos_b]
        self.specmodel.ylim = [self.y_pos_a, self.y_pos_b]


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


        self.specModelCanvas.setFocus()


    def add_model_to_specmodel(self, specmodel, specfitgui):

        model_name = self.boxAddModel.currentText()
        prefix = self.lePrefix.text()+'_'

        # Check if chosen prefix already exists. If so, change the prefix_flag
        prefix_flag = True
        for model in specmodel.model_list:
            if prefix == model.prefix:
                specfitgui.statusBar().showMessage("Model with same "
                                               "prefix"
                                             " exists already! Please choose a"
                                             " different prefix.", 5000)
                prefix_flag = False

        # If the pre_fix already exists abort, otherwise add model to
        # continuum model list including parameters.
        if prefix_flag:

            for idx, model_func in enumerate(model_func_list):
                if model_name == model_func:

                    if specfitgui.specfit.redshift is not None:
                        model, params = model_setup_list[idx](prefix,
                                          redshift=specfitgui.specfit.redshift)

                        # Add global params to params
                        if self.specmodel.global_params:
                            params.update(self.specmodel.global_params)

                        specmodel.add_model(model, params)
                    else:
                        model, params = model_setup_list[idx](prefix)

                        # Add global params to params
                        if self.specmodel.global_params:
                            params.update(self.specmodel.global_params)

                        specmodel.add_model(model, params)

        # Rebuild all model tabs
        self.rebuild_model_tabs()

    def update_specmodel(self, specfitgui):
        """
        Only update the specmodel from the internal tab. External updating is
        done with the specfitgui
        :param specfitgui:
        :return:
        """

        # Update SpecModel name
        specmodel_name = self.leName.text()
        self.specmodel.name = specmodel_name

        # Update use_weights flags for fitting
        self.specmodel.use_weights = self.checkboxUseWeights.isChecked()

        self.showFitReport = self.checkboxFitReport.isChecked()

        current_tab_index = specfitgui.tabWidget.currentIndex()
        specfitgui.tabWidget.setTabText(current_tab_index, specmodel_name)

        self.specModelCanvas.setFocus()

    def update_specmodel_plot(self):

        self.specModelCanvas.plot(self.specmodel)


    def remove_model_from_specmodel(self):

        # Get index
        idx = self.boxRemoveModel.currentIndex()

        self.specmodel.delete_model(idx)

        self.rebuild_model_tabs()



    def rebuild_global_params_tab(self):

        idx = 0

        self.modelTabWidget.setCurrentIndex(idx)
        widget = self.modelTabWidget.currentWidget()
        layout = widget.layout()

        self.deleteItemsOfLayout(layout)

        self.modelTabWidget.removeTab(idx)

        self.build_global_params_tab()



    def rebuild_model_tabs(self):

        # Reset widget, linedit, varybox losts
        self.params_widgetlist = []
        self.params_lineditlist = []
        self.params_varybox_list = []

        # for widget in tab delete widget
        for idx in reversed(range(self.modelTabWidget.count()-1)):
            self.modelTabWidget.setCurrentIndex(idx+1)
            widget = self.modelTabWidget.currentWidget()
            layout = widget.layout()

            self.deleteItemsOfLayout(layout)

            self.modelTabWidget.removeTab(idx+1)
            self.boxRemoveModel.removeItem(idx)

        for idx, model in enumerate(self.specmodel.model_list):

            self.build_model_tab(idx)


    def build_global_params_tab(self):

        globalParamWidget = QWidget()
        hLayoutModel = QHBoxLayout(globalParamWidget)

        globalParams = self.specmodel.global_params

        lineditlist = []
        varyboxlist = []
        widgetlist = []

        for jdx, param in enumerate(globalParams):

            widgetlist = []

            groupBoxParam = QGroupBox(param)
            vLayoutGroupBoxParam = QVBoxLayout(groupBoxParam)

            label = QLabel("value")
            linedit = QLineEdit('{:.4E}'.format(globalParams[param].value))
            linedit.setMaxLength(20)
            expr_linedit = QLineEdit('{}'.format(globalParams[param].expr))
            expr_linedit.setMaxLength(20)
            min_label = QLabel("min")
            min_linedit = QLineEdit('{:.4E}'.format(globalParams[param].min))
            min_linedit.setMaxLength(20)
            max_label = QLabel("max")
            max_linedit = QLineEdit('{:.4E}'.format(globalParams[param].max))
            max_linedit.setMaxLength(20)
            vary_checkbox = QCheckBox("vary")
            vary_checkbox.setChecked(globalParams[param].vary)

            widgetlist.extend(
                [label, linedit, expr_linedit, min_label, min_linedit,
                 max_label, max_linedit, vary_checkbox])
            lineditlist.extend(
                [linedit, expr_linedit, min_linedit, max_linedit])
            varyboxlist.append(vary_checkbox)

            for w in widgetlist:
                vLayoutGroupBoxParam.addWidget(w)

            hLayoutModel.addWidget(groupBoxParam)

            vary_checkbox.stateChanged.connect(self.update_model_params_from_ui)

        # TODO NEXT!!! INPUT/FIT/UPDATE
        if widgetlist:
            self.global_params_widgetlist = widgetlist
            self.global_params_lineditlist = lineditlist
            self.global_params_varybox_list = varyboxlist

            # Activate input for lineEdit lists
            for l in lineditlist:
                l.returnPressed.connect(self.update_global_params_from_ui)

        # Add model parameters in scroll area
        ScArea = QScrollArea()
        ScArea.setLayout(QHBoxLayout())
        ScArea.setWidgetResizable(True)
        ScArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        ScArea.setWidget(globalParamWidget)

        # Add tab
        self.modelTabWidget.insertTab(0, ScArea, 'Global Params')





    def build_model_tab(self, index):

        model = self.specmodel.model_list[index]
        params = self.specmodel.params_list[index]
        prefix = model.prefix

        self.boxRemoveModel.addItem(prefix)

        # Initialize tab widget
        modelWidget = QWidget()

        hLayoutModel = QHBoxLayout(modelWidget)

        lineditlist = []
        varyboxlist = []

        for jdx, param in enumerate(params):

            widgetlist = []

            groupBoxParam = QGroupBox(param)
            vLayoutGroupBoxParam = QVBoxLayout(groupBoxParam)

            label = QLabel("value")
            linedit = QLineEdit('{:.4E}'.format(params[param].value))
            linedit.setMaxLength(20)
            expr_linedit = QLineEdit('{}'.format(params[param].expr))
            expr_linedit.setMaxLength(20)
            min_label = QLabel("min")
            min_linedit = QLineEdit('{:.4E}'.format(params[param].min))
            min_linedit.setMaxLength(20)
            max_label = QLabel("max")
            max_linedit = QLineEdit('{:.4E}'.format(params[param].max))
            max_linedit.setMaxLength(20)
            vary_checkbox = QCheckBox("vary")
            vary_checkbox.setChecked(params[param].vary)

            widgetlist.extend(
                [label, linedit, expr_linedit, min_label, min_linedit,
                 max_label, max_linedit, vary_checkbox])
            lineditlist.extend(
                [linedit, expr_linedit, min_linedit, max_linedit])
            varyboxlist.append(vary_checkbox)

            for w in widgetlist:
                vLayoutGroupBoxParam.addWidget(w)

            hLayoutModel.addWidget(groupBoxParam)

            vary_checkbox.stateChanged.connect(self.update_model_params_from_ui)


        self.params_widgetlist.extend(widgetlist)
        self.params_lineditlist.append(lineditlist)
        self.params_varybox_list.append(varyboxlist)

        # Activate input for lineEdit lists
        for l in lineditlist:
            l.returnPressed.connect(self.update_model_params_from_ui)


        # Add model parameters in scroll area
        ScArea = QScrollArea()
        ScArea.setLayout(QHBoxLayout())
        ScArea.setWidgetResizable(True)
        ScArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        ScArea.setWidget(modelWidget)


        # Add tab
        self.modelTabWidget.addTab(ScArea, 'Model {}'.format(prefix))


    def update_mask(self, mode='mask'):

        self.x_pos_a = float(self.leXposA.text())
        self.x_pos_b = float(self.leXposB.text())

        self.mask(mode)

        self.update_specmodel_plot()


    def mask(self, mode='mask'):

        if hasattr(self.specmodel, 'spec'):

            mask_between = np.sort(np.array([self.x_pos_a,
                                             self.x_pos_b]))

            spec = self.specmodel.spec
            lo_index = np.argmin(np.abs(spec.dispersion - mask_between[0]))
            up_index = np.argmin(np.abs(spec.dispersion - mask_between[1]))
            if mode == 'mask':
                self.specmodel.mask[lo_index:up_index] = True
            elif mode == 'unmask':
                self.specmodel.mask[lo_index:up_index] = False


    def reset_mask(self):

        self.reset_region()
        self.specmodel.reset_fit_mask()
        self.update_specmodel_plot()


    def load_mask_preset(self):

        mask_preset_key = self.boxMaskPreset.currentText()

        self.specmodel.add_mask_preset_to_fit_mask(mask_preset_key)

        self.update_specmodel_plot()

    def update_boxSelectGlobalParam(self):

        for param in self.specmodel.global_params:
            if param not in self.specfitgui.specfit.super_params:
                if self.boxSelectGlobalParam.findText(param) < 0:
                    # Add item to QComboBox
                    self.boxSelectGlobalParam.addItem(param)

    def add_global_param(self):

        param_name = self.leGlobalParamName.text()
        # Check if item already in QComboBox
        if self.boxSelectGlobalParam.findText(param_name) < 0:
            # Add item to QComboBox
            self.boxSelectGlobalParam.addItem(param_name)
            # Add param to SpecModel
            self.specmodel.add_global_param(param_name)

        self.rebuild_global_params_tab()
        self.rebuild_model_tabs()


    def remove_global_param(self):

        param_name = self.boxSelectGlobalParam.currentText()
        # Remove global param from SpecModel
        self.specmodel.remove_global_param(param_name)

        # Remove item from QComboBox
        idx = self.boxSelectGlobalParam.findText(param_name)
        self.boxSelectGlobalParam.removeItem(idx)

        self.rebuild_global_params_tab()
        self.rebuild_model_tabs()



    def update_ui_from_specmodel_params(self):

        # Update the UI parameter values for all models
        for idx, model in enumerate(self.specmodel.model_list):

            params = self.specmodel.params_list[idx]

            for jdx, param in enumerate(params):
                temp_val = self.specmodel.params_list[idx][param].value
                self.params_lineditlist[idx][jdx * 4 + 0].setText("{0:4E}".format(
                    temp_val))
                temp_val = self.specmodel.params_list[idx][param].expr
                self.params_lineditlist[idx][jdx * 4 + 1].setText(
                    "{}".format(temp_val))

        # Update the UI global parameters
        for jdx, param in enumerate(self.specmodel.global_params):

            temp_val = self.specmodel.global_params[param].value
            self.global_params_lineditlist[jdx * 4 + 0].setText("{"
                                                                 "0:4E}".format(
                temp_val))
            temp_val = self.specmodel.global_params[param].expr
            self.global_params_lineditlist[jdx * 4 + 1].setText(
                "{}".format(temp_val))


    def update_global_params_from_ui(self):

        for jdx, param in enumerate(self.specmodel.global_params):

            # try:

            new_value = float(
                self.global_params_lineditlist[jdx * 4 + 0].text())
            new_expr = self.global_params_lineditlist[jdx * 4 + 1].text()
            new_min = float(
                self.global_params_lineditlist[jdx * 4 + 2].text())
            new_max = float(
                self.global_params_lineditlist[jdx * 4 + 3].text())
            new_vary = self.global_params_varybox_list[jdx].isChecked()

            if new_expr == 'None':
                new_expr = None

            # Set the new parameter values
            self.specmodel.global_params[param].set(value=new_value,
                                                       expr=new_expr,
                                                       min=new_min,
                                                       max=new_max,
                                                       vary=new_vary)

        self.specmodel.update_model_params_for_global_params()
        self.rebuild_global_params_tab()
        self.rebuild_model_tabs()
        self.update_specmodel_plot()

        self.specModelCanvas.setFocus()



    def update_model_params_from_ui(self):

        for idx, parset in enumerate(self.specmodel.params_list):

            params = self.specmodel.params_list[idx]

            for jdx, param in enumerate(params):
                try:
                    new_value = float(
                        self.params_lineditlist[idx][jdx * 4 + 0].text())
                    new_expr = self.params_lineditlist[idx][
                        jdx * 4 + 1].text()
                    new_min = float(
                        self.params_lineditlist[idx][jdx * 4 + 2].text())
                    new_max = float(
                        self.params_lineditlist[idx][jdx * 4 + 3].text())
                    new_vary = self.params_varybox_list[idx][jdx].isChecked()

                    # Validate the new expression input
                    old_expr = self.specmodel.params_list[idx][param].expr
                    expr_list = re.split('\*|\)|\(|/|\+|\-|\s',new_expr)
                    for item in expr_list:
                        if not item or not item.isdigit():
                            if item not in params:
                                new_expr = old_expr

                    # Set the new parameter values
                    self.specmodel.params_list[idx][param].set(value=new_value,
                                                             expr=new_expr,
                                                             min=new_min,
                                                             max=new_max,
                                                             vary=new_vary)
                except:
                    print(
                        "[ERROR] Input does not conform to string or float "
                        "limitations!")

        self.specmodel.build_model()

        self.update_specmodel_plot()

        self.specModelCanvas.setFocus()


    def fit(self):

        self.specmodel.fit()

        self.specfitgui.rebuild_super_params_widget()
        self.rebuild_global_params_tab()
        self.update_ui_from_specmodel_params()
        self.update_specmodel_plot()

        if self.showFitReport:
            msg = QMessageBox()
            msg.setStyleSheet("QLabel{min-width: 700px;}")
            msg.setWindowTitle("SpecModel {} fit report".format(
                self.specmodel.name))
            # msg.setText('Fit report for SpecModel {}'.format(
            #     self.specmodel.name))
            msg.setText(fit_report(self.specmodel.fit_result))
            x = msg.exec_()


            if self.specmodel.specfit.fitting_method == 'Maximum likelihood ' \
                                                       'via Monte-Carlo Markov Chain':
                corner_plot = corner.corner(self.specmodel.fit_result.flatchain,
                                            labels=self.specmodel.fit_result.var_names,
                                            truths=list(
                                                self.specmodel.fit_result.params.valuesdict().values()))
                corner_plot.show()


    def save_fit_result(self):

        # Select save folder
        foldername = str(
            QFileDialog.getExistingDirectory(self, "Select Directory"))

        if foldername:
            self.specmodel.save_fit_report(foldername)
            self.specmodel.save_mcmc_chain(foldername)
            self.specModelCanvas.fig.savefig(foldername +
                             '/SpecModel_{}.png'.format(self.specmodel.name))
        else:
            self.statusBar().showMessage('[WARNING] No save directory '
                                         'selected.')



    def deleteItemsOfLayout(self, layout):
         if layout is not None:
             while layout.count():
                 item = layout.takeAt(0)
                 widget = item.widget()
                 if widget is not None:
                     widget.setParent(None)
                 else:
                     self.deleteItemsOfLayout(item.layout())
