

from PyQt5.QtWidgets import QWidget, QPushButton, \
    QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, \
    QCheckBox


class ResampleWindow(QWidget):
    def __init__(self, specfitgui):
        super().__init__()

        self.specfitgui = specfitgui
        self.setWindowTitle("Resample and Fit")

        mainLayout = QVBoxLayout()

        self.labelSamples = QLabel('# of samples')
        self.leSamples = QLineEdit('{}'.format(
            self.specfitgui.nsamples))
        self.leSamples.returnPressed.connect(self.update_specfitgui)

        hLayoutSamples = QHBoxLayout()
        hLayoutSamples.addWidget(self.labelSamples)
        hLayoutSamples.addWidget(self.leSamples)

        self.labelSeed = QLabel('Seed')
        self.leSeed = QLineEdit('{}'.format(
            self.specfitgui.resample_seed))
        self.leSeed.returnPressed.connect(self.update_specfitgui)

        hLayoutSeed = QHBoxLayout()
        hLayoutSeed.addWidget(self.labelSeed)
        hLayoutSeed.addWidget(self.leSeed)

        self.labelFoldername = QLabel('Output folder name')
        self.leFoldername = QLineEdit('{}'.format(
            self.specfitgui.resample_foldername))
        self.leFoldername.returnPressed.connect(self.update_specfitgui)

        hLayoutFoldername = QHBoxLayout()
        hLayoutFoldername.addWidget(self.labelFoldername)
        hLayoutFoldername.addWidget(self.leFoldername)

        self.checkBoxSaveResultPlots = QCheckBox('Save result plots')
        self.checkBoxSaveResultPlots.setChecked(
            self.specfitgui.save_result_plots)
        self.checkBoxSaveResultPlots.stateChanged.connect(
            self.update_specfitgui)

        self.hLayoutButtons = QHBoxLayout()
        self.resampleButton = QPushButton('Resample')
        self.resampleButton.clicked.connect(self.resample)
        self.closeButton = QPushButton('Close')
        self.closeButton.clicked.connect(self.close)

        self.hLayoutButtons.addWidget(self.resampleButton)
        self.hLayoutButtons.addWidget(self.closeButton)

        mainLayout.addLayout(hLayoutSamples)
        mainLayout.addLayout(hLayoutSeed)
        mainLayout.addWidget(self.checkBoxSaveResultPlots)
        mainLayout.addLayout(hLayoutFoldername)
        mainLayout.addLayout(self.hLayoutButtons)

        self.setLayout(mainLayout)


    def resample(self):
        self.update_specfitgui()
        self.specfitgui.resample_and_fit()


    def update_specfitgui(self):

        try:
            self.specfitgui.nsamples = int(self.leSamples.text())
        except:
            self.leSamples.setText('{}'.format(self.specfitgui.nsamples))
            self.specfitgui.statusBar().showMessage('[ERROR] Input needs to be '
                                                  'an integer'
                                         'value.')

        try:
            self.specfitgui.resample_seed = int(self.leSeed.text())
        except:
            self.leSeed.setText('{}'.format(self.specfitgui.resample_seed))
            self.specfitgui.statusBar().showMessage('[ERROR] Input needs to be '
                                                  'an integer'
                                         'value.')

        self.specfitgui.save_result_plots = self.checkBoxSaveResultPlots.isChecked()

        self.specfitgui.resample_foldername = self.leFoldername.text()



class EmceeWindow(QWidget):
    def __init__(self, specfitgui):
        super().__init__()

        self.specfitgui = specfitgui
        self.setWindowTitle("MCMC parameters (emcee)")

        mainLayout = QVBoxLayout()

        self.labelSteps = QLabel('Steps')
        self.leSteps = QLineEdit('{}'.format(
            self.specfitgui.specfit.emcee_kws['steps']))
        self.leSteps.returnPressed.connect(self.update_emcee_kws)

        hLayoutSteps = QHBoxLayout()
        hLayoutSteps.addWidget(self.labelSteps)
        hLayoutSteps.addWidget(self.leSteps)


        self.labelWalkers = QLabel('Walkers')
        self.leWalkers = QLineEdit('{}'.format(
            self.specfitgui.specfit.emcee_kws['nwalkers']))
        self.leWalkers.returnPressed.connect(self.update_emcee_kws)

        hLayoutWalkers = QHBoxLayout()
        hLayoutWalkers.addWidget(self.labelWalkers)
        hLayoutWalkers.addWidget(self.leWalkers)


        self.labelBurn = QLabel('Burn')
        self.leBurn = QLineEdit('{}'.format(
            self.specfitgui.specfit.emcee_kws['burn']))
        self.leBurn.returnPressed.connect(self.update_emcee_kws)

        hLayoutBurn = QHBoxLayout()
        hLayoutBurn.addWidget(self.labelBurn)
        hLayoutBurn.addWidget(self.leBurn)

        self.labelThin = QLabel('Thin')
        self.leThin = QLineEdit('{}'.format(
            self.specfitgui.specfit.emcee_kws['thin']))
        self.leThin.returnPressed.connect(self.update_emcee_kws)

        hLayoutThin = QHBoxLayout()
        hLayoutThin.addWidget(self.labelThin)
        hLayoutThin.addWidget(self.leThin)

        self.labelWorkers = QLabel('Workers')
        self.leWorkers = QLineEdit('{}'.format(
            self.specfitgui.specfit.emcee_kws['workers']))
        self.leWorkers.returnPressed.connect(self.update_emcee_kws)

        hLayoutWorkers = QHBoxLayout()
        hLayoutWorkers.addWidget(self.labelWorkers)
        hLayoutWorkers.addWidget(self.leWorkers)

        self.hLayoutButtons = QHBoxLayout()
        self.applyButton = QPushButton('Apply')
        self.applyButton.clicked.connect(self.apply)
        self.closeButton = QPushButton('Close')
        self.closeButton.clicked.connect(self.close)

        self.hLayoutButtons.addWidget(self.applyButton)
        self.hLayoutButtons.addWidget(self.closeButton)


        mainLayout.addLayout(hLayoutSteps)
        mainLayout.addLayout(hLayoutWalkers)
        mainLayout.addLayout(hLayoutBurn)
        mainLayout.addLayout(hLayoutThin)
        mainLayout.addLayout(hLayoutWorkers)
        mainLayout.addLayout(self.hLayoutButtons)

        self.setLayout(mainLayout)


    def update_emcee_kws(self):

        try:
            self.specfitgui.specfit.emcee_kws['steps'] = int(
                self.leSteps.text())
        except:
            self.leSteps.setText('{}'.format(
                self.specfitgui.specfit.emcee_kws['steps']))
            self.specfitgui.statusBar().showMessage(
                '[ERROR] Input needs to be an integer value.')

        try:
            self.specfitgui.specfit.emcee_kws['nwalkers'] = int(
                self.leWalkers.text())
        except:
            self.leWalkers.setText('{}'.format(
                self.specfitgui.specfit.emcee_kws['nwalkers']))
            self.specfitgui.statusBar().showMessage(
                '[ERROR] Input needs to be an integer value.')

        try:
            self.specfitgui.specfit.emcee_kws['burn'] = int(
                self.leBurn.text())
        except:
            self.leBurn.setText('{}'.format(
                self.specfitgui.specfit.emcee_kws['burn']))
            self.specfitgui.statusBar().showMessage(
                '[ERROR] Input needs to be an integer value.')

        try:
            self.specfitgui.specfit.emcee_kws['thin'] = int(
                self.leThin.text())
        except:
            self.leThin.setText('{}'.format(
                self.specfitgui.specfit.emcee_kws['thin']))
            self.specfitgui.statusBar().showMessage(
                '[ERROR] Input needs to be an integer value.')

    def apply(self):
        self.update_emcee_kws()
        self.close()



class NormalizeWindow(QWidget):
    def __init__(self, specfitgui):
        super().__init__()

        mainLayout = QVBoxLayout()

        self.specfitgui = specfitgui
        self.setWindowTitle("Normalize spectrum")

        self.labelFactor = QLabel('Normalization factor')
        self.leFactor = QLineEdit('{}'.format('1e-17'))

        hLayoutFactor = QHBoxLayout()
        hLayoutFactor.addWidget(self.labelFactor)
        hLayoutFactor.addWidget(self.leFactor)

        self.normalizeToButton = QPushButton('Normalize to factor')
        self.normalizeToButton.clicked.connect(self.normalize_to)
        self.normalizeByButton = QPushButton('Normalize by factor')
        self.normalizeByButton.clicked.connect(self.normalize_by)
        self.closeButton = QPushButton('Close')
        self.closeButton.clicked.connect(self.close)

        hLayoutButtons = QHBoxLayout()
        hLayoutButtons.addWidget(self.normalizeToButton)
        hLayoutButtons.addWidget(self.normalizeByButton)
        hLayoutButtons.addWidget(self.closeButton)

        mainLayout.addLayout(hLayoutFactor)
        mainLayout.addLayout(hLayoutButtons)

        self.setLayout(mainLayout)

    def normalize_to(self):
        factor = float(self.leFactor.text())
        self.specfitgui.normalize_spectrum_to_factor(factor)
        self.close()

    def normalize_by(self):
        factor = float(self.leFactor.text())
        self.specfitgui.normalize_spectrum_by_factor(factor)
        self.close()
