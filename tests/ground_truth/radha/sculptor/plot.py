
import matplotlib


# Define colors
black = (0, 0, 0)
orange = (230/255., 159/255., 0)
blue = (86/255., 180/255., 233/255.)
green = (0, 158/255., 115/255.)
yellow = (240/255., 228/255., 66/255.)
dblue = (0, 114/255., 178/255.)
vermillion = (213/255., 94/255., 0)
purple = (181/255., 107/255., 148/255.)

# GeoDataViz Colors
gdv_div_a = ['#009392', '#39b185', '#9ccb86', '#e9e29c', '#eeb479', '#e88471', '#cf597e']
gdv_div_b = ['#045275', '#089099', '#7ccba2', '#fcde9c', '#f0746e', '#dc3977', '#7c1d6f']
gdv_div_d = ['#008042', '#6fa253', '#b7c370', '#fce498', '#d78287', '#bf5688', '#7c1d6f']


# Define default setups
def set_paper_defaults():
    # Defining the paper plotting style

    matplotlib.rc('text', usetex=True)

    # matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    # matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amssymb}"]
    matplotlib.rcParams['xtick.major.size'] = 4.5
    matplotlib.rcParams['xtick.minor.size'] = 3
    matplotlib.rcParams['ytick.major.size'] = 4.5
    matplotlib.rcParams['ytick.minor.size'] = 3
    matplotlib.rcParams['axes.linewidth'] = 1.2
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['xtick.minor.visible'] = True
    matplotlib.rcParams['ytick.minor.visible'] = True
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['ytick.right'] = True



def set_presentation_defaults():
    # Defining the presentation plotting style

    matplotlib.rc('font', weight=500)
    matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Roboto']})
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.it'] = 'Roboto'
    matplotlib.rcParams['mathtext.rm'] = 'Roboto'

    # matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    # matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amssymb}"]
    matplotlib.rcParams['xtick.major.size'] = 4.5
    matplotlib.rcParams['xtick.minor.size'] = 3
    matplotlib.rcParams['ytick.major.size'] = 4.5
    matplotlib.rcParams['ytick.minor.size'] = 3
    matplotlib.rcParams['axes.linewidth'] = 1.2
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['xtick.minor.visible'] = True
    matplotlib.rcParams['ytick.minor.visible'] = True
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['ytick.right'] = True