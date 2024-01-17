import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from chart_studio.plotly import plot, iplot
from plotly.offline import iplot
from yellowbrick.features import Manifold
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
#from bubbly.bubbly import bubbleplot
import warnings
from bubbly.bubbly import bubbleplot
import scipy.stats as stats
from IPython.core.pylabtools import figsize   # ***
import numpy as np
import matplotlib.pyplot as plt  # ***

#  heatmapLT(df):
#  lower_triangle_mask(square_array):
#  plot_band(df, time_column, value_column):
#  view_cluster_3D(df):
#  view_on_globe(df, col_name_locations, col_name_values='cluster')
#  visualize_dataframe_columns(df, color='r', bw=0.5):
#  visualize_dataframe_columns_embedding(df, manifold='tsne', labels=None):
#  visualize_dataframe_columns_kde(df):

def demo_plot():
    ''' creates a nice looking generic 1-d plot with labels '''
    import matplotlib.pyplot as plt
    %matplotlib inline

    bp_x = np.linspace(0, 2*np.pi, num=40, endpoint=True)
    bp_y = np.sin(bp_x)

    # Make the plot
    plt.plot(bp_x, bp_y, linewidth=3, linestyle="--",
             color="blue", label=r"Legend label $\sin(x)$")
    plt.xlabel(r"Description of $x$ coordinate (units)")
    plt.ylabel(r"Description of $y$ coordinate (units)")
    plt.title(r"Title here (remove for papers)")
    plt.xlim(0, 2*np.pi)
    plt.ylim(-1.1, 1.1)
    plt.legend(loc="lower left")
    plt.show()

def lower_triangle_mask(square_array):

	return np.triu(np.ones_like(square_array, dtype=bool))


def heatmapLT(df):
	''' plot correlation heatmap showing lower triangle only '''

	correlations = df.corr()
	mask = lower_triangle_mask(correlations)

	# Create a custom diverging palette
	cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
	plt.figure(figsize=(16, 12))
	sns.heatmap(correlations, mask=mask, center=0, annot=True, fmt='.2f', square=True, cmap=cmap)

	plt.show();


def visualize_dataframe_columns(df, color='r', bw=0.5):
    ''' Plot histogram and estimated density function for each numerical column of df '''

    columns = df.describe().columns.to_list()
    N = len(columns)
    plt.figure(figsize = (20, 40))
    for i in range(N):
        plt.subplot(N, 4, i+1)
        sns.distplot(df[columns[i]], color = color, kde_kws = {'bw' :bw})
        plt.title(columns[i])

    plt.tight_layout()


def visualize_dataframe_columns_kde(df):
    ''' Plot kernel density estimate (kde) function for each numerical column of df '''

    columns = df.describe().columns.to_list()
    N = len(columns)
    plt.figure(figsize = (20, 40))
    for i in range(N):
        plt.subplot(N, 4, i+1)
        sns.kdeplot(df[columns[i]], bw=0.5)
        plt.title(columns[i])

    plt.tight_layout()

#################################################################################################
def visualize_dataframe_columns_embedding(df, manifold='tsne', labels=None):
    df_numeric = df[df.describe().columns.tolist()]
    vis = Manifold(manifold=manifold)
    embedding = vis.fit_transform(df_numeric,labels)
    vis.show()
    return embedding   


#################################################################################################
#######  TODO:  finish adapting to general purpose (i.e., not just cluster indices)

init_notebook_mode(connected = True)
warnings.filterwarnings('ignore')

def view_on_globe(df, col_name_locations, col_name_values='cluster')
	# Visualizing the clusters geographically
	data = dict(type = 'choropleth', 
	           locations = df[col_name_locations],
	           locationmode = 'country names',
	           colorscale='RdYlGn',
	           z = df[col_name_values], 
	           text = df[col_name_locations],
	           colorbar = {'title':'Clusters'})

	layout = dict(title = 'Geographical Visualization', 
	              geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))

	choromap3 = go.Figure(data = [data], layout=layout)
	iplot(choromap3)


#################################################################################################
#######  TODO:  create bubbleplot wrapper function


def view_cluster_3D(df):

	figure = bubbleplot(dataset=A, 
    x_column='C', y_column='V', bubble_column='Steel',  
    color_column='cluster', z_column='Rw', size_column='Fe',
    x_title="% Carbon", y_title="% Vanadium", z_title="Hardness (Rw)",
    title='Clusters based Impact of Steel Composition on Hardness',
    colorbar_title='Cluster', marker_opacity=1, colorscale='Portland',
    scale_bubble=0.8, height=650)

	iplot(figure, config={'scrollzoom': True})



#################################################################################################
#######  TODO:  adapt function from LSTM climate change notebook to generic numerica dataframe

def plot_band(df, time_column, value_column):
# Uncertainity upper bound 
trace1 = go.Scatter(
    x = df[time_column], 
    y = np.array(df[value_column]) + np.array(df[value_column]).std, # Adding uncertinity
    name = 'Uncertainty top',
    line = dict(color = 'green'))

# Uncertainity lower bound
trace2 = go.Scatter(
    x = df_global['year'] , 
    y = np.array(df_global['AverageTemperature']) - np.array(df_global['AverageTemperatureUncertainty']), # Subtracting uncertinity
    fill = 'tonexty',
    name = 'Uncertainty bottom',
    line = dict(color = 'green'))

# Recorded temperature
trace3 = go.Scatter(
    x = df_global['year'] , 
    y = df_global['AverageTemperature'],
    name = 'Average Temperature',
    line = dict(color='red'))
data = [trace1, trace2, trace3]

layout = go.Layout(
    xaxis = dict(title = 'year'),
    yaxis = dict(title = 'Average Temperature, °C'),
    title = 'Average Land Temperatures Globally',
    showlegend = False)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


#################################################################################################
#######  TODO:  check out *** lines
# https://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter3_MCMC/Ch3_IntroMCMC_PyMC3.ipynb

%matplotlib inline
figsize(12.5, 4)  # ***


jet = plt.cm.jet
fig = plt.figure()
x = y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)

plt.subplot(121)
uni_x = stats.uniform.pdf(x, loc=0, scale=5)
uni_y = stats.uniform.pdf(y, loc=0, scale=5)
M = np.dot(uni_x[:, None], uni_y[None, :])
im = plt.imshow(M, interpolation='none', origin='lower',
                cmap=jet, vmax=1, vmin=-.15, extent=(0, 5, 0, 5))

plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title("Landscape formed by Uniform priors.")

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, M, cmap=plt.cm.jet, vmax=1, vmin=-.15)
ax.view_init(azim=390)
plt.title("Uniform prior landscape; alternate view");

