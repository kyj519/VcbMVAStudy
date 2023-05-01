import numpy as np
arr = np.load('mask.npy',allow_pickle=True)
arr =arr[()]
row_mean = np.array([arr[i] for i in range(5)])
row_mean = row_mean.transpose(1, 0, 2)
import tqdm

import plotly.graph_objs as go
import plotly.io as pio

# define the x-axis labels
varlist = ['bvsc_w_u','bvsc_w_d','cvsl_w_u','cvsl_w_d',
           'cvsb_w_u','cvsb_w_d','n_bjets','n_cjets',
           'pt_w_u','pt_w_d','eta_w_u','eta_w_d','best_mva_score']

# define the heatmap trace


# define the layout of the plot
layout = go.Layout(title='Colormap of a 5x13 Array',
                   xaxis={'tickmode': 'array', 'ticktext': varlist, 'tickvals': list(range(len(varlist))), 'tickangle': 45})
# iterate over all values of j and write the heat maps to the video
for j in tqdm.tqdm(range(6951412)):
    if j%1000 !=0:
        continue
    # define the heatmap trace
    heatmap_trace = go.Heatmap(z=row_mean[j])

    # define the layout of the plot
    layout = go.Layout(title='Colormap of a 5x13 Array',
                    xaxis={'tickmode': 'array', 'ticktext': varlist, 'tickvals': list(range(len(varlist)))})

    # create the figure object
    fig = go.Figure(data=[heatmap_trace], layout=layout)

    # display the plot

    pio.write_image(fig, f'test/heatmap_{j}.png', width=800, height=600)