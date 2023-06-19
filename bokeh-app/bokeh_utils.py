#!/usr/bin/python

#Load libs
from bokeh.core.validation import silence
from bokeh.io import output_notebook, curdoc
from bokeh.core.validation.warnings import MISSING_RENDERERS
from bokeh.events import PointEvent, Tap
from bokeh.layouts import row, layout, column
from bokeh.models import (Band, CategoricalColorMapper, ColorBar, ColumnDataSource,
                          CrosshairTool, DataRange1d, HoverTool, InlineStyleSheet, 
                          Legend, LegendItem, LinearColorMapper, NumeralTickFormatter, 
                          PanTool, PolyAnnotation, Range1d, ResetTool, Select, Slider,
                          Spacer, WheelZoomTool)
from bokeh.palettes import all_palettes, interp_palette
from bokeh.plotting import figure
import nibabel as nib
import numpy as np
import os
import pandas as pd
import scipy.interpolate as interp
import scipy.stats as stats
import statsmodels.formula.api as smf

#Ignore warning that says we have some missing renders...s
silence(MISSING_RENDERERS, True)

#Information about orienation/axes
views = ['sag', 'tra', 'cor']
view_info = {'sag':[['y', 'z'], 'x', 0],
             'tra':[['x', 'y'], 'z', 2],
             'cor':[['x', 'z'], 'y', 1]}
             
def df_to_2d_list(df, pivot_var, extract_var, mask):
       return [df[extract_var][mask][id == df[pivot_var][mask]].tolist() for id in df[pivot_var][mask].unique()]

def overlay_plot(under_source, over_sources, under_data, over_data, 
                 over_idx, ori, cmaps, coord, cross, dims, title=None):
            
    #Use orientation information to setup dimensions
    x = view_info[ori][0][0]
    y = view_info[ori][0][1]
    dw = dims[x]
    dh = dims[y]
    h_x = f"anat_{x}"
    v_y = f"anat_{y}"
    
    #Ploting options based on orientation
    if ori == 'tra':
        x_range = Range1d(dw, 0, bounds=(0, dw))
        y_range = Range1d(0, dh, bounds=(0, dh))
        title_color = 'black'
    elif ori == 'sag':
        x_range = Range1d(0, dw, bounds=(0, dw))
        y_range = Range1d(0, dh, bounds=(0, dh))
        title_color = 'white'
    else:
        x_range = Range1d(dw, 0, bounds=(0, dw))
        y_range = Range1d(0, dh, bounds=(0, dh))
        title_color = 'white'
    
    #Create plot
    p = figure(x_range=x_range, y_range=y_range, name='image')
    p.image(ori, source=under_source, x=0, y=0, dw=dw, dh=dh,
            level="image", color_mapper=cmaps[0])
    p.image(ori, source=over_sources[over_idx], x=0, y=0, dw=dw, dh=dh,
            level="image", color_mapper=cmaps[1], alpha=0.8)
    p.line(x=h_x, y=view_info[ori][0][1], line_color="black", line_width=6,
           line_alpha=0.6, line_dash='dashed', source=cross)
    p.line(x=view_info[ori][0][0], y=v_y, line_color="black", line_width=6,
           line_alpha=0.6, line_dash='dashed', source=cross)
    
    #Function for handling mouse events
    mouse_func = mouse_wrapper(ori, [under_source] + over_sources, [under_data] + over_data, coord, cross)
    p.on_event('tap', mouse_func)
    
    #Plot styling
    p.axis.visible = False
    p.grid.visible = False
    p.outline_line_color= None
    p.toolbar_location = None
    if title is not None:
        p.title = title
        p.title.text_font_style = "bold"
        p.title.text_font_size = "42px"
        p.title.align = 'center'
        p.title.text_color = title_color
    
    return p

def img_kde(img_1, img_2):
    
    #Make grid to estimate pdf on
    min_1 = img_1.min()
    max_1 = img_1.max()
    min_2 = img_2.min()
    max_2 = img_2.max()
    X, Y = np.mgrid[min_1:max_1:100j, min_2:max_2:100j]

    #Estimate kd
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([img_1, img_2])
    kernel = stats.gaussian_kde(values)
    den = kernel(positions).reshape(X.shape)

    #Interpolate image values onto pdf
    return interp.interpn([X[:, 0], Y[0, :]], den, np.stack((img_1, img_2)).T,
                          bounds_error=False, fill_value=0.0)

def create_colorbar(c_map, unit=None, orientation='vertical', loc='left'):

    #Create figure and coclorbar object
    p = figure(name='fig')
    c_b = ColorBar(color_mapper=c_map,
                   major_label_text_font_size='20px',
                   major_label_text_font_style='bold',
                   orientation=orientation,
                   name='cbar')

    #Add units label
    if unit is not None:
        c_b.title = unit
        c_b.title_text_font_size = '18px'
        c_b.title_text_font_style = 'bold'  

    #Display options
    p.add_layout(c_b, loc)
    p.outline_line_color= None
    p.toolbar_location = None

    return p

def mouse_wrapper(ori, sources, images, coords, cross):

    #Determine mask for saving coordinates
    dims = view_info[ori][0]

    #Function to actually update the coordinates
    def mouse_click(event: PointEvent):
        click_coords = [np.int32(event.x), np.int32(event.y)]
        for dim, coord in zip(dims, click_coords):
            coords.data[dim] = [coord]
            cross.data[dim] = [coord] * 2
        update_data(sources, images, coords)

    return mouse_click

#Updates data based on coordinates
def update_data(sources, images, coords):
    for source, image in zip(sources, images):
        for view in views:
            source.data[view] = [np.take(image, coords.data[view_info[view][1]], axis=view_info[view][2]).squeeze().T] 

#Update image data based on thresholding another image
def thresh_wrapper(over_data, over_source, over_ref, thresh_data, coords):
            
   def update_thresh(attrname, old, new):
      over_update = np.copy(over_ref)
      over_update[thresh_data < new] = np.nan
      over_data[:] = np.copy(over_update)
      for view in views:
         over_slice = np.take(over_data, coords.data[view_info[view][1]], axis=view_info[view][2]).squeeze().T
         thresh_slice = np.take(thresh_data, coords.data[view_info[view][1]], axis=view_info[view][2]).squeeze().T
         over_slice[thresh_slice < new] = np.nan
         over_source.data[view] = [over_slice] 

   return update_thresh

#Three rows, two of images and one with plot
def three_row(anat_path, over_paths, thresh_path, meas, unit,
              info_path=None, reg_path=None,
              over_range=[[5, 95]], over_mode=['percentile'],
              anat_range=[2.5, 97.5], anat_mode='percentile',
              over_titles=[None],
              roi_path='./data/winner_wmparc_comb_on_MNI152_2mm_masked.nii.gz',
              over_palettes=['Plasma'],
              over_thresh=[False],
              names_path='./data/wmparc_names.txt'):
              
   #Slider info
   slide_width = 400
   slide_css = InlineStyleSheet(css=".bk-slider-title { font-size: 30px; }")
   slide_wid = Slider(title="-log10(p)-threshold", value=1.3, start=0, end=5, step=0.1, 
                      width=slide_width, stylesheets=[slide_css], name='slide')


   def roi_wrapper(sources, images, info, coords, names, fig): 
     
      def update_roi(event: PointEvent):
      
         #Figure out index of roi
         roi_val = np.int32(images[0][np.int32(coords.data['x'][0]),
                                          np.int32(coords.data['y'][0]),
                                          np.int32(coords.data['z'][0])])
         roi_idx = roi_val - 1

         #Update name on scatter plot
         roi_name = names[roi_idx]
         fig.title.text = roi_name.replace(".", " ")

         #Update scatter plot points
         sources[0].data['roi'] = images[1][:, roi_idx]

         #Update scatter plot lines
         roi = [info[roi_name][subj == info['subj']].tolist() for subj in info['subj'].unique()]
         sources[1].data['roi'] = roi

      return update_roi
         
   #Load in anatomical image
   anat_hdr = nib.load(anat_path)
   anat_data = anat_hdr.get_fdata().squeeze()
   anat_x, anat_y, anat_z = np.array(anat_data.shape[0:3])
   anat_dims = {'x':anat_x, 'y':anat_y, 'z':anat_x}
     
   #Load in overlay data
   n_over = len(over_paths)
   over_data = []
   for path in over_paths:
      hdr = nib.load(path)
      data = hdr.get_fdata().squeeze()
      data[data == 0.0] = np.nan
      over_data.append(np.copy(data))

   #Load in thresholding image
   thresh_hdr = nib.load(thresh_path)
   thresh_data = thresh_hdr.get_fdata().squeeze()

   #IO specific to including scatter plot
   if info_path is not None and reg_path is not None:
         
      #Load roi data
      roi_hdr = nib.load(roi_path)
      roi_data = roi_hdr.get_fdata().squeeze()

      #Load in regional data
      reg_hdr = nib.load(reg_path)
      reg_data = reg_hdr.get_fdata().squeeze().T

      #Load in names for each regiion
      roi_names = np.loadtxt(names_path, dtype=np.str_)

      #Read in subject info data frame
      df1 = pd.read_csv(info_path, names=['subj', 'visit', 'cond'])
      df1 = df1.replace(to_replace=['basal', 'hypergly'], value=['Eugly.', 'Hyper.'])
   
      #Construct data frame containing regional data
      df2 = pd.DataFrame(reg_data, columns=roi_names)
      df = pd.concat([df1, df2], axis=1)
      df = df.groupby(['subj', 'cond']).mean().reset_index()

      #Make column sources for spageti plot
      point_source =  ColumnDataSource(data={'cond':df1['cond'],
                                             'subj':df1['subj'],
                                             'roi':reg_data[:, 47]})
      cond = [df['cond'][subj == df['subj']].tolist() for subj in df['subj'].unique()]
      roi = [df['Deep.White.Matter'][subj == df['subj']].tolist() for subj in df['subj'].unique()]
      line_source = ColumnDataSource(data={'cond':cond, 'roi':roi})

   #Define column sources for images
   anat_source = ColumnDataSource(data={'sag':[anat_data[61, :, :].T],
                                        'tra':[anat_data[:, :, 53].T],
                                        'cor':[anat_data[:, 56, :].T]})
   over_sources = []
   for data in over_data:                                     
      over_sources.append(ColumnDataSource(data={'sag':[data[61, :, :].T],
                                                 'tra':[data[:, :, 53].T],
                                                 'cor':[data[:, 56, :].T]}))
   cross_source = ColumnDataSource(data={'x':[61, 61],
                                         'y':[56, 56],
                                         'z':[53, 53],
                                         'anat_x':[0, anat_x],
                                         'anat_y':[0, anat_y],
                                         'anat_z':[0, anat_z]})
   coord_source = ColumnDataSource(data={'x':[61], 'y':[56], 'z':[53]})

   #Compute anatomical range
   anat_mask = anat_data.flatten() != 0
   if anat_mode == 'percentile':
      anat_masked = anat_data.flatten()[anat_mask]
      anat_scale = np.percentile(anat_masked, anat_range)
   else:
      anat_scale = anat_range

   #Compute overlay range if necessary
   over_scales = []
   over_masked = []
   for i in range(n_over):
      over_masked.append(over_data[i].flatten()[anat_mask])
      if over_mode[i] == 'percentile':
         over_scales.append(np.percentile(over_masked[i], over_range[i]))
      else:
         over_scales.append(over_range[i])

   #Define colormaps
   gray_map = LinearColorMapper(low=anat_scale[0], high=anat_scale[1], palette='Greys9',
                                low_color=(0, 0, 0, 0), nan_color=(0, 0, 0, 0))
   over_maps = []
   for i in range(n_over):
      palette_i = interp_palette(all_palettes[over_palettes[i]][11], 255)
      over_maps.append(LinearColorMapper(low=over_scales[i][0],
                                         high=over_scales[i][1], 
                                         palette=palette_i,
                                         nan_color=(0, 0, 0, 0)))

   #Create image plot
   over_rows = []
   for i in range(n_over):
      row_list = []
      for view in ['sag', 'tra', 'cor']:
         row_list.append(overlay_plot(anat_source, over_sources, anat_data,
                                      over_data, i, view, [gray_map, over_maps[i]],
                                      coord_source, cross_source, anat_dims,
                                      title=over_titles[i]))
   
      row_list.append(Spacer(width=25, name='spacer'))
      row_list.append(create_colorbar(over_maps[i], unit=unit))
      if i == 0:
         row_width = sum([p.width for p in row(row_list).children])
         p_r_width = np.int32(row_width / 2.75)
      over_rows.append(row(row_list))
         
      if over_thresh[i] is True: 
      
         #Update image values based on slider
         thresh_func = thresh_wrapper(over_data[i],
                                      over_sources[i],
                                      np.copy(over_data[i]),
                                      np.abs(thresh_data),
                                      coord_source)
         slide_wid.on_change('value', thresh_func)
         thresh_func('init', 1.3, 1.3)
         slide_spac = Spacer(width=p_r_width - int(slide_width / 2), name='spacer')
         slide_row = row(slide_spac, slide_wid)
         over_rows.append(row(Spacer(height=25, name='spacer')))
         over_rows.append(slide_row)

   #Make roi spagetti plot
   if info_path is not None and reg_path is not None:
      p_r = figure(x_range=df1['cond'].unique(),
                   width=p_r_width,
                   height=800,
                   y_axis_label=f'{meas} ({unit})',
                   y_range=[np.min(reg_data) * 0.5, np.max(reg_data) * 1.1])
      p_r.scatter(x='cond', y='roi', source=point_source, size=15)
      p_r.multi_line(xs='cond', ys='roi', source=line_source, line_width=5)
      p_r.axis.major_label_text_font_style = 'normal'
      p_r.axis.major_label_text_font_size  = '32px'
      p_r.yaxis.axis_label_text_font_size = '24px'
      p_r.yaxis.axis_label_text_font_style = 'bold'
      p_r.title = 'Deep White Matter'
      p_r.title.text_font_size = '42px'
      p_r.title.align = 'center'
      p_r.title.text_font_style = 'bold'

      #KDE estimate
      over_dir = os.path.dirname(over_paths[0])
      over_names = [ os.path.basename(x).split('.')[0] for x in over_paths ] 
      den_path = os.path.join(over_dir, f'{over_names[0]}_{over_names[1]}_den.npy')
      try:

         den = np.load(den_path)
      except:
         den = img_kde(over_masked[0], over_masked[1])
         np.save(den_path, den)
      dens_palette = interp_palette(all_palettes['Plasma'][11], 255)
      dens_map = LinearColorMapper(low=np.percentile(den, 2.5),
                                   high=np.percentile(den, 97.5),
                                   palette=dens_palette)

      #Make column source for scatter plot
      scatter_source = ColumnDataSource(data={'baseline':over_masked[0], 
                                              'delta':over_masked[1],
                                              'den':den})

      #Make baseline vs. change scatter plot
      p_s = figure(width=p_r_width, height=800,
                   x_axis_label=f'Eugly. {meas} ({unit})',
                   y_axis_label=f'Delta. {meas} ({unit})')     
      p_s.scatter(x='baseline', y='delta', source=scatter_source,
                  color={'field': 'den', 'transform': dens_map}) 
      p_s.axis.major_label_text_font_style = 'normal'
      p_s.axis.major_label_text_font_size  = '32px'
      p_s.axis.axis_label_text_font_size = '24px'
      p_s.axis.axis_label_text_font_style = 'bold'

      #Add roi update event each each plot
      roi_func = roi_wrapper([point_source, line_source], [roi_data, reg_data], df, coord_source, roi_names, p_r)
      for over_row in over_rows:
         for child in over_row.children:
            if child.name == 'image':
               child.on_event('tap', roi_func)
         
      #Join plots to form final row
      bot_row = row(p_r, Spacer(width=np.int32(row_width / 15), name='spacer'), p_s)
      over_rows.append(Spacer(height=50, name='spacer'))
      over_rows.append(bot_row)

   #Join up all the figures
   out = column(over_rows)

   #Add to document
   return out
   
def figure_one(data_dir):


   def gen_src(df, cond, var):
    
       #Make column sources
       mask = df['Condition'] == cond
       subj = df_to_2d_list(df, 'Id', 'Subject', mask)
       time = df_to_2d_list(df, 'Id', 'Time', mask)
       val = df_to_2d_list(df, 'Id', f'{var}', mask)
       src = ColumnDataSource(df[(mask)])
       ln_src = ColumnDataSource(data={'subj':subj, 'val':val, 'time':time})

       return src, ln_src

   def fit_data(df, var, eu_src, hy_src, knot_srcs, knot=55):

       #Run model fit on glucose
       df['Piece'] = np.maximum(df['Time'] - knot, 0)
       mdl_str = f"{var} ~ Time + Condition + Piece + Time:Condition + Piece:Condition"
       model = smf.mixedlm(mdl_str, df, groups=df["Id"])
       fit = model.fit()

       #Get model predictions
       n_pred = 100
       t_pred = np.linspace(0, 300, n_pred)
       p_pred = np.maximum(t_pred - knot, 0)
       eu_A = np.stack((np.ones(n_pred),
                        np.zeros(n_pred),
                        t_pred,
                        np.zeros(n_pred),
                        p_pred,
                        np.zeros(n_pred))).T
       hy_A = np.stack((np.ones(n_pred),
                        np.ones(n_pred),
                        t_pred,
                        t_pred,
                        p_pred,
                        p_pred)).T
       eu_hat = eu_A @ fit.fe_params
       hy_hat = hy_A @ fit.fe_params

       #Get approximate 95% cis
       eu_ci = np.zeros((n_pred, 2))
       hy_ci = np.zeros((n_pred, 2))
       cov = np.array(fit.cov_params())[0:6, 0:6]
       for i in range(n_pred):
           eu_grad = np.insert(eu_A[i, 1::], 0, 0, axis=0)
           eu_se = np.sqrt(eu_grad @ cov @ eu_grad)
           eu_ci[i, 0] = eu_hat[i] - 1.96 * eu_se
           eu_ci[i, 1] = eu_hat[i] + 1.96 * eu_se
           hy_grad = np.insert(hy_A[i, 1::], 0, 0)
           hy_se = np.sqrt(hy_grad @ cov @ hy_grad)
           hy_ci[i, 0] = hy_hat[i] - 1.96 * hy_se
           hy_ci[i, 1] = hy_hat[i] + 1.96 * hy_se
    
       #Make data sources for predictions
       eu_src.data['time'] = t_pred
       eu_src.data['hat'] = eu_hat
       eu_src.data['lower'] = eu_ci[:, 0]
       eu_src.data['upper'] = eu_ci[:, 1]
       hy_src.data['time'] = t_pred
       hy_src.data['hat'] = hy_hat
       hy_src.data['lower'] = hy_ci[:, 0]
       hy_src.data['upper'] = hy_ci[:, 1]
       knot_srcs[0].data['time'] = [knot]
       knot_srcs[1].data['time'] = [knot]
       knot_srcs[0].data['hat'] = [np.array([1, 0, knot, 0, 0, 0]) @ fit.fe_params]
       knot_srcs[1].data['hat'] = [np.array([1, 1, knot, knot, 0, 0]) @ fit.fe_params]
    
   def slide_wrapper(df, var, eu_src, hy_src, knt_srcs):
       def slide_update(attr, old, new):
           fit_data(df, var, eu_src, hy_src, knt_srcs, new)
       return slide_update

   def blood_fig(sources, var, y_label):

       #Define bokeh interactive tools
       hover = HoverTool(mode='mouse',
                         line_policy='nearest',
                         tooltips=[("(x,y)", "($x, $y)"),
                                   ("Subject", "@Subject"),
                                   ("Condition", "@Condition")])
       tools = [hover, WheelZoomTool(), PanTool(), ResetTool()]
    
       #Data scatter and lines
       p = figure(x_axis_label='Time (min)', title=var, y_axis_label=y_label, tools=tools, height=800, width=1000)
       scatter_1 = p.scatter(x='Time', y=var, source=sources[0], color='#006bb6', alpha=0.3, size=10, legend_label="Eugly.")
       p.multi_line(xs='time', ys='val', source=sources[1], line_width=3, color='#006bb6', alpha=0.1)
       scatter_2 = p.scatter(x='Time', y=var, source=sources[2], color='#b6006b', alpha=0.3, size=10, legend_label="Hyper.")
       p.multi_line(xs='time', ys='val', source=sources[3], line_width=3, color='#b6006b', alpha=0.1)
       p.axis.axis_label_text_font_size = '36px'
       p.axis.axis_label_text_font_style = 'bold'
       p.title.text_font_size = '42px'
       p.axis.major_label_text_font_size = '32px'
       p.title.align = 'center'
       p.title.text_font_style = 'bold'
       p.legend.label_text_font_size = '28px'
       hover.renderers = [scatter_1, scatter_2]
    
    
       #Prediction lines
       p.line(x='time', y='hat', source=sources[4], line_width=8, color='#006bb6')
       p.line(x='time', y='hat', source=sources[5], line_width=8, color='#b6006b')
       eu_band = Band(base='time', lower='lower', upper='upper', source=sources[4],
                      fill_alpha=0.3, fill_color="gray")
       hy_band = Band(base='time', lower='lower', upper='upper', source=sources[5],
                      fill_alpha=0.3, fill_color="gray")
       p.add_layout(eu_band)
       p.add_layout(hy_band)

       #Forrmat legend
       p.legend.background_fill_alpha = 0
       p.legend.location = 'top_left'
       p.legend.margin = 30

       #Add knots
       p.scatter(x='time', y='hat', source=sources[6][0], color='black', size=18)
       p.scatter(x='time', y='hat', source=sources[6][1], color='black', size=18)

       return p
    
   #Load in glucose
   glc_df = pd.read_csv(f'{data_dir}/new_blood_long.csv', delimiter=',')
   glc_df = glc_df.replace(to_replace=['basal', 'hypergly'], value=['Eugly.', 'Hyper.'])

   #Load in insulin data
   ins_df = pd.read_csv(f'{data_dir}/insulin_update_long_filt.csv', delimiter=',')
   ins_df = ins_df.replace(to_replace=['basal', 'hypergly'], value=['Eugly.', 'Hyper.'])

   #Get sources for ploting
   glc_eu_src, glc_eu_ln_src = gen_src(glc_df, 'Eugly.', 'Glucose')
   glc_hy_src, glc_hy_ln_src = gen_src(glc_df, 'Hyper.', 'Glucose')
   ins_eu_src, ins_eu_ln_src = gen_src(ins_df, 'Eugly.', 'Insulin')
   ins_hy_src, ins_hy_ln_src = gen_src(ins_df, 'Hyper.', 'Insulin')  

   #Get inital fit with knot=55
   glc_eu_hat_src = ColumnDataSource(data={})
   glc_hy_hat_src = ColumnDataSource(data={})
   glc_knt_srcs = [ColumnDataSource(data={}), ColumnDataSource(data={})]
   fit_data(glc_df, "Glucose", glc_eu_hat_src, glc_hy_hat_src, glc_knt_srcs)
   ins_eu_hat_src = ColumnDataSource(data={})
   ins_hy_hat_src = ColumnDataSource(data={})
   ins_knt_srcs = [ColumnDataSource(data={}), ColumnDataSource(data={})]
   fit_data(ins_df, "Insulin", ins_eu_hat_src, ins_hy_hat_src, ins_knt_srcs)

   #Join up all the sources
   glc_srcs = [glc_eu_src, glc_eu_ln_src,
               glc_hy_src, glc_hy_ln_src,
               glc_eu_hat_src, glc_hy_hat_src,
               glc_knt_srcs]
   ins_srcs = [ins_eu_src, ins_eu_ln_src,
               ins_hy_src, ins_hy_ln_src,
               ins_eu_hat_src, ins_hy_hat_src,
               ins_knt_srcs]

   #Create figures
   glc_fig = blood_fig(glc_srcs, 'Glucose', 'Conc. (mg/dL)')
   ins_fig = blood_fig(ins_srcs, 'Insulin', 'Conc. (pmol/L)')

   #Create slider
   slide_css = InlineStyleSheet(css=".bk-slider-title { font-size: 30px; }")
   slider = Slider(title="Knot Location (min)", value=55, start=10, end=250, step=1, 
                   width=400, stylesheets=[slide_css], name='slide', align='center')

   figs = [glc_fig, Spacer(width=60), ins_fig]
   vars = ['Glucose', 'Insulin']
   sources = [glc_srcs, ins_srcs]
   dfs = [glc_df, ins_df]
   
   #Add update function to each figure
   for fig, var, src, df in zip(figs, vars, sources, dfs):
      func = slide_wrapper(df, var, src[4], src[5], src[6])
      slider.on_change('value_throttled', func)

   #Join everything up
   return column(row(figs), Spacer(height=25), slider)

def figure_six(data_dir):

   #Function to update slice shown based on slider
   def slider_slc_wrap(src, img_dic):
       def update_src(attrname, old, new):
           for key in img_dic.keys():
               src.data[key] = [img_dic[key][:, :, new].T]
       return update_src

   #Function to update data shown in scatter plot
   def select_wrap(gly, axis):
       def select_update(attrname, old, new):
           if axis == 'x':
               gly.glyph.x = new
           else:
                gly.glyph.y = new
       return select_update

   #Function to update data shown in bar plots
   def select_frac_wrap():
       def select_update(attrname, old, new):

           #Select new data
           if new == "Total":
               class_data = hk_class_dic
               type_data = hk_type_dic
           else:
               class_data = hk_class_norm_dic
               type_data = hk_type_norm_dic

           #Update data source
           hk_class_src.data = class_data
           hk_type_src.data = type_data
    
       return select_update

   #Load in roi names
   roi_names = np.loadtxt(f'{data_dir}/wmparc_with_tiss.csv', delimiter=',', 
                          usecols=[1, 3], skiprows=1, dtype=np.str_)
   #Load in all roi data
   mods = ['fdg', 'om', 'oef', 'ho', 'oc', 'ogi']
   roi_src = ColumnDataSource(data={'names':roi_names[:, 0],
                                    'class':roi_names[:, 1]})
   for mod in mods:
       mod_path = f'{data_dir}/{mod}_wmparc_hypergly_coef.nii.gz'
       roi_src.data[mod] = nib.load(mod_path).get_fdata().squeeze()

   #Add expression data to roi data
   exp_data = np.loadtxt(f'{data_dir}/hk_expression_data.csv', delimiter=',')
   roi_src.data['HK1'] = exp_data[:, 0]
   roi_src.data['HK2'] = exp_data[:, 1]
   roi_src.data['HK1/HK2'] = exp_data[:, 0] / exp_data[:, 1]

   #Load in cell-type fractions
   hk_class = pd.read_csv(f'{data_dir}/hk_class_frac.csv')
   hk_type = pd.read_csv(f'{data_dir}/hk_non_frac.csv')

   #Make dictionaries for class plot
   hk_class_pivot = hk_class.pivot(index='class', values='den', columns='iso')
   hk_class_dic = hk_class_pivot.to_dict('list')
   hk_class_norm = hk_class_pivot.divide(hk_class.groupby('class')['den'].sum(), axis=0)
   hk_class_norm_dic = hk_class_norm.to_dict('list')
   hk_classes = np.unique(hk_class['class'])
   hk_class_dic['class'] = hk_classes
   hk_class_src = ColumnDataSource(data=hk_class_dic)
   hk_class_norm_dic['class'] = hk_classes

   #Make dictionaries for type plot
   hk_type_pivot = hk_type.pivot(index='type', values='den', columns='iso')
   hk_type_dic = hk_type_pivot.to_dict('list')
   hk_type_norm = hk_type_pivot.divide(hk_type.groupby('type')['den'].sum(), axis=0)
   hk_type_norm_dic = hk_type_norm.to_dict('list')
   hk_types = np.unique(hk_type['type'])
   hk_type_dic['type'] = hk_types
   hk_type_src = ColumnDataSource(data=hk_type_dic)
   hk_type_norm_dic['type'] = hk_types

   #Make colormap for gene images
   gene_palette = interp_palette(all_palettes['Plasma'][11], 255)
   gene_map = LinearColorMapper(low=0.25,
                                high=0.75, 
                                palette=gene_palette,
                                nan_color=(0, 0, 0, 0))
   ratio_map = LinearColorMapper(low=0.5,
                                 high=2.5, 
                                 palette=gene_palette,
                                 nan_color=(0, 0, 0, 0))

   #Make colorbars
   gene_bar = create_colorbar(gene_map, unit='Normalized Expression',
                              orientation='horizontal', loc='above')
   ratio_bar = create_colorbar(ratio_map, unit='HK1 / HK2',
                               orientation='horizontal', loc='above')
   gene_bar.height = 100
   ratio_bar.height = 100

   #Make an image plot for each gene
   gene_dic = {}
   gene_src = ColumnDataSource(data={})
   gene_figs = []
   for gene, img in zip(['HK1', 'HK2', 'HK1/HK2'], ['hk1', 'hk2', 'ratio']):

       #Load in expression image
       img_path = f'{data_dir}/{img}_img.nii.gz'
       gene_data = nib.load(img_path).get_fdata().squeeze()
       gene_data[gene_data == 0] = np.nan
       gene_dic[gene] = gene_data
       gene_src.data[gene] = [gene_dic[gene][:, :, 90].T]

       #Make figure
       dw = gene_dic[gene].shape[0]
       dh = gene_dic[gene].shape[1]
       p = figure(x_range=[0, dw], y_range=[0, dh], height=500, width=500, title=gene)
       if gene == 'HK1/HK2':
           p.image(gene, source=gene_src, x=0, y=0, dw=dw, dh=dh,
                   level="image", color_mapper=ratio_map)
       else:
           p.image(gene, source=gene_src, x=0, y=0, dw=dw, dh=dh,
                   level="image", color_mapper=gene_map)

       #Style figure
       p.axis.visible = False
       p.grid.visible = False
       p.outline_line_color= None
       p.toolbar_location = None
       p.title.text_font_style = "bold"
       p.title.text_font_size = "42px"
       p.title.align = 'center'
    
       gene_figs.append(p)

   #Make categorical color map
   u_class = np.unique(roi_names[:, 1])
   n_class = u_class.shape[0]
   class_map = CategoricalColorMapper(palette=all_palettes['Set1'][n_class],
                                      factors=u_class)

   #Create scatter plot
   p_s_tips = [("(x,y)", "($x, $y)"),
               ("Region", "@names"),
               ("Class", "@class")]
   p_s = figure(tooltips=p_s_tips, height=800, width=1000,
                y_axis_label='Delta SUVR',
                x_axis_label='Normallized Gene Expression')
   p_s.add_layout(Legend(location='center'), 'above')
   gene_sc = p_s.scatter(x="HK1/HK2", y='fdg', size=15, source=roi_src,
                         color={'field': 'class', 'transform': class_map},
                         legend_group='class')
   p_s.axis.axis_label_text_font_size = '28px'
   p_s.axis.axis_label_text_font_style = 'bold'
   p_s.axis.major_label_text_font_size = '24px'
   p_s.legend.label_text_font_size = '24px'
   p_s.legend.orientation = 'horizontal'

   #Make slider
   slide_height = 400
   slide_style = InlineStyleSheet(css=".bk-slider-title { font-size: 30px; }")
   slide = Slider(title="Slice", value=90, start=0, end=gene_data.shape[2] - 1,
                  step=1, stylesheets=[slide_style])
   slide.on_change('value', slider_slc_wrap(gene_src, gene_dic))

   #Create metabolite selecter
   sel_css = InlineStyleSheet(css="select {font-size: 24px} label {font-size: 28px; font-weight: bold}")
   met_sel = Select(title="Metabolic Param:", value="fdg",
                    options=[("fdg", "CMRglc"),
                             ("ho", "CBF"),
                             ("om", "CMRO2"),
                             ("ogi", "OGI"),
                             ("oef", "OEF"),
                             ("oc", "CBV")],
                   stylesheets=[sel_css])
   met_sel.on_change('value', select_wrap(gene_sc, 'y'))

   #Create gene selector
   gene_sel =  Select(title="Gene:", value="HK1/HK2",
                      options=["HK1", "HK2", "HK1/HK2"], stylesheets=[sel_css])
   gene_sel.on_change('value', select_wrap(gene_sc, 'x'))

   #Create cell class figure
   isoforms = np.unique(hk_class['iso']).tolist() 
   n_iso = len(isoforms)
   p_c_tips = "$name: @$name"
   p_c = figure(x_range=hk_classes,
                y_axis_label='Fraction of Cells',
                x_axis_label='Cell Class',
                tooltips=p_c_tips,
                height=1000, width=800)
   p_c.add_layout(Legend(location='center'), 'above')
   p_c_stack = p_c.vbar_stack(isoforms, x="class", source=hk_class_src,
                              legend_label=isoforms,
                              color=all_palettes['Set1'][n_iso][::-1],
                              line_color="black", line_width=2)
   p_c.axis.axis_label_text_font_size = '32px'
   p_c.axis.axis_label_text_font_style = 'bold'
   p_c.axis.major_label_text_font_size = '28px'
   p_c.legend.label_text_font_size = '28px'
   p_c.legend.orientation = 'horizontal'

   #Create cell type figure
   p_t = figure(x_range=hk_types, width=1200, height=1000,
                y_axis_label='Fraction of Cells',
                x_axis_label='Cell Type',
                tooltips=p_c_tips)
   p_t.add_layout(Legend(location='center'), 'above')
   p_t_stack = p_t.vbar_stack(isoforms, x="type", source=hk_type_src,
                              legend_label=isoforms,
                              color=all_palettes['Set1'][n_iso][::-1],
                              line_color="black", line_width=2)
   p_t.axis.axis_label_text_font_size = '32px'
   p_t.axis.axis_label_text_font_style = 'bold'
   p_t.axis.major_label_text_font_size = '28px'
   p_t.legend.label_text_font_size = '28px'
   p_t.xaxis.major_label_orientation = 3.14 / 4
   p_t.legend.orientation = 'horizontal'
   
   #Create selector for data type
   frac_sel =  Select(title="Fraction:", value="Total",
                      options=["Total", "Expressing"], stylesheets=[sel_css])
   frac_sel.on_change('value', select_frac_wrap())

   #Join up figure
   fig = column(row(column(row(slide),
                           Spacer(height=25),
                           row(gene_figs),
                           row(Spacer(width=200),
                               gene_bar,
                               Spacer(width=175),
                               ratio_bar
                              )
                          ),
                    Spacer(width=100),
                    p_s,
                    Spacer(width=50),
                    column(met_sel,
                           Spacer(height=25),
                           gene_sel
                          )
                    ),
                Spacer(height=100),
                row(Spacer(width=200),
                    frac_sel,
                    Spacer(width=50),
                    p_c,
                    Spacer(width=250),
                    p_t
                   ) 
               )

   return fig
   
def figure_seven(data_dir):
   
   def gen_src_sim(df, km, src=None):
    
    #Make column sources
    mask = df['Km_scale'] == km
    vhex = df_to_2d_list(df, 'Si_scale', 'Vhex', mask)
    vhex_p = (np.array(vhex) - 3.67) / 3.67 * 100
    so = df_to_2d_list(df, 'Si_scale', 'So', mask)
    if src is None:
        return ColumnDataSource(data={'vhex_q':vhex, 'so':so, 'vhex_p':vhex_p.tolist()})
    else:
        src.data['vhex_q'] = vhex
        src.data['vhex_p'] = vhex_p.tolist()
        src.data['so'] = so

   def unit_change(attr, old, new):
        if new == 'Percent Δ':
            sim_img_src.data['img'] = [sim_mat[:, :, 0]]
            sim_map.high = 60
            p_l_line.glyph.ys = 'vhex_p'
            p_l_gm.glyph.y = 'vhex_p'
            p_l.y_range.start = -100
            p_l.y_range.end = 75
            p_l.yaxis.axis_label = 'Δ HK Flux (%)'
            sim_cb.select('cbar').title = 'Δ HK Flux (%)'
            p_l_eu.data_source.data['y'] = [-100, 75]
            p_l_hy.data_source.data['y'] = [-100, 75]
        else:
            sim_img_src.data['img'] = [sim_mat[:, :, 1]]
            sim_map.high = 2
            p_l_line.glyph.ys = 'vhex_q'
            p_l_gm.glyph.y = 'vhex_q'
            p_l.y_range.start= 0
            p_l.y_range.end = 7.25
            p_l.yaxis.axis_label = 'HK Flux (μM/s)'
            sim_cb.select('cbar').title = 'Δ HK Flux (μM/s)'
            p_l_eu.data_source.data['y'] = [0, 7.25]
            p_l_hy.data_source.data['y'] = [0, 7.25]

   def mouse_click(event: PointEvent):
      y_low = np.floor(event.y)
      y_high = np.ceil(event.y)
      grid_poly.ys = [y_high, y_low, y_low, y_high]
      gen_src_sim(sim_data, km_scales[int(y_low)], sim_l_src)

   #Load in simulation data
   sim_data = pd.read_csv(f'{data_dir}/full_sim_data.csv')
   sim_data['So'] *= 18.0182 * 1E3
   km_scales = np.unique(sim_data['Km_scale'])
   si_scales = np.unique(sim_data['Si_scale'])
   si_labels = [str(i) for i in np.round(np.log10(si_scales), 2)]
   n_si = si_scales.shape[0]

   #Load in gray matter data
   gm_data = pd.read_csv(f'{data_dir}/gm_barros.csv')
   gm_data['so'] *= 18.0182 * 1E3
   gm_data['vhex_q'] = gm_data['vhex']
   gm_data['vhex_p'] = (gm_data['vhex'] - 6.5) / 6.5 * 100
   gm_src = ColumnDataSource(data=gm_data)

   #Generate a simulation curve plot
   sim_l_src = gen_src_sim(sim_data, km_scales[7])
   l_col = all_palettes['Plasma'][n_si][::-1]
   sim_l_src.data['color'] = l_col
   p_l = figure(x_axis_label='Plasma Glucose (mg/dL)',
                y_axis_label='Δ HK Flux (%)',
                height=800, width=800,
                y_range=[-100, 75])
   p_l_line = p_l.multi_line(xs='so', ys='vhex_p', source=sim_l_src,
                             line_color='color', line_width=6)
   p_l_gm = p_l.line(x='so', y='vhex_p', source=gm_src,
                     line_color='gray', line_width=10)
   p_l_eu = p_l.line(x=[100, 100], y=[-100, 75], line_color='black',
                     dash='dashed', line_width=5)
   p_l_hy = p_l.line(x=[300, 300], y=[-100, 75], line_color='black',
                     dash='dashed', line_width=5)

   #Style line plot
   p_l.axis.axis_label_text_font_size = '32px'
   p_l.axis.axis_label_text_font_style = 'bold'
   p_l.axis.major_label_text_font_size = '28px'

   #Make legend
   l_leg_items = []
   for i in range(n_si):
       l_leg_items.append(LegendItem(label=si_labels[i], renderers=[p_l_line], index=i))
   l_leg = Legend(location='center', items=l_leg_items)

   #Style legend
   p_l.add_layout(l_leg, 'above')
   p_l.legend.label_text_font_size = '24px'
   p_l.legend.orientation = 'horizontal'
   p_l.title = r"\[10^{GM \, S_i \, Fraction}\]"
   p_l.title.text_font_size = '24px'
   p_l.title.vertical_align = 'top'

   #Create palette for matrix plot
   sim_palette = interp_palette(all_palettes['Plasma'][11], 255)
   sim_map = LinearColorMapper(low=0,
                               high=60, 
                               palette=sim_palette,
                               nan_color=(0, 0, 0, 0))

   #Compute difference between hyperglycemia and euglycemia
   sim_mat = np.zeros((70, 2))
   sim_data_list = [i for i in sim_data.groupby(['Km_scale', 'Si_scale'])]
   for i in range(70):
       so = sim_data_list[i][1]['So']
       vhex = sim_data_list[i][1]['Vhex']
       new = interp.interp1d(so, vhex, kind='linear')([100, 300])
       sim_mat[i, 0] = (new[1] - new[0]) / new[0] * 100
       sim_mat[i, 1] = (new[1] - new[0])
   sim_mat = sim_mat.reshape((10, 7, 2))
   sim_img_src = ColumnDataSource(data={'img':[sim_mat[:, :, 0]]})

   #Make image figure
   p_m = figure(x_range=[0, 7], y_range=[0, 10],
                x_axis_label='Fraction of GM Intracellular',
                y_axis_label='Fraction of HK1 Kₘ',
                width=900, height=800)
   p_m.image("img", source=sim_img_src, x=0, y=0, dw=7, dh=10,
             level='image', color_mapper=sim_map)

   #Create a colorbar for image
   sim_cb = create_colorbar(sim_map, unit='Δ HK Flux (%)',
                            orientation='vertical', loc='left')
   sim_cb.height = 800
   sim_cb.width= 100
   sim_cb.select('cbar').major_label_text_font_size = '24px'

   #Style image plot
   p_m.xaxis.ticker = [0.5, 2.5, 4.5, 6.5]
   p_m.yaxis.ticker = [0.5, 2.5, 4.5, 6.5, 8.5]
   p_m.xaxis.major_label_overrides = {0.5:r"\[10^{-1}\]", 2.5:r"\[10^{-0.5}\]", 
                                      4.5:r"\[10^{0}\]", 6.5:r"\[10^{0.5}\]"}
   p_m.yaxis.major_label_overrides = {0.5:r"\[10^{-1}\]", 2.5:r"\[10^{-0.5}\]", 
                                      4.5:r"\[10^{0}\]", 6.5:r"\[10^{0.5}\]",  
                                      8.5:r"\[10^{1}\]"}
   p_m.axis.major_label_text_font_size = '28px'
   p_m.axis.axis_label_text_font_size = '32px'
   p_m.axis.axis_label_text_font_style = 'bold'
   p_m.xaxis.ticker.minor_ticks = [0, 1, 2, 3, 4, 5, 6]
   p_m.yaxis.ticker.minor_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
   p_m.grid.minor_grid_line_color = 'black'
   p_m.grid.minor_grid_line_width = 5
   p_m.grid.minor_grid_line_alpha = 1
   p_m.grid.grid_line_alpha = 0
   p_m.axis.major_tick_out = 10
   p_m.axis.major_tick_in = 10
   p_m.axis.minor_tick_line_alpha = 0
   grid_poly = PolyAnnotation(line_color="white",
                              line_width=8,
                              line_alpha=1,
                              xs=[0, 0, 7, 7], #top left, bottom left, bottom right, top right
                              ys=[8, 7, 7,8],
                              fill_alpha=0)
   p_m.add_layout(grid_poly)

   #Add mouse slick event to image plot
   p_m.on_event('tap', mouse_click)

   #Create quant/per selector
   norm_css = InlineStyleSheet(css="select {font-size: 24px} label {font-size: 28px; font-weight: bold}")
   norm_sel =  Select(title="Normalization:", value="Percent Δ",
                      options=["Percent Δ", "None"], stylesheets=[norm_css])
   norm_sel.on_change('value', unit_change)
   
   #Join up all the plots
   out = row(p_m, Spacer(width=50), sim_cb, Spacer(width=150), p_l,
             Spacer(width=50), norm_sel)
             
   return out
   