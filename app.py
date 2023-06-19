#Load libs
from bokeh.models import TabPanel, Tabs
from bokeh_utils import *
import os

#Prep
app_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = f'{app_dir}/data'

#CMRglc figure
fig_2 = three_row(f'{data_dir}/template_2mm_masked.nii.gz', 
                  [f'{data_dir}/cmrglc_basal.nii.gz',
                  f'{data_dir}/cmrglc_hypergly_coef.nii.gz'],
                  f'{data_dir}/cmrglc_hypergly_logp_fdr_05.nii.gz',
                  "CMRglc", "uMol/hg/min",
                  info_path = f'{data_dir}/cmrglc_info.csv',
                  reg_path = f'{data_dir}/cmrglc_wmparc.nii.gz',
                  over_titles=['CMRglc: Eugly', 'Hyper. - Eugly.'],
                  over_range = [[10, 52], [-15, 15]],
                  over_thresh = [False, True],
                  over_palettes = ['Plasma', 'RdBu'],
                  over_mode = ['absolue', 'absolute'],
                  roi_path=f'{data_dir}/winner_wmparc_comb_on_MNI152_2mm_masked.nii.gz',
                  names_path=f'{data_dir}/wmparc_names.txt')

#Oxygen figure
fig_3 = three_row(f'{data_dir}/template_2mm_masked.nii.gz', 
                  [f'{data_dir}/oxy_suvr_basal.nii.gz',
                  f'{data_dir}/oxy_suvr_hypergly_coef.nii.gz'],
                  f'{data_dir}/oxy_suvr_hypergly_logp_fdr_05.nii.gz',
                  "CMRO2", "SUVR",
                  info_path=f'{data_dir}/oxy_suvr_info.csv',
                  reg_path=f'{data_dir}/all_om_suvr_po.nii.gz',
                  over_range = [[0.4, 1.4], [-0.1, 0.1]], 
                  over_thresh = [False, True],
                  over_palettes = ['Plasma', 'RdBu'],
                  over_titles= ['CMRO2: Eugly', 'Hyper. - Eugly.'],
                  over_mode = ['absolue', 'absolute'],
                  roi_path=f'{data_dir}/winner_wmparc_comb_on_MNI152_2mm_masked.nii.gz',
                  names_path=f'{data_dir}/wmparc_names.txt')

#OGI figure               
fig_4 = three_row(f'{data_dir}/template_2mm_masked.nii.gz', 
                  [f'{data_dir}/rogi_suvr_basal.nii.gz',
                  f'{data_dir}/rogi_suvr_hypergly.nii.gz',
                  f'{data_dir}/rogi_suvr_hypergly_coef.nii.gz'],
                  f'{data_dir}/rogi_suvr_hypergly_logp_fdr_05.nii.gz',
                  "rOGI", "SUVR",
                  over_titles=['rOGI: Eugly', 'Hyper.', 'Hyper. - Eugly.'],
                  over_range = [[0.6, 1.5], [0.6, 1.5], [-0.5, 0.5]], 
                  over_thresh = [False, False, True],
                  over_palettes = ['Plasma', 'Plasma', 'RdBu'],
                  over_mode = ['absolue', 'absolute', 'absolute'])

#CBF Figure
fig_5 = three_row(f'{data_dir}/template_2mm_masked.nii.gz', 
                  [f'{data_dir}/ho_suvr_basal.nii.gz',
                  f'{data_dir}/ho_suvr_hypergly_coef.nii.gz'],
                  f'{data_dir}/ho_suvr_hypergly_logp_fdr_05.nii.gz',
                  "CBF", "SUVR",
                  over_titles=['CBF: Eugly', 'Hyper. - Eugly.'],
                  info_path=f'{data_dir}/ho_suvr_info.csv',
                  reg_path=f'{data_dir}/all_ho_suvr_po.nii.gz',
                  over_range = [[0.4, 1.4], [-0.1, 0.1]], 
                  over_thresh = [False, True],
                  over_palettes = ['Plasma', 'RdBu'],
                  over_mode = ['absolue', 'absolute'],
                  roi_path=f'{data_dir}/winner_wmparc_comb_on_MNI152_2mm_masked.nii.gz',
                  names_path=f'{data_dir}/wmparc_names.txt')

#Create tabs
tab_1 = TabPanel(child=figure_one(data_dir), title="Figure 1")
tab_2 = TabPanel(child=fig_2, title="Figure 2")
tab_3 = TabPanel(child=fig_3, title="Figure 3")
tab_4 = TabPanel(child=fig_4, title="Figure 4")
tab_5 = TabPanel(child=fig_5, title="Figure 5")
tab_6 = TabPanel(child=figure_six(data_dir), title="Figure 6")
tab_7 = TabPanel(child=figure_seven(data_dir), title="Figure 7")

#Create css
tab_css = ":host(.bk-Tabs) .bk-header {font-size: 36px; font-weight: bold; border-color:black}"
act_css = ":host{.bk-Tabs} .bk-tab.bk-active {background-color:#D3D3D3; font-color: black; border-color:black}"
bor_css = ":host{.bk-Tabs} .bk-tab {border-color:black}"

#Show app
tab_list = [tab_1, tab_2, tab_3, tab_4, tab_5, tab_6, tab_7]
curdoc().add_root(Tabs(tabs=tab_list, stylesheets=[tab_css, act_css, bor_css]))




