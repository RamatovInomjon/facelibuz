# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------------------
# the number of output scales (loss branches)
param_num_output_scales = 4

# bbox lower bound for each scale
param_bbox_small_list = [10, 20, 40, 80]
assert len(param_bbox_small_list) == param_num_output_scales

# bbox upper bound for each scale
param_bbox_large_list = [20, 40, 80, 160]
assert len(param_bbox_large_list) == param_num_output_scales
param_receptive_field_list = param_bbox_large_list

# RF stride for each scale
param_receptive_field_stride = [4, 8, 16, 32]

# the start location of the first RF of each scale
param_receptive_field_center_start = [3, 7, 15, 31]

#----------------------------------
# models path's to load at runtime
#----------------------------------
#onnx file path
onnx_file_path = './onnx_files/v2.onnx'
symbol_file_path = 'model/lfhd.json'
model_file_path = 'model/lfhd_v3.params'
# model_file_path = 'model/lfhd.params'
age_weight_file_path = 'model/age_v3.h5'
gender_weight_file_path = 'model/gender_v3.h5'

#trt_file_cache
trt_file_cache = '../trt_file_cache/'

# backup settings
backup = '../packets'
send_report = 'history.csv'
unsend_report = 'unsend.csv'

settings_file = '../server_configs.json' # json file where all settings from server are stored
unsend_marketing_results = '../packets/marketing/unsend.csv'

"""
API System (api.tass-vision.uz)
"""
api_base = 'http://api.tass-vision.uz:9600/'

# API methods
api_send_packet = 'api/packets/add' # send packets to hr server
api_header = { 'Content-Type': 'application/json; charset=utf-8', 'Connection': 'keep-alive', 'Authorization': ''} # internal token must be formatted for an empty key

"""
DASHBOARD Data System (data.tass-vision.uz) [DDS]
"""

dashboard_base = "https://data.tass-vision.uz:3005/" # 3000 for http, 3005 for https

# DDS methods
dashboard_get_settings = "api/device-info/{}?organization_token={}&version={}" # passed mac and internal token of device
dashboard_send_exception = "api/device-info/debug?organization_token={}" # passed internal token and debugging string in request body
dashboard_header = { 'Content-Type': 'application/json' }
dashboard_images_send = 'admin/heatmap_images/'

# Snap save path
imagepath = 'tmp_image.jpeg'