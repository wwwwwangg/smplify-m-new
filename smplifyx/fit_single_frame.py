# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img

from optimizers import optim_factory

import fitting
from human_body_prior.tools.model_loader import load_vposer

# MobileNetV3 initialization
try:
    from mobilenet_init import SMPLInitNet
    HAS_MOBILENET = True
except ImportError:
    HAS_MOBILENET = False
    print("Warning: MobileNetV3 not available, using default initialization")

# PyVista for 3D visualization with acupoints and meridians
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("Warning: PyVista not available, 3D visualization disabled")


# =========================
# 穴位和经络配置数据
# =========================

# 面部穴位索引
FACIAL_ACUPOINTS = {
    "印堂": 8939, "鱼腰一": 9225, "鱼腰二": 9351, "太阳穴一": 3161, "太阳穴二": 2142,
    "球后一": 2331, "球后二": 953, "素髎": 8970, "水沟": 8959, "兑端": 8985,
    "承浆": 8946, "攒竹一": 2177, "攒竹二": 672, "睛明一": 3151, "睛明二": 2126,
    "丝竹空一": 9158, "丝竹空二": 9316, "瞳子髎一": 2433, "瞳子髎二": 1164,
    "颧髎一": 3115, "颧髎二": 2080, "迎香一": 2741, "迎香二": 1605,
    "承泣一": 2527, "承泣二": 1376, "四白一": 3121, "四白二": 499,
    "地仓一": 2735, "地仓二": 1598
}

# 面部穴位颜色映射
FACIAL_ACUPOINT_COLOR_MAP = {
    "印堂": "purple", "鱼腰": "purple", "太阳穴": "purple", "球后": "purple", "上迎香": "purple",
    "素髎": "orange", "水沟": "orange", "兑端": "orange", "承浆": "green",
    "攒竹": "blue", "睛明": "blue", "丝竹空": "darkgreen", "瞳子髎": "gold",
    "阳白": "gold", "颧髎": "skyblue", "迎香": "brown", "口禾髎": "brown",
    "承泣": "red", "四白": "red", "下关": "red", "颊车": "red", "大迎": "red",
    "巨髎": "red", "地仓": "red"
}

# 经络定义（左右两侧）
MERIDIANS = {
    # 肺经
    "lung": {
        "云门穴": 3237, "中府穴": 4167, "天府穴": 3975, "侠白穴": 4037, "尺泽穴": 4313,
        "孔最穴": 4209, "列缺穴": 4537, "经渠穴": 4539, "太渊穴": 4632, "鱼际穴": 4637, "少商穴": 5336
    },
    "lung2": {
        "云门穴": 6911, "中府穴": 6000, "天府穴": 6784, "侠白穴": 6181, "尺泽穴": 7113,
        "孔最穴": 6964, "列缺穴": 7301, "经渠穴": 7277, "太渊穴": 7459, "鱼际穴": 7598, "少商穴": 8073
    },
    # 肝经
    "LR": {
        "大敦": 5786, "行间": 5785, "太冲": 5918, "中封": 5874, "蠡沟": 4006,
        "中都": 4004, "膝关": 3629, "曲泉": 3632, "阴包": 3596, "足五里": 3538,
        "阴廉": 3864, "急脉": 3508, "章门": 3476, "期门": 3852
    },
    "LR2": {
        "大敦": 8481, "行间": 8540, "太冲": 8648, "中封": 8574, "蠡沟": 6468,
        "中都": 6465, "膝关": 6450, "曲泉": 6449, "阴包": 6346, "足五里": 8840,
        "阴廉": 6829, "急脉": 6663, "章门": 6859, "期门": 6051
    },
    # 心包经
    "PC": {
        "天池": 3828, "天泉": 3259, "曲泽": 4267, "郄门": 4180, "间使": 4561,
        "内关": 4534, "大陵": 4893, "劳宫": 4828, "中冲": 5060
    },
    "PC2": {
        "天池": 6017, "天泉": 7225, "曲泽": 7024, "郄门": 7305, "间使": 7283,
        "内关": 7592, "大陵": 7449, "劳宫": 7440, "中冲": 7818
    },
    # 小肠经
    "SI": {
        "少泽": 5289, "前谷": 5225, "后溪": 4675, "腕骨": 4898, "阳谷": 4703,
        "养老": 4721, "支正": 4588, "小海": 4371, "肩贞": 5597, "臑俞": 4503,
        "天宗": 5605, "秉风": 3883, "曲垣": 5564, "肩外俞": 3362, "肩中俞": 3849,
        "天窗": 3186, "天容": 496, "颧髎": 2081, "听宫": 565
    },
    "SI2": {
        "少泽": 8009, "前谷": 8047, "后溪": 7434, "腕骨": 7438, "阳谷": 7381,
        "养老": 7458, "支正": 7290, "小海": 7110, "肩贞": 6108, "臑俞": 7239,
        "天宗": 6126, "秉风": 6633, "曲垣": 8276, "肩外俞": 6123, "肩中俞": 6604,
        "天窗": 5957, "天容": 8801, "颧髎": 3127, "听宫": 1050
    },
    # 脾经
    "SP": {
        "隐白": 5787, "大都": 5890, "太白": 5918, "公孙": 8861, "商丘": 5747,
        "三阴交": 3791, "漏谷": 3789, "地机": 3784, "阴陵泉": 3781, "血海": 3672,
        "箕门": 3580, "冲门": 3511, "府舍": 3916, "腹结": 4422, "大横": 5513,
        "腹哀": 3977, "食窦": 3556, "天溪": 3559, "胸乡": 3230, "周容": 5436, "大包": 3274
    },
    "SP2": {
        "隐白": 8481, "大都": 8584, "太白": 8612, "公孙": 8649, "商丘": 8441,
        "三阴交": 6549, "漏谷": 6547, "地机": 6542, "阴陵泉": 6411, "血海": 6537,
        "箕门": 6341, "冲门": 6272, "府舍": 6664, "腹结": 7158, "大横": 8235,
        "腹哀": 6725, "食窦": 6317, "天溪": 6320, "胸乡": 6334, "周容": 8170, "大包": 6040
    },
    # 胃经
    "ST": {
        "承泣": 9374, "四白": 2094, "巨髎": 1585, "地仓": 1768, "大迎": 9192,
        "颊车": 9032, "下关": 566, "头维": 1888, "人迎": 372, "水突": 3189,
        "气舍": 5618, "缺盆": 3217, "气户": 3936, "库房": 3220, "屋翳": 3296,
        "膺窗": 5436, "乳中": 5645, "乳根": 3299, "不容": 3555, "承满": 3551,
        "梁门": 3554, "关门": 3977, "太乙": 3549, "滑肉门": 3839, "天枢": 5531,
        "外陵": 4423, "大巨": 3842, "水道": 3545, "归来": 3794, "气冲": 4148,
        "髀关": 4134, "伏兔": 3575, "阴市": 3660, "梁丘": 3661, "犊鼻": 3700,
        "足三里": 3729, "上巨虚": 3750, "条口": 3751, "下巨虚": 3769, "丰隆": 3745,
        "解溪": 5745, "冲阳": 5880, "陷谷": 5922, "内庭": 5895, "厉兑": 5810
    },
    "ST2": {
        "承泣": 9218, "四白": 3121, "巨髎": 2712, "地仓": 9245, "大迎": 8830,
        "颊车": 8735, "下关": 1961, "头维": 2953, "人迎": 1210, "水突": 5932,
        "气舍": 5618, "缺盆": 5980, "气户": 6684, "库房": 5983, "屋翳": 6059,
        "膺窗": 8170, "乳中": 8339, "乳根": 6206, "不容": 6316, "承满": 6312,
        "梁门": 6315, "关门": 6725, "太乙": 6310, "滑肉门": 6594, "天枢": 8235,
        "外陵": 7159, "大巨": 6597, "水道": 6306, "归来": 6551, "气冲": 6892,
        "髀关": 6264, "伏兔": 6336, "阴市": 6421, "梁丘": 6422, "犊鼻": 6461,
        "足三里": 6490, "上巨虚": 6508, "条口": 6509, "下巨虚": 6527,
        "解溪": 8570, "冲阳": 8574, "陷谷": 8614, "内庭": 8588, "厉兑": 8504
    },
    # 三焦经
    "TE": {
        "关冲": 5140, "液门": 5206, "中渚": 4769, "阳池": 4679, "外关": 4556,
        "支沟": 4553, "会宗": 4554, "三阳络": 4592, "四渎": 4323, "天井": 4383,
        "清泠渊": 4019, "消泺": 5475, "臑会": 4506, "肩髎": 4439, "天髎": 5463,
        "天牖": 552, "翳风": 9104, "瘈脉": 1902, "颅息": 412, "角孙": 1921,
        "耳门": 1553, "（耳）和髎": 4, "丝竹空": 1440
    },
    "TE2": {
        "关冲": 7909, "液门": 7539, "中渚": 8126, "阳池": 7415, "外关": 7308,
        "支沟": 6956, "会宗": 6957, "三阳络": 6934, "四渎": 6937, "天井": 8322,
        "清泠渊": 7006, "消泺": 6182, "臑会": 6784, "肩髎": 6002, "天髎": 8188,
        "天牖": 3142, "翳风": 957, "瘈脉": 600, "颅息": 1040, "角孙": 2263,
        "耳门": 1935, "（耳）和髎": 920, "丝竹空": 2575
    },
    # 膀胱经
    "BL": {
        "睛明": 9261, "攒竹": 2127, "眉冲": 575, "曲差": 573, "五处": 632,
        "承光": 588, "通天": 1877, "络却": 9331, "玉枕": 1299, "天柱": 11,
        "大杼": 3198, "风门": 3446, "肺俞": 3941, "厥阴俞": 3850, "心俞": 4391,
        "督俞": 3358, "膈俞": 5548, "肝俞": 5633, "胆俞": 3383, "脾俞": 3400,
        "胃俞": 5415, "三焦俞": 5629, "肾俞": 5502, "气海俞": 4402, "大肠俞": 4403,
        "关元俞": 4405, "小肠俞": 5675, "膀胱俞": 3472, "中膂俞": 3473, "白环俞": 3884,
        "上髎": 5614, "次髎": 5613, "中髎": 5934, "下髎": 5575, "会阳": 5574,
        "承扶": 3464, "殷门": 4093, "浮郄": 3634, "委阳": 3680, "委中": 3693,
        "附分": 3444, "魄户": 3377, "膏肓": 5458, "神堂": 3365, "譩譆": 3525,
        "膈关": 5521, "魂门": 5427, "阳纲": 3844, "意舍": 3845, "胃仓": 5405,
        "肓门": 3888, "志室": 3887, "胞肓": 5697, "秩边": 5683, "合阳": 3723,
        "承筋": 4105, "承山": 3761, "飞扬": 3767, "跗阳": 5760, "昆仑": 8841,
        "仆参": 8728, "申脉": 8840, "金门": 5925, "京骨": 5924, "束骨": 5901,
        "足通谷": 5872, "至阴": 5835
    },
    "BL2": {
        "睛明": 9041, "攒竹": 3152, "眉冲": 2002, "曲差": 2000, "五处": 2129,
        "承光": 2016, "通天": 2959, "络却": 9188, "玉枕": 2475, "天柱": 112,
        "大杼": 5961, "风门": 7207, "肺俞": 6689, "厥阴俞": 6605, "心俞": 7127,
        "督俞": 6119, "膈俞": 8261, "肝俞": 8327, "胆俞": 6144, "脾俞": 6161,
        "胃俞": 8149, "三焦俞": 8323, "肾俞": 8224, "气海俞": 7138, "大肠俞": 7139,
        "关元俞": 7141, "小肠俞": 8369, "膀胱俞": 6233, "中膂俞": 6234, "白环俞": 6634,
        "上髎": 5614, "次髎": 5613, "中髎": 5934, "下髎": 5575, "会阳": 5574,
        "承扶": 6225, "殷门": 6837, "浮郄": 6395, "委阳": 6441, "委中": 6454,
        "附分": 6205, "魄户": 6139, "膏肓": 8192, "神堂": 6126, "譩譆": 6286,
        "膈关": 8241, "魂门": 8161, "阳纲": 6599, "意舍": 6600, "胃仓": 8139,
        "肓门": 6638, "志室": 6637, "胞肓": 8391, "秩边": 8377, "合阳": 6484,
        "承筋": 6849, "承山": 6519, "飞扬": 6525, "跗阳": 8454, "昆仑": 8629,
        "仆参": 8624, "申脉": 8628, "金门": 8617, "京骨": 8616, "束骨": 8595,
        "足通谷": 8566, "至阴": 8529
    },
    # 胆经
    "GB": {
        "瞳子髎": 2046, "听会": 786, "上关": 2011, "颔厌": 1975, "悬颅": 1973,
        "悬厘": 1997, "曲鬓": 1979, "率谷": 1992, "天冲": 2038, "浮白": 1970,
        "头窍阴": 1947, "完骨": 1896, "本神": 9322, "阳白": 573, "头临泣": 709,
        "目窗": 581, "正营": 1892, "承灵": 638, "脑空": 1308, "风池": 1493,
        "肩井": 3375, "渊腋": 4033, "辄筋": 5447, "日月": 3836, "京门": 4118,
        "带脉": 5427, "五枢": 4084, "维道": 3543, "居髎": 4111, "环跳": 5684,
        "风市": 3534, "中渎": 3603, "膝阳关": 3640, "阳陵泉": 3682, "阳关": 3815,
        "外丘": 3715, "光明": 3747, "阳辅": 3748, "悬钟": 3765, "丘墟": 8935,
        "足临泣": 5929, "地五会": 5900, "侠溪": 5901, "足窍阴": 5822
    },
    "GB2": {
        "瞳子髎": 2322, "听会": 2253, "上关": 3059, "颔厌": 3035, "悬颅": 3033,
        "悬厘": 3053, "曲鬓": 3039, "率谷": 3048, "天冲": 3073, "浮白": 3030,
        "头窍阴": 3011, "完骨": 2974, "本神": 9171, "阳白": 2000, "头临泣": 2200,
        "目窗": 2013, "正营": 2970, "承灵": 2131, "脑空": 2476, "风池": 2629,
        "肩井": 6136, "渊腋": 6780, "辄筋": 8337, "日月": 6591, "京门": 6862,
        "带脉": 8161, "五枢": 6828, "维道": 6304, "居髎": 6855, "环跳": 8379,
        "风市": 6295, "中渎": 6364, "膝阳关": 6401, "阳陵泉": 6443, "阳关": 6847,
        "外丘": 6476, "光明": 6504, "阳辅": 6503, "悬钟": 6526, "丘墟": 8627,
        "足临泣": 8621, "地五会": 8594, "侠溪": 8595, "足窍阴": 8516
    },
    # 任脉
    "ren": {
        "承浆穴": 8946, "廉泉穴": 8793, "天突穴": 5618, "璇玑穴": 5619, "华盖穴": 5528,
        "紫宫穴": 5935, "玉堂穴": 5937, "膻中穴": 5945, "中庭穴": 5532, "鸠尾穴": 5534,
        "巨阙穴": 3855, "上脘穴": 3856, "中脘穴": 5950, "建里穴": 3851, "下脘穴": 3852,
        "水分穴": 5948, "神阙穴": 5939, "阴交穴": 4291, "气海穴": 5942, "石门穴": 5946,
        "关元穴": 4320, "中极穴": 4321, "曲骨穴": 5600, "会阴穴": 3736
    },
    # 督脉
    "GV": {
        "长强": 4066, "腰俞": 5934, "腰阳关": 5494, "命门": 5495, "悬枢": 5496,
        "脊中": 5489, "中枢": 5486, "筋缩": 5487, "至阳": 5499, "灵台": 5500,
        "神道": 5932, "身柱": 5921, "淘道": 3832, "大椎": 5484, "哑门": 9006,
        "风府": 8954, "脑户": 8980, "强间": 8989, "后顶": 8974, "百会": 9237,
        "前顶": 9011, "囟会": 8972, "上星": 9012, "神庭": 8963, "素髎": 8970,
        "水沟": 8981, "兑端": 8990, "龈交": 8977, "印堂": 9016
    },
    # 肾经
    "KI": {
        "涌泉": 8898, "然谷": 8868, "太溪": 5729, "大钟": 5757, "水泉": 8876,
        "照海": 8878, "复溜": 6517, "交信": 6519, "筑宾": 6500, "阴谷": 6387,
        "横骨": 5601, "大赫": 5949, "气穴": 4320, "四满": 5615, "中注": 5946,
        "肓俞": 4292, "商曲": 5948, "石关": 3852, "阴都": 3851, "腹通谷": 5950,
        "幽门": 3856, "步廊": 3557, "神封": 3317, "灵墟": 3982, "神藏": 3224,
        "彧中": 3296, "俞府": 3220
    },
    "KI2": {
        "涌泉": 8686, "然谷": 8650, "太溪": 8682, "大钟": 8642, "水泉": 8663,
        "照海": 8680, "复溜": 8445, "交信": 8424, "筑宾": 6906, "阴谷": 6454,
        "横骨": 3493, "大赫": 3495, "气穴": 4424, "四满": 5723, "中注": 3491,
        "肓俞": 3546, "商曲": 3967, "石关": 3963, "阴都": 3962, "腹通谷": 5426,
        "幽门": 3837, "步廊": 6642, "神封": 8164, "灵墟": 6321, "神藏": 6089,
        "彧中": 6090, "俞府": 6687
    },
    # 大肠经
    "dachang": {
        "迎香": 3107, "口禾髎": 8931, "扶突": 373, "天鼎": 3192, "巨骨": 5465,
        "肩髃": 3875, "臂臑": 4076, "手五里": 4007, "肘髎": 4348, "曲池": 4523,
        "手三里": 4337, "上廉": 4525, "下廉": 4368, "温溜": 4180, "偏历": 4561,
        "阳溪": 4584, "合谷": 4837, "三间": 4609, "二间": 4874, "商阳": 4919
    },
    "dachang2": {
        "迎香": 7657, "口禾髎": 7473, "扶突": 7343, "天鼎": 7344, "巨骨": 7422,
        "肩髃": 7271, "臂臑": 7297, "手五里": 6923, "肘髎": 6920, "曲池": 6942,
        "手三里": 6991, "上廉": 7040, "下廉": 7085, "温溜": 6883, "偏历": 7203,
        "阳溪": 6210, "合谷": 6213, "三间": 2976, "二间": 2751, "商阳": 3096
    },
    # 心经
    "xin": {
        "极泉": 4174, "青灵": 4375, "少海": 4322, "灵道": 4586, "通里": 4552,
        "阴郄": 4549, "神门": 4762, "少府": 5391, "少冲": 5263
    },
    "xin2": {
        "极泉": 4398, "青灵": 4011, "少海": 4296, "灵道": 4572, "通里": 4573,
        "阴郄": 4900, "神门": 4719, "少府": 4665, "少冲": 5251
    }
}

# 经络颜色映射
MERIDIAN_COLOR_MAP = {
    "lung": "blue", "lung2": "blue",
    "LR": "silver", "LR2": "silver",
    "PC": "yellow", "PC2": "yellow",
    "SI": "white", "SI2": "white",
    "SP": "black", "SP2": "black",
    "ST": "grey", "ST2": "grey",
    "TE": "pink", "TE2": "pink",
    "BL": "khaki", "BL2": "khaki",
    "GB": "brown", "GB2": "brown",
    "ren": "green",
    "GV": "beige",
    "KI": "gold", "KI2": "gold",
    "dachang": "orange", "dachang2": "orange",
    "xin": "purple", "xin2": "purple"
}


def compute_meridian_paths(mesh, meridian_dict):
    """
    计算经络路径（使用测地线连接穴位）
    
    Args:
        mesh: pyvista PolyData网格
        meridian_dict: 经络穴位字典 {穴位名: 顶点索引}
    
    Returns:
        path: pyvista PolyData路径对象，如果穴位数<2则返回None
    """
    indices = list(meridian_dict.values())
    if len(indices) < 2:
        return None
    
    path = None
    for i in range(len(indices) - 1):
        try:
            segment = mesh.geodesic(indices[i], indices[i + 1])
            path = segment if path is None else path.merge(segment)
        except Exception as e:
            print(f"[Warning] Failed to compute geodesic between indices {indices[i]} and {indices[i+1]}: {e}")
            continue
    return path


def visualize_acupoints_and_meridians(vertices, faces, output_path="result.png"):
    """
    可视化人体网格、穴位和经络
    
    Args:
        vertices: 顶点坐标 numpy array (N, 3)
        faces: 面数据 numpy array (M, 3)
        output_path: 输出图片路径
    """
    if not HAS_PYVISTA:
        print("[Warning] PyVista not available, skipping acupoint visualization")
        return
    
    # 构建 PyVista 网格
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
    mesh = pv.PolyData(vertices, faces_pv)
    
    # 计算所有经络路径
    meridian_paths = {}
    for mer_name, mer_dict in MERIDIANS.items():
        meridian_paths[mer_name] = compute_meridian_paths(mesh, mer_dict)
    
    # 创建绘图器
    plotter = pv.Plotter()
    plotter.set_background('white')
    plotter.enable_eye_dome_lighting()
    
    # 添加人体网格
    plotter.add_mesh(mesh, color=(1.0, 1.0, 1.0), opacity=1.0, 
                    show_edges=False, smooth_shading=True)
    
    # 添加面部穴位
    for name, idx in FACIAL_ACUPOINTS.items():
        if idx >= len(vertices):
            print(f"[警告] 穴位 '{name}' 索引 {idx} 超出顶点范围，跳过")
            continue
        
        center = vertices[idx]
        # 匹配颜色
        point_color = "black"
        for key in FACIAL_ACUPOINT_COLOR_MAP:
            if key in name:
                point_color = FACIAL_ACUPOINT_COLOR_MAP[key]
                break
        
        sphere = pv.Sphere(radius=0.004, center=center)
        plotter.add_mesh(sphere, color=point_color, smooth_shading=True)
        plotter.add_point_labels(
            [center], [name],
            font_size=9, text_color=point_color,
            shadow=False, shape_opacity=0.0,
            point_size=0, point_color=None,
            always_visible=True
        )
    
    # 添加经络穴位和标签
    for mer_name, mer_dict in MERIDIANS.items():
        color = MERIDIAN_COLOR_MAP.get(mer_name, "red")
        for name, idx in mer_dict.items():
            if idx >= len(vertices):
                continue
            center = vertices[idx]
            sphere = pv.Sphere(radius=0.004, center=center)
            plotter.add_mesh(sphere, color=color, smooth_shading=True)
            plotter.add_point_labels(
                [center], [name],
                font_size=10, text_color='black',
                shadow=False
            )
    
    # 添加经络连线
    for mer_name, path in meridian_paths.items():
        if path is not None:
            color = MERIDIAN_COLOR_MAP.get(mer_name, "red")
            plotter.add_mesh(path, color=color, line_width=3)
    
    # 渲染并保存
    plotter.camera.zoom(1.2)
    plotter.enable_depth_peeling(number_of_peels=100)
    plotter.show(auto_close=False)
    plotter.screenshot(output_path)
    plotter.close()
    
    print(f"✅ 穴位经络可视化保存完成: {output_path}")


def fit_single_frame(img,
                     keypoints,
                     body_model,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     # NEW: MobileNetV3 initialization parameters
                     use_mobilenet_init=False,
                     mobilenet_ckpt='',
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
                                     dtype=dtype)
    else:
        body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    # Initialize MobileNetV3 network if requested
    mobilenet_init_net = None
    if use_mobilenet_init and mobilenet_ckpt:
        if HAS_MOBILENET:
            print(f"[MobileNetV3] Initializing SMPLInitNet from: {mobilenet_ckpt}")
            mobilenet_init_net = SMPLInitNet(ckpt_path=mobilenet_ckpt, device=device)
            mobilenet_init_net.eval()
            print("[MobileNetV3] Network loaded successfully")
        else:
            print("Warning: MobileNetV3 requested but not available, using default initialization")

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    edge_indices = kwargs.get('body_tri_idxs')
    init_t = fitting.guess_init(body_model, gt_joints, edge_indices,
                                use_vposer=use_vposer, vposer=vposer,
                                pose_embedding=pose_embedding,
                                model_type=kwargs.get('model_type', 'smpl'),
                                focal_length=focal_length, dtype=dtype)

    camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=init_t,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      dtype=dtype).to(device=device)
    camera_loss.trans_estimation[:] = init_t

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

        img = torch.tensor(img, dtype=dtype)

        H, W, _ = img.shape

        data_weight = 1000 / H
        # The closure passed to the optimizer
        camera_loss.reset_loss_weights({'data_weight': data_weight})

        # Reset the parameters to estimate the initial translation of the
        # body model
        # Use MobileNetV3 predictions if available, otherwise use default
        if mobilenet_init_net is not None:
            # Preprocess image for MobileNetV3
            img_mb = img.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            img_mb = torch.nn.functional.interpolate(img_mb, size=(224, 224), mode='bilinear', align_corners=False)
            img_mb = (img_mb / 255.0) * 2.0 - 1.0  # Normalize to [-1, 1]
            
            # Move to same device as mobilenet_init_net
            img_mb = img_mb.to(device=next(mobilenet_init_net.parameters()).device)

            with torch.no_grad():
                pred_betas, pred_body_pose, pred_transl = mobilenet_init_net(img_mb)
                
                # Pad betas to match body_model num_betas (MobileNet outputs 10, SMPL-X uses 16)
                if pred_betas.shape[1] < body_model.num_betas:
                    pad_size = body_model.num_betas - pred_betas.shape[1]
                    pred_betas = torch.cat([pred_betas, torch.zeros(pred_betas.shape[0], pad_size, device=pred_betas.device)], dim=1)
                
                # Move predictions to main device if needed
                pred_betas = pred_betas.to(device=device)
                pred_body_pose = pred_body_pose.to(device=device)

            print(f"[MobileNetV3] Using predicted initialization:")
            print(f"  betas mean: {pred_betas.mean().item():.4f}")
            print(f"  body_pose norm: {torch.norm(pred_body_pose).item():.4f}")
            print(f"  betas shape: {pred_betas.shape}")

            body_model.reset_params(betas=pred_betas, body_pose=pred_body_pose)
        else:
            body_model.reset_params(body_pose=body_mean_pose)

        # If the distance between the 2D shoulders is smaller than a
        # predefined threshold then try 2 fits, the initial one and a 180
        # degree rotation
        shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
                                   gt_joints[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < side_view_thsh

        # Update the value of the translation of the camera as well as
        # the image center.
        with torch.no_grad():
            camera.translation[:] = init_t.view_as(camera.translation)
            camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

        # Re-enable gradient calculation for the camera translation
        camera.translation.requires_grad = True

        camera_opt_params = [camera.translation, body_model.global_orient]

        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params,
            **kwargs)

        # The closure passed to the optimizer
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer, body_model, camera, gt_joints,
            camera_loss, create_graph=camera_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            pose_embedding=pose_embedding,
            return_full_pose=False, return_verts=False)

        # Step 1: Optimize over the torso joints the camera translation
        # Initialize the computational graph by feeding the initial translation
        # of the camera and the initial pose of the body model.
        camera_init_start = time.time()
        cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=use_vposer,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            tqdm.write('Camera initialization done after {:.4f}'.format(
                time.time() - camera_init_start))
            tqdm.write('Camera initialization final loss {:.4f}'.format(
                cam_init_loss_val))

        # If the 2D detections/positions of the shoulder joints are too
        # close the rotate the body by 180 degrees and also fit to that
        # orientation
        if try_both_orient:
            body_orient = body_model.global_orient.detach().cpu().numpy()
            flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
            flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

            flipped_orient = torch.tensor(flipped_orient,
                                          dtype=dtype,
                                          device=device).unsqueeze(dim=0)
            orientations = [body_orient, flipped_orient]
        else:
            orientations = [body_model.global_orient.detach().cpu().numpy()]

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            new_params = defaultdict(global_orient=orient)
            if mobilenet_init_net is None:
                new_params['body_pose'] = body_mean_pose
            body_model.reset_params(**new_params)
            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):

                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight
                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 25:67] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 67:] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
            if use_vposer:
                result['body_pose'] = pose_embedding.detach().cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})

        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)

    if save_meshes or visualize:
        body_pose = vposer.decode(
            pose_embedding,
            output_type='aa').view(1, -1) if use_vposer else None

        model_type = kwargs.get('model_type', 'smpl')
        append_wrists = model_type == 'smpl' and use_vposer
        if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

        model_output = body_model(return_verts=True, body_pose=body_pose)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        import trimesh

        out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh.export(mesh_fn)

    if visualize:
        import pyrender

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_center = camera.center.detach().cpu().numpy().squeeze()
        camera_transl = camera.translation.detach().cpu().numpy().squeeze()
        # Equivalent to 180 degrees around the y-axis. Transforms the fit to
        # OpenGL compatible coordinate system.
        camera_transl[0] *= -1.0

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_transl

        camera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center[0], cy=camera_center[1])
        scene.add(camera, pose=camera_pose)

        # Get the lights from the viewer
        light_nodes = monitor.mv.viewer._create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        r = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H,
                                       point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        input_img = img.detach().cpu().numpy()
        output_img = (color[:, :, :-1] * valid_mask +
                      (1 - valid_mask) * input_img)

        img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        img.save(out_img_fn)
        
        # 穴位和经络可视化（使用PyVista）
        visualize_acupoints = kwargs.get('visualize_acupoints', False)
        if visualize_acupoints and HAS_PYVISTA:
            print("[Acupoints] Starting 3D visualization with acupoints and meridians...")
            acupoint_output_path = kwargs.get('acupoint_output_path', 'acupoint_visualization.png')
            visualize_acupoints_and_meridians(
                vertices=vertices,
                faces=body_model.faces,
                output_path=acupoint_output_path
            )
