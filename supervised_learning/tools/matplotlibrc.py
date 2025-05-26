#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局 Matplotlib 配置
用于支持中文显示和美化默认样式
"""

import matplotlib.pyplot as plt
import platform
import matplotlib

# 设置中文字体
system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 优先使用微软雅黑，备选黑体
elif system == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB']  # 苹方、冬青黑体
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 文泉驿微米黑

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 设置默认颜色循环为护眼色系
plt.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=[
    '#61a0a8', '#d48265', '#91c7ae', '#749f83', '#ca8622',
    '#bda29a', '#6e7074', '#546570', '#c4ccd3'
])

# 设置全局样式
plt.style.use('seaborn-v0_8-darkgrid')
