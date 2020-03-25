# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

"""
绘决策树的函数
"""

'''定义分支点的样式'''
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
'''定义叶节点的样式'''
leafNode = dict(boxstyle="round4", fc="0.8")
'''定义箭头标识样式'''
arrow_args = dict(arrowstyle="<-")


def get_num_leafs(tree):
    """
    计算树的叶子节点数量
    :param tree: 树
    :return: 叶子节点数
    """
    num_leafs = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == "dict":
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_depth(tree):
    """
    计算树的最大深度
    :param tree: 树
    :return: 树的最大深度
    """
    max_depth = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == "dict":
            this_depth = 1 + get_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def plot_node(node_txt, center_pt, parent_pt, node_type):
    """
    画出节点
    :param node_txt:
    :param center_pt:
    :param parent_pt:
    :param node_type:
    :return:
    """
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                             textcoords='axes fraction', va="center", ha="center", bbox=node_type,
                             arrowprops=arrow_args)


def plot_mid_text(center_pt, parent_pt, txt_string):
    """
    标箭头上的文字
    :param center_pt:
    :param parent_pt:
    :param txt_string:
    :return:
    """
    lens = len(txt_string)
    x_mid = (parent_pt[0] + center_pt[0]) / 2.0 - lens * 0.002
    y_mid = (parent_pt[1] + center_pt[1]) / 2.0
    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_tree(tree, parent_pt, node_txt):
    """
    :param tree:
    :param parent_pt:
    :param node_txt:
    :return:
    """
    num_leafs = get_num_leafs(tree)
    depth = get_depth(tree)
    first_str = list(tree.keys())[0]
    center_pt = (plot_tree.x0ff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.y0ff)
    plot_mid_text(center_pt, parent_pt, node_txt)
    plot_node(first_str, center_pt, parent_pt, decisionNode)
    second_dict = tree[first_str]
    plot_tree.y0ff = plot_tree.y0ff - 1.0 / plot_tree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], center_pt, str(key))
        else:
            plot_tree.x0ff = plot_tree.x0ff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.x0ff, plot_tree.y0ff), center_pt, leafNode)
            plot_mid_text((plot_tree.x0ff, plot_tree.y0ff), center_pt, str(key))
    plot_tree.y0ff = plot_tree.y0ff + 1.0 / plot_tree.totalD


def create_plot(tree):
    """
    :param tree:
    :return:
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    apropos = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **apropos)
    plot_tree.totalW = float(get_num_leafs(tree))
    plot_tree.totalD = float(get_depth(tree))
    plot_tree.x0ff = -0.5 / plot_tree.totalW
    plot_tree.y0ff = 1.0
    plot_tree(tree, (0.5, 1.0), '')
    plt.show()
