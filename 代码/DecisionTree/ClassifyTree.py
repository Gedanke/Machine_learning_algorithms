# -*- coding: utf-8 -*-

import treePlotter
import csv
import math
import operator


def is_number(num):
    """
    :param num: 传入字符串
    :return: 判断字符串是否为一个数(整数,浮点数)
    """
    try:
        float(num)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(num)
        return True
    except (TypeError, ValueError):
        pass
    return False


class ClassifyTree(object):
    def __init__(self, path_train, path_test):
        """
        :param path_train: 训练数据集路径
        :param path_test: 测试数据集路径
        """
        self.train_data = list(csv.reader(open(path_train, "r")))
        self.attributes = self.train_data.pop(0)
        self.attributes.pop(-1)
        self.test_data = list(csv.reader(open(path_test, "r")))
        self.test_data.pop(0)
        self.test_label = list()
        self.result_label = list()
        self.init_data()

    def init_data(self):
        """
        初始化数据
        :return:
        """
        '''训练集'''
        for i in range(len(self.train_data)):
            for j in range(len(self.train_data[i])):
                '''数字'''
                if is_number(self.train_data[i][j]):
                    self.train_data[i][j] = float(self.train_data[i][j])
        '''测试集'''
        for i in range(len(self.test_data)):
            self.test_label.append(self.test_data[i].pop(-1))
            for j in range(len(self.test_data[i])):
                if is_number(self.test_data[i][j]):
                    self.test_data[i][j] = float(self.test_data[i][j])

    def gain_entropy(self, data_set):
        """
        计算给定数据集的香农熵
        熵越大,数据集的混乱程度越大
        :param data_set: 数据集
        :return: 数据集的香农熵
        """
        num_entropy = len(data_set)
        label_counts = dict()
        for feat_vec in data_set:
            current_label = feat_vec[-1]
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0
            label_counts[current_label] += 1
        entropy = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / num_entropy
            entropy -= prob * math.log(prob, 2)
        return entropy

    def split_data_set(self, data_set, axis, value):
        """
        按照给定特征划分数据集,去除选择维度中等于选择值的项
        :param data_set: 数据集
        :param axis: 选择维度
        :param value: 选择值
        :return: 划分数据集
        """
        ret_data_set = list()
        for feat_vec in data_set:
            if feat_vec[axis] == value:
                reduce_feat_vec = feat_vec[:axis]
                reduce_feat_vec.extend(feat_vec[axis + 1:])
                ret_data_set.append(reduce_feat_vec)
        return ret_data_set

    def gain_best_split_feature(self, data_set):
        """
        选择最好的数据集划分维度
        :param data_set: 数据集
        :return: 最好的划分维度
        """
        num_features = len(data_set[0]) - 1
        base_entropy = self.gain_entropy(data_set)
        best_info_ratio = 0.0
        best_feature = -1
        for i in range(num_features):
            feat_list = [example[i] for example in data_set]
            unique_val = set(feat_list)
            new_entropy = 0.0
            split_info = 0.0
            for value in unique_val:
                sub_data_set = self.split_data_set(data_set, i, value)
                prob = len(sub_data_set) / float(len(data_set))
                new_entropy += prob * self.gain_entropy(sub_data_set)
                split_info += -prob * math.log(prob, 2)
            info_gain = base_entropy - new_entropy

            if split_info == 0:
                continue
            info_gain_ratio = info_gain / split_info
            if info_gain_ratio > best_info_ratio:
                best_info_ratio = info_gain_ratio
                best_feature = i
        return best_feature

    def major_count(self, class_list):
        """
        数据集已经处理了所有属性,但是类标签依然不是唯一的
        采用多数判决的方法决定该子节点的分类
        :param class_list: 分类类别列表
        :return: 子节点的分类
        """
        class_count = dict()
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] = 1
        sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sort_class_count[0][0]

    def create_tree(self, data_set, attributes):
        """
        递归构建决策树
        :param data_set: 数据集
        :param attributes: 特征标签
        :return: 决策树
        """
        class_list = [example[-1] for example in data_set]
        if class_list.count(class_list[0]) == len(data_set):
            '''类别完全相同,停止划分'''
            return class_list[0]
        if len(data_set[0]) == 1:
            '''遍历完所有特征时返回出现次数最多的'''
            return self.major_count(class_list)

        best_feature = self.gain_best_split_feature(data_set)
        best_feature_attribute = attributes[best_feature]
        tree = {
            best_feature_attribute: {}
        }
        del (attributes[best_feature])
        '''得到列表包括节点所有的属性值'''
        feat_val = [example[best_feature] for example in data_set]
        unique_val = set(feat_val)
        for value in unique_val:
            sub_attributes = attributes[:]
            tree[best_feature_attribute][value] = self.create_tree(self.split_data_set(data_set, best_feature, value),
                                                                   sub_attributes)
        return tree

    def classify(self, tree, feat_attributes, test_vec):
        """
        跑决策树
        :param tree: 决策树
        :param feat_attributes: 分类标签
        :param test_vec: 测试数据
        :return: 决策结果
        """
        first_str = list(tree.keys())[0]
        second_dict = tree[first_str]
        feat_index = feat_attributes.index(first_str)
        class_label = None
        for key in second_dict.keys():
            if test_vec[feat_index] == key:
                if type(second_dict[key]).__name__ == "dict":
                    class_label = self.classify(second_dict[key], feat_attributes, test_vec)
                else:
                    class_label = second_dict[key]
        return class_label

    def classify_all(self, tree):
        """
        跑决策树
        :param tree: 决策树
        :return: 决策结果
        """
        for test_vec in self.test_data:
            self.result_label.append(self.classify(tree, self.attributes, test_vec))

    def get_result(self):
        """
        得到结果并绘制树
        :return:
        """
        data_set = self.train_data
        labels_tmp = self.attributes[:]
        decision_tree = self.create_tree(data_set, labels_tmp)
        print("decisionTree:\n", decision_tree)
        treePlotter.createPlot(decision_tree)
        print("classifyResult:\n")
        print([label for label in self.result_label])


if __name__ == "__main__":
    path_train_ = "train_.csv"
    path_test_ = "test_.csv"
    tree_ = ClassifyTree(path_train_, path_test_)
    tree_.get_result()
