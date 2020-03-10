# -*- coding: utf-8 -*-

import math
import csv


def is_number(num):
    try:  # 如果能运行float(s)语句，返回True(字符串s是浮点数)
        float(num)
        return True
    except ValueError:  # ValueError为Python的一种标准异常,表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常,不做任何事情(pass:不做任何事情，一般用做占位语句)
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(num)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False


class DecisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        """
        :param col: 待检验的判断条件所对应的列索引值
        :param value: 为了使结果为True,当前列必须匹配的值
        :param results: 保存的是针对当前分支的结果,字典类型
        :param tb: DecisionNode,对应于结果为true时,树上相对于当前节点的子树上的节点
        :param fb: DecisionNode,对应于结果为false时,树上相对于当前节点的子树上的节点
        """
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


class DecisionTree(object):
    def __init__(self, path_train, path_test):
        """
        :param path_train: 训练数据集路径
        :param path_test: 测试数据集路径
        列表中的数据均为str类型
        使用is_number可以判断一个字符串是否为数值型
        """
        '''训练集'''
        self.train_data = list(csv.reader(open(path_train, "r", encoding="UTF-8-sig")))
        '''去除第一行(列名)'''
        self.train_data.pop(0)
        '''测试集'''
        self.test_data = list(csv.reader(open(path_test, "r", encoding="UTF-8-sig")))
        '''去除第一行(列名)'''
        self.test_data.pop(0)
        '''构成出的树'''
        self.tree = None
        '''测试集行数'''
        self.rows = len(self.train_data)
        '''样本特征个数'''
        self.lists = len(self.train_data[0])
        '''测试集标签集'''
        self.test_label = list()
        '''预测得到的标签集'''
        self.result_label = list()
        self.init_data()

    def init_data(self):
        """
        :return:
        """
        '''数值化为数值的字符串'''
        '''训练集'''
        for i in range(len(self.train_data)):
            for j in range(len(self.train_data[i])):
                '''数字'''
                if is_number(self.train_data[i][j]):
                    self.train_data[i][j] = float(self.train_data[i][j])
        '''测试集'''
        for i in range(len(self.test_data)):
            for j in range(len(self.test_data[i])):
                if is_number(self.test_data[i][j]):
                    self.test_data[i][j] = float(self.test_data[i][j])
        '''拆分测试集'''
        for line in self.test_data:
            self.test_label.append(line.pop(self.lists - 1))

    def unique_counts(self, data):
        """
        :param data:
        :return:
        """
        results = dict()
        for row in data:
            r = row[len(row) - 1]
            if r not in results:
                results[r] = 0
            results[r] += 1
        return results

    def entropy(self, data):
        """
        :param data:
        :return:
        """
        ent = 0.0
        results = self.unique_counts(data)
        for r in results.keys():
            p = float(results[r]) / len(data)
            ent = ent - p * math.log2(p)
        return ent

    def divide_set(self, data, column, value):
        """
        :param data:
        :param column:
        :param value:
        :return: 数据集被拆分成的两个集合
        """
        '''数值型(含浮点数和整数型),str类型'''
        if is_number(value):
            def split_function(row):
                return row[column] >= value
        else:
            def split_function(row):
                return row[column] == value
        '''将数据集拆分成两个集合,并返回'''
        set1 = [row for row in data if split_function(row)]
        set2 = [row for row in data if not split_function(row)]
        return set1, set2

    def build_tree(self, data, function):
        """
        :param data:
        :param function:
        :return:
        """
        if len(data) == 0:
            return DecisionNode()
        current_score = function(data)

        '''定义一些变量以记录最佳拆分条件'''
        best_gain = 0.0
        best_criteria = None
        best_sets = None

        column_count = len(data[0]) - 1
        for col in range(0, column_count):
            column_values = dict()
            for row in data:
                column_values[row[col]] = 1
            for value in column_values.keys():
                (set1, set2) = self.divide_set(data, col, value)

                '''信息增益'''
                p = float(len(set1)) / len(data)
                gain = current_score - p * function(set1) - (1 - p) * function(set2)
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)

        '''创建子分支'''
        if best_gain > 0:
            true_branch = self.build_tree(best_sets[0], function)
            false_branch = self.build_tree(best_sets[1], function)
            return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                                tb=true_branch, fb=false_branch)
        else:
            return DecisionNode(results=self.unique_counts(data))

    def print_tree_(self, tree, indent=""):
        """
        :param tree
        :param indent:
        :return:
        """
        '''判断是否是叶子节点'''
        if tree.results is not None:
            print(str(tree.results))
        else:
            '''打印判断条件'''
            print(str(tree.col) + ":" + str(tree.value) + "?")
            '''打印分支'''
            print(indent + "T->", end=" ")
            self.print_tree_(tree.tb, indent + " ")
            print(indent + "F->", end=" ")
            self.print_tree_(tree.fb, indent + " ")

    def print_tree(self):
        """
        可以直接使用上面那个函数
        个人C++的习惯
        :return:
        """
        self.print_tree_(self.tree)

    def classify(self, sample, tree):
        """
        :param sample: 样本(不含标签)
        :param tree: 树
        :return:
        """
        if tree.results is not None:
            return tree.results
        else:
            node_value = sample[tree.col]
            if is_number(node_value):
                if node_value >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if node_value == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classify(sample, branch)

    def gain_impurity(self, data):
        """
        :param data:
        :return:
        基尼不纯度
        随机放置的数据项出现于错误分类中的概率
        """
        total = len(data)
        counts = self.unique_counts(data)
        imp = 0
        for k in counts.keys():
            p1 = float(counts[k]) / total
            imp += p1 * (1 - p1)
        return imp

    def pruning_(self, tree, impurity):
        """
        :param tree:
        :param impurity:
        :return:
        剪枝
        """
        '''如果分支不是叶节点,则对其进行剪枝'''
        if tree.tb.results is None:
            self.pruning_(tree.tb, impurity)
        if tree.fb.results is None:
            self.pruning_(tree.fb, impurity)
        '''如果两个子分支都是叶节点,判断是否能够合并'''
        if tree.tb.results is not None and tree.fb.results is not None:
            '''构造合并后的数据集'''
            tb = list()
            fb = list()
            for v, c in tree.tb.results.items():
                tb += [[v]] * c
            for v, c in tree.fb.results.items():
                fb += [[v]] * c
            '''检查熵的减少量'''
            decrease = self.entropy(tb + fb) - (self.entropy(tb) + self.entropy(fb) / 2)
            if decrease < impurity:
                '''合并分支'''
                tree.tb = None
                tree.fb = None
                tree.results = self.unique_counts(tb + fb)

    def pruning(self, impurity):
        """
        :param impurity:
        :return:
        """
        self.pruning_(self.tree, impurity)

    def get_result(self, args):
        """
        :param args: 参数列表 第一个为函数名
        :return: 正确率
        """
        correct = 0
        self.tree = self.build_tree(self.train_data, args[0])
        '''剪枝'''
        if len(args) > 1:
            self.pruning(args[1])
        len_test_data = len(self.test_data)
        for index in range(len_test_data):
            sample_label = self.classify(self.test_data[index], self.tree)
            sample_label = list(sample_label)[0]
            self.result_label.append(sample_label)
            if sample_label == self.test_label[index]:
                correct += 1
        return correct / len_test_data


if __name__ == "__main__":
    path_train_ = "train.csv"
    path_test_ = "test.csv"
    decisionTree = DecisionTree(path_train_, path_test_)
    '''指定参数'''
    result = decisionTree.get_result([decisionTree.entropy])
    '''打印树'''
    decisionTree.print_tree()
    print("\n" + str(result) + "\n")
    '''剪枝'''
    # 剪枝效果不佳,待改进
    result_ = decisionTree.get_result([decisionTree.gain_impurity, 0.1])
    decisionTree.print_tree()
