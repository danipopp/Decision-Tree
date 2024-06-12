from sklearn.datasets import load_iris

class TreeNode:
    pass

class DecisionTree:

    def gini_impurty(y):
        pass

    def get_information(actual_impurty,left,right):
        pass

    def split(X,y):
        pass

    def generate_tree(X,y,max_depth=10):
        # TODO
        pass
    
if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target
