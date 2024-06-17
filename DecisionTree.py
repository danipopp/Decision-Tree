from sklearn.datasets import load_iris
import numpy as np

class TreeNode:
    def __inti__(self,feature=None,left=None,right=None):
        pass

class DecisionTree:
    def __inti__(self,max_depth=0,criterium='entropy'):
        self.max_depth = max_depth
        self.criterium = criterium

    def information_gain(y,y_left,y_right,criterium='entropy'):
        def entropy(y):
            entropy = 0
            if len(y) == 0:
                return 0
            for i in np.unique(y):
                p = np.sum(y==i)/len(y) # proportion of one class
                entropy += -p * np.log2(p) - (1-p) * np.log2(1-p) 

            return 
        def gini_impurty(y):
            gini = 0
            for i in np.unique(y):
                gini = -(np.sum(y==i)/len(y))**2
            return 1.0 - gini
        
        information_gain = 0
        m = len(y)
        if criterium == 'entropy':
            information_gain = entropy(y) - (len(y_left)/m * entropy(y_left) + len(y_right)/ m * entropy(y_right))
        elif criterium == 'gini':
            information_gain = gini_impurty(y) - (len(y_left) / m * gini_impurty(y_left) + len(y_right) / m * gini_impurty(y_right))
        return information_gain


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

    print(np.shape(X))
    print(np.shape(y))
