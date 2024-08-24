from sklearn.datasets import load_iris
import numpy as np

class TreeNode:
    def __inti__(self,feature=None,left=None,right=None):
        pass

class DecisionTree:
    def __inti__(self,max_depth=0,criterium='entropy'):
        self.max_depth = max_depth
        self.criterium = criterium

    def information_gain(self,y,y_left,y_right,criterium='entropy'):
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

    def split(self,X,y,feature,decision):
        left_index = np.where(X[:,feature] <= decision)
        right_index = np.where(X[:,feature] > decision)
        return X[left_index],X[right_index],y[left_index],y[right_index]

    def best_split(self,X,y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:,feature])
            for threshold in thresholds:
                X_left,X_right,y_left,y_right = self.split(X,y,feature,threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gain = self.information_gain(y,y_left,y_right,self.criterium)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def generate_tree(self,X,y,max_depth=10):
        # TODO
        pass
    
if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target

    print(np.shape(X))
    print(np.shape(y))
