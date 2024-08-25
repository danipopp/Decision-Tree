from sklearn.datasets import load_iris
import numpy as np

class TreeNode:
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self,max_depth=0,criterium='entropy'):
        self.max_depth = max_depth
        self.criterium = criterium
        self.root = None

    def information_gain(self, y, y_left, y_right, criterium='entropy'):
        def entropy(y):
            entropy = 0
            if len(y) == 0:
                return 0
            for i in np.unique(y):
                p = np.sum(y == i) / len(y)  # proportion of one class
                entropy += -p * np.log2(p)   # entropy for the class
            
            return entropy
    
        def gini_impurity(y):
            gini = 0
            for i in np.unique(y):
                p = np.sum(y == i) / len(y)
                gini += p ** 2
            return 1.0 - gini
    
        m = len(y)
        if criterium == 'entropy':
            information_gain = entropy(y) - (len(y_left) / m * entropy(y_left) + len(y_right) / m * entropy(y_right))
        elif criterium == 'gini':
            information_gain = gini_impurity(y) - (len(y_left) / m * gini_impurity(y_left) + len(y_right) / m * gini_impurity(y_right))
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

    def generate_tree(self,X,y,depth=0):
        num_samples, num_feature = X.shape
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels == 1 or num_samples < 2:
            leaf_value = self.most_common_value(y)
            return TreeNode(value = leaf_value)
        
        feature, threshold = self.best_split(X,y)
        if feature is None:
            return TreeNode(value = self.most_common_value(y))
        
        X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
        left_child = self.generate_tree(X_left, y_left, depth + 1)
        right_child = self.generate_tree(X_right, y_right, depth + 1)
        
        return TreeNode(feature = feature, threshold = threshold, left= left_child, right = right_child)

    def most_common_value(self,y):
        return np.bincount(y).argmax()

    def fit(self,X,y):
        self.root = self.generate_tree(X,y)

    def predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)
    
    def predict(self, X):
        return [self.predict_one(x, self.root) for x in X]
    
if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target

    tree = DecisionTree(max_depth=3,criterium='entropy')
    tree.fit(X, y)
    predictions = tree.predict(X)

    accuracy = np.sum(predictions == y) / len(y)
    print(f'Accuracy: {accuracy * 100:.2f}%')
