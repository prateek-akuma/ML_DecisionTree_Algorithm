import sys
import math
import pandas as pd

class DecisionNode:
    # A DecisionNode contains an attribute and a dictionary of children. 
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level = 0):
        if self.children == {}: # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)
     
    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}: # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)

# Illustration of functionality of DecisionNode class
def funTree(node,data,prev,target,attributes,proot):
    #proot=previous_root(parent)
    #prev=subtree_value
    if node==None:#for initializing root node
        temp=information_gain(data,target,attributes)
        newnode=DecisionNode(temp)
        node=newnode
        return funTree(node,data,None,target,attributes,temp)
    
    if len(data[proot].unique())>1:#if root node has more root children
        for i in data[proot].unique():
            node.children[i]=funTree(node,data[data[proot]==i],i,target,attributes,proot)
                     
    if prev!=None:#for initializing children
        if len(data[target].unique())==1:
            dummy=data[target].unique()
            node.children[prev]=DecisionNode(dummy[0]) 
            return node.children[prev]
        if len(attributes)<=1:#if there are no attributes,printing mode of target
            freq=data[target].mode()
            return DecisionNode(freq[0]) 
        
        if proot in attributes:
            attributes.remove(proot)
        temp2=information_gain(data,target,attributes)
        node.children[prev]=DecisionNode(temp2) 
        return funTree(node.children[prev],data,None,target,attributes,temp2)  
    
    return node


def entropy(data,target): #calculating entropy for all attributes
    target_att=data[target]
    yes=0
    no=0
    uni_el=data[target].unique().tolist()
    count=[0]*len(uni_el)
   
    
    for i in range(target_att.shape[0]):
        
        
        for j in range(len(uni_el)):
            if (target_att.iloc[i]==uni_el[j]):
                count[j]+=1
    totalsum=0
    for i in count:
        h=-((i/sum(count))*(math.log(i/sum(count)))/math.log(2))
        totalsum=totalsum+h
    return totalsum
    
def information_gain(data,target,attributes):#calculating information gain for all attributes
    s=entropy(data,target)
    info_gain=[0]*len(attributes)
    for i in range(len(attributes)):
        x=data[attributes[i]].unique().tolist()
        each_feature_entro=[0]*len(x)
        each_feature_count=[0]*len(x)
        for j in range(len(x)):
            temp=data[data[attributes[i]]==x[j]]
            each_feature_count[j]=temp.shape[0]
            each_feature_entro[j]=entropy(temp,target)
        total_g=0
        for j in range(len(x)):
            g=(each_feature_count[j]/sum(each_feature_count))*each_feature_entro[j]
            total_g=total_g+g
        info_gain[i]=s-total_g
    print(f'attribute with i entropy is:   {attributes[info_gain.index(max(info_gain))]}  ')
    return attributes[info_gain.index(max(info_gain))]
    
def id3(examples, target, attributes):
    total_entropy=0
    tree=funTree(None,examples,None,target,attributes,0)
    
    return tree

####################   MAIN PROGRAM ######################

# Reading input data
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
target = sys.argv[3]
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train,target,attributes)
tree.display()

# Evaluating the tree on the test data
correct = 0
for i in range(0,len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i,target]):
        correct += 1
print("\nThe accuracy is: ", correct/len(test))