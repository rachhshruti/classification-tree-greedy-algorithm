# Greedy Algorithm for Classification Tree

Implemented a greedy algorithm for building classification tree given a data set. It uses gini or information gain as a spliting criteria to decide the best attribute based on the user input.
The algorithm gives an average accuracy of 0.95 which was evaluated by performing 10 fold cross validation on 10 different data sets.

# Running the code:

	python ClassficationTree.py "dataset" "," "gini" 
	
The code accepts the following command line arguments:

1. The path of the dataset file
2. The delimiter used in the file to separate attributes
3. A string either "gini" or "info" for deciding the best spliting attribute.