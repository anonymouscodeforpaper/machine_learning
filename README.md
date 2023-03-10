# Machine_learning
This is the implementation of our paper titled "Revisiting matrix factorization: from an additive and argumentative point of view"  

## Example of A<sup>3</sup>R

![A toy example in book recommendation showing how A<sup>3</sup>R works. The thickness of the arrow reflects the importance of attribute type. In this toy example, the user accords more importance to the genre of the book.](https://github.com/anonymouscodeforpaper/machine_learning/blob/master/framework_dsaa.png)

A toy example in book recommendation showing how A<sup>3</sup>R works. The thickness of the arrow reflects the importance of attribute type. In this toy example, the user accords more importance to the genre of the book.

## Major steps of A<sup>3</sup>R

![The major steps of A<sup>3</sup>R](https://github.com/anonymouscodeforpaper/machine_learning/blob/master/zhuzhu.png)

## Strusture of code
-- In data_loader.py, we preprocess the dataset and returns the training set, testset

-- utilities.py contains the function necessary for data processing and model training


To run the code, python3 main.py--name = 'dbbook2014', this is to run on the Dbook2014 dataset.


## Datasets:

-- The source for DBook 2014: https://2014.eswc-conferences.org/important-dates/call-RecSys.html to get the types of attributes (aka. authors and genres) of items, we crawl the DBpedia links provided in the original dataset. For the books whose information is missing, we manually complete them according to Goodreads (https://www.goodreads.com/)

-- The source for Netflix, MovieLens Development (Dev.) and the MovieLens 100K, we reuse the version provided in A. Rago, O. Cocarascu, C. Bechlivanidis, D. Lagnado, and F. Toni, “Argumentative explanations for interactive recommendations,” Artificial Intelligence, vol. 296, p. 103506, 2021.

## Other files
MovieLens100k,MovieLenssmall,NETFLIX, dbook2014 contains the the parameters of the models, the training and test, and the original datasets. In the NETFLIX, brand_new_netflix.ipynb contains codes for generating the cluster. Note that, in order to compare the A-I model proposed by Rago et al, we have adapted the codes proposed by the authors. (Argumenta-
tion as a framework for interactive explanations for recommendations.
Proceedings of the international conference on principles of knowledge
representation and reasoning (Vol. 17, pp. 805–815))
