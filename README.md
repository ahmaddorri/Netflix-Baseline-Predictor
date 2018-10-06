Most of Netflix recommender systems use data frame in this format:
`(user_id, item_id, rating)`
but the main dataset of Netflix prize is not in this style so in `PrepareDataframe.py` I attempt to convert the main dataset to `(user_id, item_id, rating)`
After this, you can use various type of module like [Surprise](http://surpriselib.com/) for SVD, SVD++, KNN,...

But in this project i try to impelemnt baseline predictor of BellKor team (https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf) that try to minimize :

![alt text](https://user-images.githubusercontent.com/19234324/46571650-81c7fa00-c985-11e8-837e-ebe128ab5b14.PNG)
