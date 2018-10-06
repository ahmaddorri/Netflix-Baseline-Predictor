Most of Netflix recommender systems use dataframe in this format:
`(user_id, item_id, rating)`
but the main dataset of Netflix prize is not in this style so in `PrepareDataframe.py` I attempt to convert the main dataset to   
`(user_id, item_id, rating)`  
After this, you can use various type of module like [Surprise](http://surpriselib.com/), [lightfm](https://github.com/lyst/lightfm) for SVD, SVD++, KNN,...

But in this project i try to impelemnt baseline predictor of BellKor team (https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf) that try to minimize :

<p align="left">
  <img src="https://user-images.githubusercontent.com/19234324/46571650-81c7fa00-c985-11e8-837e-ebe128ab5b14.PNG" width="400"/>
</p>

I use fully vectorize approach by getting help from `numpy` to get the best performance.  

When I compute partial derivative with respect to inputs(bu, bi), you can notice some vectorize techniques.
 
 ```python
def derivative_bu(bu, bi, sample_df, movie_, i):
    """
    partial derivative with respect to bu 
    """
    sample_df = sample_df[sample_df["Cust_Id"] == i]
    s =0
    if not sample_df.empty:
        s = np.sum(np.dot(-2, sample_df["Rating"] - bu[i] - mu - bi[movie_[sample_df.index]])) + (2 * alpha * bu[i])
    return s
```

```python
def derivative_bi(bu, bi, sample_df, user_, i):
    """
    partial derivative with respect to bi 
    """
    sample_df = sample_df[sample_df["Movie_Id"] == i]
    s = 0
    if not sample_df.empty:
        s = np.sum(np.dot(-2, sample_df["Rating"] - bu[user_[sample_df.index]] - mu - bi[i])) + (2 * alpha * bi[i])
    return s
```
