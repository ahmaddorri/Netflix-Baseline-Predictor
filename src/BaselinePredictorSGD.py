import time
import pickle
import numpy as np
p_number = 1
alpha =25
mu=3.6


def calculate_loss(bu,bi,user_,movie_,df_example):

    return sum(np.power((df_example["Rating"]-bu[df_example["Cust_Id"]]-mu-bi[df_example["Movie_Id"]]),2))


def f(bu,bi,user_,movie_,df_example):
    bu = np.array(bu)
    bi = np.array(bi)
    res = calculate_loss(bu =bu,bi=bi,user_=user_,movie_=movie_,df_example=df_example)
    s = res
    a =0
    for b in bu:
        a += b**2

    for b in bi:
        a += b**2
    a = a*alpha
    s += a
    return s


def derivative_bu(bu, bi, sample_df, movie_, i):
    """
    partial derivative with respect to bu 
    """
    sample_df = sample_df[sample_df["Cust_Id"] == i]
    s =0
    if not sample_df.empty:
        s = np.sum(np.dot(-2, sample_df["Rating"] - bu[i] - mu - bi[movie_[sample_df.index]])) + (2 * alpha * bu[i])
    return s


def derivative_bi(bu, bi, sample_df, user_, i):
    """
    partial derivative with respect to bi 
    """
    sample_df = sample_df[sample_df["Movie_Id"] == i]
    s = 0
    if not sample_df.empty:
        s = np.sum(np.dot(-2, sample_df["Rating"] - bu[user_[sample_df.index]] - mu - bi[i])) + (2 * alpha * bi[i])
    return s


def SGD(df_example,user_dict,movie_dict,initial_guess):
    print("SGD start")
    lr = 0.0001
    bu = initial_guess[0:df_example.Cust_Id.unique().size]
    bi = initial_guess[df_example.Cust_Id.unique().size:df_example.Cust_Id.unique().size + df_example.Movie_Id.unique().size]
    bu = np.array(bu)
    bi = np.array(bi)

    iteration = 100
    sample_size = 1000000
    vectorize_user_dict = np.vectorize(user_dict.get)
    vectorize_movie_dict = np.vectorize(movie_dict.get)

    user_ = vectorize_user_dict(df_example["Cust_Id"])
    movie_ = vectorize_movie_dict(df_example["Movie_Id"])

    del vectorize_movie_dict,vectorize_user_dict

    df_example["Cust_Id"] = user_
    df_example["Movie_Id"] = movie_
    print(df_example)
    for itr in range(1, iteration):
        done = False
        copy_df = df_example.copy()
        while done == False:
            tic_toc = time.time()
            if copy_df.shape[0] <= sample_size:
                sample_df = copy_df
                done = True
            else:
                sample_df = copy_df.sample(sample_size, replace=False)
                copy_df = copy_df.drop(sample_df.index)
            # print(sample_df)
            # print("-------------------")


            for i in range(0,len(bu)):
                bu[i] -= lr * derivative_bu(bu, bi, sample_df, movie_, i)
            for i in range(0,len(bi)):
                bi[i] -= lr * derivative_bi(bu, bi, sample_df, user_, i)


            print("loss:",f(bu, bi,user_,movie_,df_example))
            print("time:",time.time()-tic_toc)

        del copy_df

        with open("bu.pickle", 'wb') as output:
            pickle.dump(bu, output, pickle.HIGHEST_PROTOCOL)

        with open("bi.pickle", 'wb') as output:
            pickle.dump(bi, output, pickle.HIGHEST_PROTOCOL)

        with open("user_dict.pickle", 'wb') as output:
            pickle.dump(user_dict, output, pickle.HIGHEST_PROTOCOL)

        with open("movie_dict.pickle", 'wb') as output:
            pickle.dump(movie_dict, output, pickle.HIGHEST_PROTOCOL)
        print("saved\n")
        print("epch number %d finished" % itr)

    return bu,bi


def main():
    import numpy as np
    import pickle
    ########################################################
    df = pickle.load(open("../data/netflix_dataframe.pickle","rb"))
    print(df.head())
    df = df.reset_index()
    print('-Dataset examples-')
    # df_example = df.iloc[::5000, :]
    df_example = df
    print(df_example)

    #####################################

    mu = df_example.Rating.mean()
    print(mu)

    #################################

    user_dict = dict(zip(df_example.Cust_Id.unique(), range(0,df_example.Cust_Id.unique().size)))

    #################################

    movie_dict = dict(zip(df_example.Movie_Id.unique(), range(0,df_example.Movie_Id.unique().size)))

    #################################
    print("dictionary fixed")
    b_initial = np.zeros(df_example.Cust_Id.unique().size+df_example.Movie_Id.unique().size).tolist()
    initial_guess = b_initial
    t0 = time.time()
    bu,bi = SGD(df_example,user_dict,movie_dict,initial_guess)
    t1 = time.time()
    total = t1-t0
    print("time:  ",total)
    print(bu)
    print(bi)
    ####------###################



    with open("bu.pickle", 'wb') as output:
        pickle.dump(bu, output, pickle.HIGHEST_PROTOCOL)

    with open("bi.pickle", 'wb') as output:
        pickle.dump(bi, output, pickle.HIGHEST_PROTOCOL)

    with open("user_dict.pickle", 'wb') as output:
        pickle.dump(user_dict, output, pickle.HIGHEST_PROTOCOL)

    with open("movie_dict.pickle", 'wb') as output:
        pickle.dump(movie_dict, output, pickle.HIGHEST_PROTOCOL)
    print("saved\n")



if __name__=="__main__":
    main()

