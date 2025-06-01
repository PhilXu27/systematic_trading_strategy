from algo.s0_data_preparing import s0_data_prepare
from algo.s1_feature_engineering import s1_feature_engineering
from algo.s2_creating_labels import s2_creating_labels


def main():
    train_start = "2018-01-01"
    train_end = "2020-12-31"
    test_start = "2021-01-01"
    test_end = "2021-12-31"

    #####################
    # 0. Data Preparing #
    #####################
    prices = s0_data_prepare()
    prices = prices.loc[train_start: test_end]
    ##########################
    # 1. Feature Engineering #
    ##########################
    features = s1_feature_engineering(prices)

    #######################
    # 2.  Creating Labels #
    #######################

    s2_creating_labels(prices)

    return


if __name__ == '__main__':
    main()
