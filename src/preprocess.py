import pandas as pd


def parse_data(train_path_file='data_src/agoda_cancellation_train.csv', test_set_path=None):
    train_data = pd.read_csv(train_path_file, low_memory=False, encoding='utf-8')
    train_data, target, h_booking_id = process_data(train_data, with_label=True)
    train_data.to_csv('output.csv', index=False)
    test_data = pd.read_csv(test_set_path, low_memory=False, encoding='utf-8')
    test_features, _ = process_data(test_data)
    train_data.to_csv('output.csv', index=False)
    return train_data, target, test_features


def process_data(data: pd.DataFrame, with_label=False):
    # remove dup
    target = None
    if data.duplicated().any():
        data.drop_duplicates(inplace=False)
    h_booking_id = data['h_booking_id']
    data.drop('h_booking_id', axis=1)

    check_in = pd.to_datetime(data['checkin_date']).dt.dayofyear + (
            pd.to_datetime(data['checkin_date']).dt.year * 365)
    booking_time = pd.to_datetime(data['booking_datetime']).dt.dayofyear + (
            pd.to_datetime(data['booking_datetime']).dt.year * 365)
    # new reservation_wait_time column
    data['time_until_checking_in_days'] = check_in - booking_time
    data['booking_datetime'] = pd.to_datetime(data['booking_datetime']).dt.month
    data['checkin_date_month'] = pd.to_datetime(data['checkin_date']).dt.month
    data['travel_days_long'] = (pd.to_datetime(data['checkout_date']).dt.dayofyear + (
            pd.to_datetime(data['checkout_date']).dt.year * 365)) - (
                                       pd.to_datetime(data['checkin_date']).dt.dayofyear + (
                                       pd.to_datetime(data['checkin_date']).dt.year * 365))
    if with_label:
        data['is_cancelled'] = (pd.to_datetime(data['cancellation_datetime']).dt.dayofyear).apply(
            lambda x: 1 if x > 0 else 0)
        target = data['is_cancelled']
        data = data.drop(["cancellation_datetime", "is_cancelled"], axis=1)

    encoded_data_accommadation_type_name = pd.get_dummies(data['accommadation_type_name'], prefix='accommadation_type')
    data = pd.concat([data, encoded_data_accommadation_type_name], axis=1)
    dummies = pd.get_dummies(data['charge_option'], prefix='charge_option')
    data = pd.concat([data, dummies], axis=1)
    # clean data
    data = data.fillna(0)
    # additional features
    data = data.drop(["h_booking_id", "hotel_id", "checkout_date", "hotel_area_code", "hotel_brand_code",
                      "origin_country_code", "hotel_country_code", "hotel_city_code",
                      "hotel_chain_code", "hotel_live_date", "h_customer_id", "customer_nationality",
                      "guest_is_not_the_customer", "guest_nationality_country_name", "language",
                      "original_payment_method", "original_payment_type", "request_nonesmoke",
                      "original_payment_currency", "cancellation_policy_code",
                      "is_first_booking", "no_of_room", "no_of_extra_bed", "no_of_children",
                      "no_of_adults", "is_user_logged_in", "request_latecheckin",
                      "request_highfloor", "request_highfloor", "request_twinbeds", "request_airport",
                      "request_earlycheckin",
                      "accommadation_type_name", "charge_option", "checkin_date"
                      ], axis=1)
    # TODO: remove
    if with_label:
        return data, target, h_booking_id
    else:
        return data, h_booking_id
