def sort_features(feature_importances, columns):
    important_features_dict = {}
    for x, i in enumerate(feature_importances):
        important_features_dict[columns[x]] = i

    important_features_list = sorted(important_features_dict,
                                     key=important_features_dict.get,
                                     reverse=True)

    return important_features_list
