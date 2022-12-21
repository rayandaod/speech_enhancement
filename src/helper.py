
# Converts a dictionary of tensors to a dictionary of strings and take care of internal dictionaries as well
def dict_of_tensors_to_dict_of_strings(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_of_tensors_to_dict_of_strings(value)
        else:
            d[key] = value.numpy()
    return d