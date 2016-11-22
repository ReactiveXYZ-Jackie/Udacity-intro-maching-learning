#!/usr/bin/python
import math

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here

    residuals = {
        x: abs(net_worths[x][0]-predictions[x][0]) for x in range(len(ages))
    }

    residuals_sorted_keys = sorted(residuals, key = residuals.get)

    key_len = len(residuals_sorted_keys)

    num_to_remove = math.floor(key_len / 10)

    num_to_slice = int(key_len - num_to_remove + 1)

    residuals_sorted_keys = residuals_sorted_keys[:num_to_slice]

    for key in residuals_sorted_keys:
        cleaned_data.append((ages[key], net_worths[key], residuals[key]));

    return cleaned_data

