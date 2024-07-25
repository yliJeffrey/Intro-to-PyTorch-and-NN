# Part 2
# Linear Regression Review


def predict_rent(sz_sqft, bedrooms):
    # sz_sqft and bedrooms are features; 3 and 10 are weights; 250 is bias
    predicted_rent = 3 * sz_sqft + 10 * bedrooms + 250
    return predicted_rent

