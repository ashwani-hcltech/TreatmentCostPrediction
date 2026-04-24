from sklearn.linear_model import LinearRegression, Lasso, Ridge


def build_linear_regression():
    return LinearRegression()


def build_lasso():
    return Lasso(
        # remove unnecessary features coefficients
        alpha=0.01,         

        # permission for interation 
        # lasso use the iterative algorithm
        max_iter=5000,

        # to find the productive result
        random_state=2
    )


def ridge_regression_model(alpha=1.0):
    
    return Ridge(alpha=alpha, random_state=42)



