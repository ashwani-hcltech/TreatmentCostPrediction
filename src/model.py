from sklearn.linear_model import LinearRegression, Lasso, Ridge


#  Linear Regression
def build_linear_regression():
    return LinearRegression()


#  Lasso (Feature selection model)
def build_lasso():
    return Lasso(
        alpha=0.01,
        max_iter=5000,
        random_state=42
    )


#  Ridge (Best for multicollinearity)
def build_ridge(alpha=1.0):
    return Ridge(
        alpha=alpha,
        random_state=42
    )


#  Model selector 
def get_model(model_type="linear"):
    
    if model_type == "linear":
        return build_linear_regression()
    
    elif model_type == "lasso":
        return build_lasso()
    
    elif model_type == "ridge":
        return build_ridge()
    
    else:
        raise ValueError("Invalid model type. Choose: linear, lasso, ridge")