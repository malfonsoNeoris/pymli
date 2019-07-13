from sklearn.utils.validation import check_is_fitted


def only_fitted(props):
    def decorator(func):
        def wrapper(*args, **kwargs):
            check_is_fitted(args[0], props)
            return func(*args, **kwargs)
        return wrapper
    return decorator
