def peel_decorations(decorated_function):
    undecorated_function = decorated_function
    while hasattr(undecorated_function, "__wrapped__"):
        undecorated_function = undecorated_function.__wrapped__

    return undecorated_function
