def abstract(func):
    # noinspection PyUnusedLocal
    # pylint: disable=unused-argument
    def wrapper(*args, **kwargs):
        raise NotImplementedError('{} has not been implemented'.format(func.__name__))

    func.__isabstractmethod__ = True
    return wrapper
