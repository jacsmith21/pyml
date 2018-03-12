from argparse import ArgumentError, ArgumentParser, SUPPRESS

parser = ArgumentParser()
args = None


# noinspection PyShadowingBuiltins
def add_argument(arg_name, help, flag, nargs='?'):
    try:
        if flag:
            parser.add_argument(arg_name, help=help, action='store_true', default=SUPPRESS)
        else:
            parser.add_argument(arg_name, help=help, action='store', default=SUPPRESS, nargs=nargs)
    except ArgumentError:
        pass


def parse():
    global args
    args = vars(parser.parse_args())


def reset():
    global args
    args = None


def contains(key):
    if args is None:
        parse()

    return key in args


def get(key, default=None):
    if args is None:
        parse()

    return args.get(key, default)
