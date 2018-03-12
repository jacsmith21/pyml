import parser

RUN = 0
TEST = 1

ctx = RUN


def set_test():
    global ctx
    ctx = TEST

    parser.add_argument(
        'test_config',
        help=None,
        flag=False,
        nargs='?'
    )


def in_test():
    return ctx == TEST

