import argparse
import os.path


def is_valid_file(argparser, arg):
    """
    ref: https://stackoverflow.com/questions/11540854/file-as-command-line-argument-for-argparse-error-message-if-argument-is-not-va
    """
    if arg is None: return None
    fpath = os.path.abspath(arg)
    if not os.path.exists(fpath):
        argparser.error("Could not find the file %s." % arg)
    else:
        return fpath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classification Using Linear Discriminant Functions and Boosting Algorithms',
        fromfile_prefix_chars='@',
        # allow_abbrev=True
    )

    # Positional Arguments
    parser.add_argument('method', nargs='?', metavar='classifier_method', type=str, help="Classification method.")
    parser.add_argument('training_file', nargs='?', type=lambda x: is_valid_file(parser, x), default=None, help='Training data filepath')
    parser.add_argument('testing_file', nargs='?', type=lambda x: is_valid_file(parser, x), default=None, help='Testing data filepath')

    # Flag Arguments
    parser.add_argument('-c', '--cache', action='store_true', default=False, help="Cache trained classifiers.")
    parser.add_argument('-i', '--prompt', action='store_true', default=False, help="Use interactive prompt.")
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Detailed output and debugging information')

    # Parse
    args = parser.parse_args()

    # Run
    import proj2

    if args.prompt and args.method is None:
        print("Choose a classification method, options are:\n\t - %s" % '\n\t - '.join(proj2.METHODS))
        args.method = input("Method: ")

    proj2.run(**vars(args))
