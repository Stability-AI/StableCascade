import sys
import argparse
from .. import WarpCore
from .. import templates


def template_init(args):
    return ''''


    '''.strip()


def init_template(args):
    parser = argparse.ArgumentParser(description='WarpCore template init tool')
    parser.add_argument('-t', '--template', type=str, default='WarpCore')
    args = parser.parse_args(args)

    if args.template == 'WarpCore':
        template_cls = WarpCore
    else:
        try:
            template_cls = __import__(args.template)
        except ModuleNotFoundError:
            template_cls = getattr(templates, args.template)
    print(template_cls)


def main():
    if len(sys.argv) < 2:
        print('Usage: core <command>')
        sys.exit(1)
    if sys.argv[1] == 'init':
        init_template(sys.argv[2:])
    else:
        print('Unknown command')
        sys.exit(1)


if __name__ == '__main__':
    main()
