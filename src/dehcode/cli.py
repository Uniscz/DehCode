import argparse
import json
from dehcode.hardware import detect
from dehcode.core.registry import models
from dehcode.core.recommend import recommend


def main():
    parser = argparse.ArgumentParser(prog='dehcode', description='FREE-FIRST local AI video runtime')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('doctor','models','recommend'):
        sub.add_parser(name).add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.command == 'doctor':
        value = detect()
    elif args.command == 'models':
        value = models()
    else:
        hardware = detect()
        value = [recommend(hardware, m) for m in models()]
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
