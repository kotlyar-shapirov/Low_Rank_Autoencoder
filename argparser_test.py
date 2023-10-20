import argparse

parser = argparse.ArgumentParser(description='testing print')

parser.add_argument('-n', '--number', type=str, default=1, help='number to print')

args = parser.parse_args()

print(args.number)