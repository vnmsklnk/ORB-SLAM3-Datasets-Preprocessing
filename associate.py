import argparse
import os


def read_file_list(filename, isRGB):
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    lines = [line for line in lines if line != '']
    lines = list(map(float, lines))
    files = None
    if isRGB:
        files = os.listdir(os.path.join(os.path.dirname(filename), "color"))
    else:
        files = os.listdir(os.path.join(os.path.dirname(filename), "depth"))
    if isRGB:
        files = [os.path.join("color", x) for x in files]
    else:
        files = [os.path.join("depth", x) for x in files]
    files.sort()
    d = dict(zip(lines, files))
    return d


def associate(first_list, second_list,offset,max_difference):
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    This script takes two data files with timestamps and associates them   
    ''')
    parser.add_argument('first_file', help='first text file (format: timestamp data)')
    parser.add_argument('second_file', help='second text file (format: timestamp data)')
    parser.add_argument('--first_only', help='only output associated lines from first file', action='store_true')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 1000)',default=1000)
    args = parser.parse_args()

    first_list = read_file_list(args.first_file, True)
    second_list = read_file_list(args.second_file, False)

    matches = associate(first_list, second_list,float(args.offset),float(args.max_difference))   

    if args.first_only:
        for a,b in matches:
            print("%f %s"%(a," ".join(first_list[a])))
    else:
        for a,b in matches:
            print(a / 1000000, first_list[a], b / 1000000, second_list[b])
            
        
