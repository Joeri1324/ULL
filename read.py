def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_simlex(file_name):
    parse_line = lambda line: {'word1': line[0], 'word2': line[1], 'similarity': float(line[3])} if is_number(line[3]) else False

    with open(file_name, 'r') as file:
        return [
            parse_line(line.split())
            for line in file
        ]