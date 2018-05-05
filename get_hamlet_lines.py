import csv

in_filename = 'shakespeare-plays/Shakespeare_data.csv'
out_filename = 'Hamlet-lines.txt'

count = 0

f_in = open(in_filename, 'rb')
f_out = open(out_filename, 'w')
reader = csv.reader(f_in)

prev_player = None

for row in reader:
    if row[4] == 'HAMLET':
        if row[4] != prev_player:
            f_out.write('----------------\n')
        count += 1
        f_out.write(row[5] + '\n')
    prev_player = row[4]



