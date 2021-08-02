import os 

for textfile in [x for x in os.listdir('./new_data') if '.txt' in x]:
    content = []
    with open(os.path.join('new_data',textfile)) as f:
        for line in f:
            splits = line.split(' ')
            cls, x, y, width, height = splits[0:5]
            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)
            x = (2*x + width) / 2.0
            y = (2 * y + height) /2.0
            content.append('{} {} {} {} {}'.format(cls, x,y,width,height))
    with open(os.path.join('new_data', textfile), 'w') as f:
        for c in content:
            f.write(c)
            f.write('\n')


