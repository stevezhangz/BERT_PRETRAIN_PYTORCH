import os

paths=[]

for i in os.listdir('dataset'):
    if i.split('.')[-1]=='txt':
        paths.append(i)
if not os.path.exists('seq_pairs'):
    os.mkdir('seq_pairs')

for path in paths:
    with open(os.path.join('dataset',path),'r') as f:
        new_pairs=[]
        allseqs=f.read().split('\n')
        for i in allseqs:
            seq_num=len(i.split('.'))
            if seq_num>1:
                for i in range(seq_num-1):
                    new_pairs.append(allseqs[i]+'\t'+allseqs[i+1])
        with open(os.path.join('seq_pairs',path),'w') as f2:
            f2.write('\n'.join(new_pairs))
        f2.close()
    f.close()

